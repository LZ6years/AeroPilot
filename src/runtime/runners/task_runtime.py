"""Long-lived facade for sequential model-driven UAV tasks."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator

from src.agent.config import ToolLoopConfig
from src.agent.mission_state import MissionState
from src.agent.safety import SafetySupervisor
from src.agent.transcript import AgentTranscript
from src.env_client import EnvClient, SceneInstance
from src.env_client.errors import SceneRecoveryRequired
from src.memory.context.builder import ModelContextBuilder
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.configs.task_execution_options import TaskExecutionOptions
from src.runtime.context import ContextCheckpointService
from src.runtime.control.stabilization import UavStabilizer
from src.runtime.middleware import ToolTelemetryMiddleware
from src.runtime.task.evidence import FlightEvidenceRecorder
from src.runtime.task.journal import RuntimeJournal
from src.runtime.task.loop import TaskLoop
from src.runtime.task.physical_window import (
    PHYSICAL_TRACE_SCHEMA_VERSION,
    PhysicalWindowExecutor,
)
from src.runtime.task.scene_lifecycle import SceneLifecycleService
from src.runtime.task.session import TaskSession
from src.runtime.task.turn_executor import TurnExecutor
from src.tools.dispatcher import ToolDispatcher


class TaskRuntimeBusyError(RuntimeError):
    """Raised when one Runtime is already mutating its scene state."""


class TaskRuntime:
    """Facade that creates an isolated TaskSession for every ``run_task``."""

    SHUTDOWN_POLICIES = {"hover", "land", "none"}
    SCENE_POLICIES = {"isolated", "reuse_running"}

    def __init__(
        self,
        env_client: EnvClient,
        recorder: Any,
        planner: Any,
        context_builder: ModelContextBuilder,
        tool_dispatcher: ToolDispatcher,
        knowledge_service_factory: Callable[[], Any],
        safety: SafetySupervisor | None = None,
        shutdown_policy: str = "hover",
        stabilization_config: UavStabilizationConfig | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        runtime_loop_guard_windows: int = 300,
        tool_loop_config: ToolLoopConfig | None = None,
        read_only_execution_config: ReadOnlyExecutionConfig | None = None,
    ) -> None:
        if shutdown_policy not in self.SHUTDOWN_POLICIES:
            raise ValueError(f"Unsupported shutdown policy: {shutdown_policy}")
        if (
            isinstance(runtime_loop_guard_windows, bool)
            or not isinstance(runtime_loop_guard_windows, int)
            or runtime_loop_guard_windows < 1
        ):
            raise ValueError(
                "runtime_loop_guard_windows must be a positive integer"
            )
        self.env_client = env_client
        self.recorder = recorder
        self.planner = planner
        self.context_builder = context_builder
        self.tools = tool_dispatcher
        self.safety = safety or SafetySupervisor()
        self._shutdown_policy = shutdown_policy
        self.runtime_loop_guard_windows = runtime_loop_guard_windows
        self.tool_loop_config = (
            tool_loop_config
            or getattr(planner, "tool_loop_config", None)
            or ToolLoopConfig()
        )
        self.read_only_execution_config = (
            read_only_execution_config or ReadOnlyExecutionConfig()
        )
        self.knowledge_service_factory = knowledge_service_factory

        self._operation_lock = threading.Lock()
        self._memory_writer = MemoryEventWriter(recorder.run_dir)
        self._memory_reader = MemoryRecordReader(recorder.run_dir)
        self._journal = RuntimeJournal(recorder, event_sink)
        self._stabilizer = UavStabilizer(
            self.tools, recorder, stabilization_config
        )
        self._tool_telemetry = ToolTelemetryMiddleware(
            self.tools, self._journal.debug_event
        )
        self._evidence = FlightEvidenceRecorder(
            recorder,
            self.tools,
            self._memory_writer,
            self._journal,
        )
        self._physical_windows = PhysicalWindowExecutor(
            self.tools,
            self._tool_telemetry,
            self._stabilizer,
            self._evidence,
            self._journal,
            recorder,
        )
        self._turn_executor = TurnExecutor(
            self.tools,
            self.safety,
            self._physical_windows,
            self._evidence,
            self._journal,
            recorder,
            self.read_only_execution_config,
            self._tool_telemetry,
        )
        self._task_loop = TaskLoop(
            self.planner,
            self.tools.registry,
            self._turn_executor,
            self._journal,
            self._evidence,
            max_consecutive_protocol_rejections=(
                self.tool_loop_config.max_consecutive_protocol_rejections
            ),
        )
        self._scenes = SceneLifecycleService(
            self.env_client,
            recorder,
            self.tools,
            self._stabilizer,
            self._evidence,
            self._journal,
            self._memory_writer,
            self._memory_reader,
            shutdown_policy=shutdown_policy,
        )

    @property
    def shutdown_policy(self) -> str:
        """Return the active lifecycle shutdown policy."""

        return self._shutdown_policy

    @shutdown_policy.setter
    def shutdown_policy(self, value: str) -> None:
        """Keep facade-level policy overrides synchronized with lifecycle logic."""

        if value not in self.SHUTDOWN_POLICIES:
            raise ValueError(f"Unsupported shutdown policy: {value}")
        self._shutdown_policy = value
        scenes = getattr(self, "_scenes", None)
        if scenes is not None:
            scenes.shutdown_policy = value

    def run_task(
        self,
        map_id: str,
        instruction: str,
        agent_id: str = "uav_1",
        scene_policy: str = "isolated",
        stop_scene_on_finish: bool = True,
        scene_id: str | None = None,
        execution_options: TaskExecutionOptions | None = None,
        *,
        protocol_context: str | None = None,
    ) -> MissionState:
        """Run one task through a fresh TaskSession and return MissionState."""

        with self._operation():
            primary_error: Exception | None = None
            try:
                return self._run_task(
                    map_id,
                    instruction,
                    agent_id,
                    scene_policy,
                    stop_scene_on_finish,
                    scene_id,
                    execution_options,
                    protocol_context=protocol_context,
                )
            except Exception as exc:
                primary_error = exc
                raise
            finally:
                flush = getattr(self.recorder, "flush", None)
                if callable(flush):
                    try:
                        flush()
                    except Exception as flush_error:
                        if primary_error is None:
                            raise
                        _attach_cleanup_error(
                            primary_error, flush_error, "artifact_flush"
                        )

    def _run_task(
        self,
        map_id: str,
        instruction: str,
        agent_id: str,
        scene_policy: str,
        stop_scene_on_finish: bool,
        scene_id: str | None,
        execution_options: TaskExecutionOptions | None,
        *,
        protocol_context: str | None,
    ) -> MissionState:
        if scene_policy not in self.SCENE_POLICIES:
            raise ValueError(f"Unsupported scene policy: {scene_policy}")
        options = self._resolve_execution_options(execution_options)
        state = MissionState(
            run_id=self.recorder.run_dir.name,
            user_instruction=instruction,
            protocol_context=protocol_context or "",
            observation_modalities=options.observation_modalities,
            max_physical_windows=self.runtime_loop_guard_windows,
            max_nonphysical_rounds=self.tool_loop_config.max_rounds,
        )
        self._journal.reset_instruction()
        hydrator = self._media_hydrator()
        if hydrator is not None:
            hydrator.clear()

        scene: SceneInstance | None = None
        session: TaskSession | None = None
        primary_error: Exception | None = None
        instruction_ready = False
        try:
            scene = self._scenes.acquire(map_id, scene_policy, scene_id)
            state.connected = self._scenes.is_preflighted(scene)
            state.airborne = bool(
                state.connected
                and scene.capabilities
                and scene.capabilities.requires_takeoff
            )
            memory = self._scenes.resolve_memory(
                map_id,
                scene,
                agent_id,
                scene_policy,
                instruction,
                state.run_id,
            )
            transcript = AgentTranscript(self._memory_writer)
            checkpoints = ContextCheckpointService(
                self.context_builder, transcript, self.recorder
            )
            session = TaskSession(
                scene=scene,
                agent_id=agent_id,
                state=state,
                memory=memory,
                transcript=transcript,
                checkpoints=checkpoints,
                knowledge_service=self.knowledge_service_factory(),
            )
            checkpoints.begin_instruction(state, memory)
            self._journal.event(
                "instruction_started",
                {
                    "instruction": instruction,
                    "instruction_id": memory.active_instruction_id,
                    "protocol_context": protocol_context,
                    "loop_schema": "unified_tool_loop_v2",
                    "physical_trace_schema": PHYSICAL_TRACE_SCHEMA_VERSION,
                },
            )
            instruction_ready = True
            self._scenes.prepare(session, options)
            checkpoints.publish(state, memory, "instruction_start")
            self._task_loop.run(session)
            self._scenes.finalize_instruction(session)
            if stop_scene_on_finish:
                self._scenes.shutdown(session)
            self._journal.event(
                "instruction_completed",
                self._evidence.state_summary(state),
            )
            return state
        except Exception as exc:
            primary_error = exc
            if isinstance(exc, SceneRecoveryRequired) and scene is not None:
                self._scenes.mark_unusable(scene)
            state.record_error(type(exc).__name__, str(exc))
            if session is not None:
                try:
                    self._evidence.write_state(session)
                    self._journal.event(
                        "instruction_failed",
                        {"error_type": type(exc).__name__, "error": str(exc)},
                    )
                except Exception as audit_error:
                    _attach_cleanup_error(exc, audit_error, "failure_audit")
                if (
                    instruction_ready
                    and not any(
                        item.kind == "task_terminal"
                        for item in session.transcript.items
                    )
                ):
                    try:
                        self._scenes.finalize_instruction(
                            session,
                            scene_unusable=isinstance(
                                exc, SceneRecoveryRequired
                            ),
                        )
                    except Exception as final_error:
                        _attach_cleanup_error(
                            exc, final_error, "final_observation"
                        )
            raise
        finally:
            if session is not None:
                session.history_compaction_coordinator.abandon()
            if stop_scene_on_finish and scene is not None:
                try:
                    self._scenes.stop(scene)
                except Exception as cleanup_error:
                    try:
                        self._journal.event(
                            "scene_cleanup_failed",
                            {
                                "error_type": type(cleanup_error).__name__,
                                "error": str(cleanup_error),
                            },
                        )
                    except Exception as audit_error:
                        _attach_cleanup_error(
                            cleanup_error, audit_error, "cleanup_audit"
                        )
                    if primary_error is None:
                        raise
                    _attach_cleanup_error(
                        primary_error, cleanup_error, "scene_stop"
                    )
            if hydrator is not None:
                hydrator.clear()

    def ensure_reused_scene(
        self, map_id: str, scene_id: str | None = None
    ) -> SceneInstance:
        with self._operation():
            return self._scenes.acquire(map_id, "reuse_running", scene_id)

    def safe_hover_reused_scene(
        self,
        map_id: str,
        agent_id: str = "uav_1",
        scene_id: str | None = None,
    ) -> None:
        with self._operation():
            self._scenes.safe_hover_reused_scene(map_id, agent_id, scene_id)

    def finish_reused_scene(
        self,
        map_id: str,
        state: MissionState | None,
        agent_id: str = "uav_1",
        scene_id: str | None = None,
    ) -> None:
        with self._operation():
            self._scenes.finish_reused_scene(
                map_id, state, agent_id, scene_id
            )

    @staticmethod
    def _resolve_execution_options(
        execution_options: TaskExecutionOptions | None,
    ) -> TaskExecutionOptions:
        if execution_options is None:
            return TaskExecutionOptions()
        if not isinstance(execution_options, TaskExecutionOptions):
            raise ValueError(
                "execution_options must be a TaskExecutionOptions or None"
            )
        return execution_options

    def _media_hydrator(self) -> Any | None:
        projector = getattr(self.planner, "history_projector", None)
        return getattr(
            getattr(projector, "assembler", None), "media_hydrator", None
        )

    @contextmanager
    def _operation(self) -> Iterator[None]:
        if not self._operation_lock.acquire(blocking=False):
            raise TaskRuntimeBusyError(
                "TaskRuntime is already executing a scene-mutating operation"
            )
        try:
            yield
        finally:
            self._operation_lock.release()


def _attach_cleanup_error(primary: Exception, cleanup: Exception, stage: str) -> None:
    """Retain cleanup evidence without replacing the primary failure."""

    errors = list(getattr(primary, "cleanup_errors", []))
    errors.append(
        {
            "stage": stage,
            "error_type": type(cleanup).__name__,
            "error": str(cleanup),
        }
    )
    primary.cleanup_errors = errors
