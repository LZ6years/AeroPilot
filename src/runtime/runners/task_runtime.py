"""Run semantic tasks through short-horizon planning windows."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from collections.abc import Callable
from typing import Any
from src.agent.mission_state import MissionState
from src.agent.safety import SafetySupervisor
from src.agent.transcript import AgentTranscript, AssistantTurn
from src.env_client import EnvClient, SceneInstance
from src.env_client.errors import SceneRecoveryRequired
from src.llm.audit import redact_provider_value
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.memory.context.builder import ModelContextBuilder
from src.perception.raw_observation import RawObservation
from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
from src.runtime.control.stabilization import UavStabilizer
from src.agent.config import ToolLoopConfig
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.configs.task_execution_options import InitialPose, TaskExecutionOptions
from src.runtime.middleware import ToolTelemetryMiddleware
from src.runtime.context import ContextCheckpointService
from src.runtime.task import RuntimeJournal
from src.runtime.task.loop import InstructionLoop, TaskLoopGuard
from src.runtime.task.physical_window import (
    PHYSICAL_TRACE_SCHEMA_VERSION,
    PhysicalWindowExecutor,
)
from src.runtime.task.scene_session import SceneSessionService
from src.runtime.task.session import InstructionSession
from src.runtime.task.turn_executor import TurnExecutor
from src.tools.dispatcher import ToolDispatcher
from src.uav import clean_pose_for_model

class TaskRuntime:
    SHUTDOWN_POLICIES = {"hover", "land", "none"}
    SCENE_POLICIES = {"isolated", "reuse_running"}

    def __init__(
        self, env_client: EnvClient, recorder, planner,
        context_builder: ModelContextBuilder,
        tool_dispatcher: ToolDispatcher,
        knowledge_service_factory: Callable[[], Any],
        safety: SafetySupervisor | None = None, shutdown_policy: str = "hover",
        stabilization_config: UavStabilizationConfig | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        runtime_loop_guard_windows: int = 300,
        tool_loop_config: ToolLoopConfig | None = None,
        read_only_execution_config: ReadOnlyExecutionConfig | None = None,
    ) -> None:
        if shutdown_policy not in self.SHUTDOWN_POLICIES:
            raise ValueError(f"Unsupported shutdown policy: {shutdown_policy}")
        if isinstance(runtime_loop_guard_windows, bool) or not isinstance(runtime_loop_guard_windows, int) or runtime_loop_guard_windows < 1:
            raise ValueError("runtime_loop_guard_windows must be a positive integer")
        self.env_client, self.recorder = env_client, recorder
        self.tool_loop_config = (
            tool_loop_config
            or getattr(planner, "tool_loop_config", None)
            or ToolLoopConfig()
        )
        self.read_only_execution_config = read_only_execution_config or ReadOnlyExecutionConfig()
        self.planner = planner
        self.safety = safety or SafetySupervisor()
        self.shutdown_policy = shutdown_policy
        self.runtime_loop_guard_windows = runtime_loop_guard_windows
        self.tools = tool_dispatcher
        self.knowledge_service_factory = knowledge_service_factory
        self.stabilizer = UavStabilizer(self.tools, recorder, stabilization_config)
        self.journal = RuntimeJournal(recorder, event_sink)
        self.memory_writer = MemoryEventWriter(recorder.run_dir)
        self.memory_reader = MemoryRecordReader(recorder.run_dir)
        self.agent_transcript = AgentTranscript(self.memory_writer)
        self.context_builder = context_builder
        self.checkpoints = ContextCheckpointService(
            self.context_builder, self.agent_transcript, recorder
        )
        self.tool_telemetry = ToolTelemetryMiddleware(self.tools, self._debug_event)
        self.loop_guard = TaskLoopGuard(
            self.tool_loop_config.max_consecutive_protocol_rejections
        )
        self.physical_windows = PhysicalWindowExecutor(
            self, self._pose_from_observation
        )
        self.turn_executor = TurnExecutor(self, self.physical_windows)
        self.instruction_loop = InstructionLoop(
            self, self.loop_guard, self.turn_executor
        )
        self._active_flight_memories = {}
        self._state_flight_memories = {}
        self.scene_sessions = SceneSessionService(
            self.env_client,
            record_contract=lambda scene: self._record_scene_contract(scene),
            debug_event=self._debug_event,
            on_forget=self._forget_scene_memory,
        )
        self._current_turn_agent_messages = []
        self._decision_index = 0
        self.journal.reset_instruction()

    def run_task(
        self, map_id: str, instruction: str, agent_id: str = "uav_1",
        scene_policy: str = "isolated", stop_scene_on_finish: bool = True,
        scene_id: str | None = None, execution_options: TaskExecutionOptions | None = None,
        *, protocol_context: str | None = None,
    ) -> MissionState:
        """Execute raw intent through the single model/tool loop; never parse phases."""
        if scene_policy not in self.SCENE_POLICIES:
            raise ValueError(f"Unsupported scene policy: {scene_policy}")
        options = self._resolve_execution_options(execution_options)
        state = MissionState(
            run_id=self.recorder.run_dir.name, user_instruction=instruction,
            protocol_context=protocol_context or "",
            observation_modalities=options.observation_modalities,
            max_physical_windows=self.runtime_loop_guard_windows,
            max_nonphysical_rounds=self.tool_loop_config.max_rounds)
        self._decision_index = 0
        self._current_turn_agent_messages = []
        self.journal.reset_instruction()
        projector = getattr(self.planner, "history_projector", None)
        hydrator = getattr(getattr(projector, "assembler", None), "media_hydrator", None)
        if hydrator is not None:
            hydrator.clear()
        self.tools.knowledge_service = self.knowledge_service_factory()
        scene = self.scene_sessions.acquire(map_id, scene_policy, scene_id)
        instruction_session: InstructionSession | None = None
        primary_error = None
        instruction_ready = False
        try:
            state.connected = self.scene_sessions.is_preflighted(scene)
            state.airborne = bool(state.connected and scene.capabilities and scene.capabilities.requires_takeoff)
            memory = self._resolve_flight_memory(map_id, scene, agent_id, scene_policy, instruction, state)
            self._state_flight_memories[state.run_id] = memory
            instruction_session = InstructionSession(
                scene=scene,
                agent_id=agent_id,
                state=state,
                memory=memory,
                transcript=self.agent_transcript,
            )
            self.checkpoints.begin_instruction(state, memory)
            self._event("instruction_started", {
                "instruction": instruction, "instruction_id": memory.active_instruction_id,
                "protocol_context": protocol_context,
                "loop_schema": "unified_tool_loop_v2", "physical_trace_schema": PHYSICAL_TRACE_SCHEMA_VERSION})
            instruction_ready = True
            self.scene_sessions.assert_session_usable(instruction_session)
            self._validate_execution_options(scene, options)
            self._prepare_scene_for_task(scene, agent_id, state, options)
            self.checkpoints.publish(state, memory, "instruction_start")
            self.instruction_loop.run(instruction_session)
            # Terminal evidence precedes any optional land/scene cleanup. A
            # refresh is not a formal window and never changes motion accounting.
            self._finalize_instruction(scene, agent_id, state)
            if stop_scene_on_finish:
                self._run_shutdown(scene, agent_id, state)
            self._event("instruction_completed", self._state_summary(state))
            return state
        except Exception as exc:
            primary_error = exc
            if isinstance(exc, SceneRecoveryRequired):
                if instruction_session is not None:
                    self.scene_sessions.mark_session_unusable(
                        instruction_session
                    )
                else:
                    self.scene_sessions.mark_unusable(scene)
            state.record_error(type(exc).__name__, str(exc))
            try:
                self._write_state(state)
                self._event("instruction_failed", {"error_type": type(exc).__name__, "error": str(exc)})
            except Exception as audit_error:
                _attach_cleanup_error(exc, audit_error, "failure_audit")
            # Preserve model outcome separately from an infrastructure failure.
            # Best-effort final evidence is frozen by the caller only after this
            # method has completed all scene cleanup.
            if instruction_ready and not any(item.kind == "task_terminal" for item in self.agent_transcript.items):
                try:
                    self._finalize_instruction(scene, agent_id, state, scene_unusable=isinstance(exc, SceneRecoveryRequired))
                except Exception as final_error:
                    _attach_cleanup_error(exc, final_error, "final_observation")
            raise
        finally:
            if stop_scene_on_finish:
                try:
                    if instruction_session is not None:
                        self.scene_sessions.stop_session(instruction_session)
                    else:
                        self.scene_sessions.stop(scene)
                except Exception as cleanup_error:
                    try:
                        self._event("scene_cleanup_failed", {"error_type": type(cleanup_error).__name__, "error": str(cleanup_error)})
                    except Exception as audit_error:
                        _attach_cleanup_error(cleanup_error, audit_error, "cleanup_audit")
                    if primary_error is None:
                        raise
                    _attach_cleanup_error(primary_error, cleanup_error, "scene_stop")
            if hydrator is not None:
                hydrator.clear()

    def latest_turn_agent_messages(self) -> list[dict[str, Any]]:
        """Return an isolated UI snapshot for InteractiveTaskSession."""

        return copy.deepcopy(self._current_turn_agent_messages)

    def _finalize_instruction(self, scene, agent_id, state, *, scene_unusable=False) -> None:
        """Close the instruction with real final evidence; preserve cleanup errors."""
        failures = []
        if state.connected and not scene_unusable:
            if scene.capabilities is None or scene.capabilities.supports("hover"):
                try:
                    result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
                    self._record_action(state, "uav.hover", {}, result.to_dict())
                    if result.execution_state.get("requires_scene_reset"):
                        scene_unusable = True
                        raise SceneRecoveryRequired(result.message)
                    if not result.ok:
                        raise RuntimeError(result.message)
                except Exception as exc:
                    failures.append(exc)
            try:
                if not scene_unusable:
                    self._record_observation(scene, agent_id, state, list(state.observation_modalities), "task_terminal")
                    self.checkpoints.publish(
                        state, self._memory_for_state(state), "task_terminal"
                    )
            except Exception as exc:
                failures.append(exc)
        terminal = {**self._state_summary(state),
                    "cleanup_errors": [{"error_type": type(e).__name__, "error": str(e)} for e in failures]}
        self.agent_transcript.append_task_terminal(terminal)
        self.journal.write_terminal(terminal)
        self._write_state(state)
        if failures:
            raise failures[0]

    def finish_reused_scene(
        self,
        map_id: str,
        state: MissionState | None,
        agent_id: str = "uav_1",
        scene_id: str | None = None,
    ) -> None:
        # Apply final shutdown policy and stop a reused scene owned by this runtime.
        scene = self.scene_sessions.reused(map_id, scene_id)
        if scene is None:
            self._debug_event("scene_stop_skipped", {"map_id": map_id, "scene_id": scene_id, "reason": "not_running"})
            return
        primary_error = None
        try:
            if state is not None:
                self._run_shutdown(scene, agent_id, state)
        except Exception as exc:
            primary_error = exc
            raise
        finally:
            try:
                self.scene_sessions.stop(scene)
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                _attach_cleanup_error(primary_error, cleanup_error, "scene_stop")

    def safe_hover_reused_scene(self, map_id: str, agent_id: str = "uav_1", scene_id: str | None = None) -> None:
        # Best-effort hover after an instruction failure in a reused scene.
        scene = self.scene_sessions.reused(map_id, scene_id)
        if scene is None or self.scene_sessions.is_unusable(scene):
            self._debug_event("continuous_error_hover_skipped", {"reason": "not_running"})
            return
        if scene.capabilities and not scene.capabilities.supports("hover"):
            self._debug_event(
                "continuous_error_hover_skipped",
                {
                    "reason": "backend_capability",
                    "protocol": scene.capabilities.protocol,
                    "control_mode": scene.capabilities.control_mode,
                },
            )
            return
        try:
            result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            self._debug_event("continuous_error_hover_completed", result.to_dict())
        except Exception as exc:
            self._debug_event("continuous_error_hover_failed", {"error_type": type(exc).__name__, "error": str(exc)})

    def ensure_reused_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        """Start or reuse the interactive scene before the first instruction.

        This only opens/prepares the simulator scene. Preflight connect, takeoff,
        and initial observation still happen inside the first task instruction.
        """

        return self.scene_sessions.acquire(map_id, "reuse_running", scene_id)

    def _record_scene_contract(self, scene: SceneInstance) -> None:
        """Persist deployment-neutral scene facts used by Runtime."""

        self.recorder.write_json("state/scene_instance.json", scene.to_dict())
        if scene.capabilities is not None:
            self.recorder.write_json(
                "state/scene_capabilities.json",
                {
                    "protocol": scene.capabilities.protocol,
                    "control_mode": scene.capabilities.control_mode,
                    "supported_actions": list(
                        scene.capabilities.supported_actions
                    ),
                    "observation_modalities": list(
                        scene.capabilities.observation_modalities
                    ),
                    "requires_takeoff": scene.capabilities.requires_takeoff,
                    "supports_soft_reset": (
                        scene.capabilities.supports_soft_reset
                    ),
                },
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

    @staticmethod
    def _validate_execution_options(
        scene: SceneInstance,
        options: TaskExecutionOptions,
    ) -> None:
        """Fail before planning when a scene cannot satisfy the task contract."""

        capabilities = scene.capabilities
        if capabilities is None:
            # EnvClient v1 permits compatible replay/test clients without a
            # capability snapshot. Preserve that structural compatibility and
            # let the concrete observation/action call report unsupported work.
            return
        supported_modalities = {
            str(value).strip().lower()
            for value in capabilities.observation_modalities
        }
        unsupported = [
            value
            for value in options.observation_modalities
            if value not in supported_modalities
        ]
        if unsupported:
            raise RuntimeError(
                "Scene does not support requested observation modalities: "
                + ", ".join(unsupported)
            )
        if (
            options.initial_pose is not None
            and not capabilities.supports("set_pose")
        ):
            raise RuntimeError(
                "Scene does not support set_pose required by initial_pose"
            )

    def _prepare_scene_for_task(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        options: TaskExecutionOptions,
    ) -> None:
        first_scene_use = not self.scene_sessions.is_preflighted(scene)
        if first_scene_use:
            self._run_preflight(scene, agent_id, state)
        if options.initial_pose is not None:
            self._apply_initial_pose(scene, agent_id, state, options.initial_pose)
        if first_scene_use:
            # Hold only after an episode pose has replaced the packaged scene's
            # default pose. This prevents a visible/default-pose hover followed
            # by a large camera jump at the start of an evaluation episode.
            self._run_preflight_hover(scene, agent_id, state)
            self.scene_sessions.mark_preflighted(scene)

        stabilization = None
        if first_scene_use or options.initial_pose is not None:
            stabilization = self.stabilizer.wait_after_preflight(
                scene.scene_instance_id,
                agent_id,
                capabilities=scene.capabilities,
                warmup_modalities=list(state.observation_modalities),
            )
            stabilization_event = (
                "preflight_stabilization_completed"
                if first_scene_use
                else "episode_initialization_stabilization_completed"
            )
            self._debug_event(stabilization_event, stabilization)
        if first_scene_use:
            self._event(
                "preflight_completed",
                {
                    "scene_instance_id": scene.scene_instance_id,
                    "stabilization": self._stabilization_summary(stabilization),
                },
            )
        source = "preflight" if first_scene_use else "instruction_start"
        observation_data = self._record_observation(
            scene,
            agent_id,
            state,
            list(state.observation_modalities),
            source=source,
            overwrite_task_initial=options.initial_pose is not None,
        )
        if first_scene_use:
            self._debug_event("initial_observation_updated", observation_data)
        else:
            self._debug_event(
                "instruction_start_observation_updated", observation_data
            )
        self._event(
            "observation_updated",
            self._observation_summary_event(observation_data, source=source),
        )
        self._write_state(state)

    def _run_preflight(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        connect_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.connect", {})
        connect_data = connect_result.to_dict()
        self._record_action(state, "uav.connect", {}, connect_data)
        self._debug_event("control_connected", connect_data)
        if not connect_result.ok:
            raise RuntimeError(f"Preflight connect failed: {connect_result.message}")

        capabilities = scene.capabilities
        if capabilities is None or capabilities.requires_takeoff:
            self._debug_event("preflight_takeoff_started", {"scene_instance_id": scene.scene_instance_id})
            takeoff_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.takeoff", {})
            takeoff_data = takeoff_result.to_dict()
            self._record_action(state, "uav.takeoff", {}, takeoff_data)
            self._debug_event("preflight_takeoff_completed", takeoff_data)
            if not takeoff_result.ok:
                raise RuntimeError(f"Preflight takeoff failed: {takeoff_result.message}")
        else:
            self._debug_event(
                "preflight_takeoff_skipped",
                {
                    "protocol": capabilities.protocol,
                    "control_mode": capabilities.control_mode,
                },
            )

    def _run_preflight_hover(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
    ) -> None:
        """Hold the initialized pose after connect/takeoff and set_pose."""

        capabilities = scene.capabilities
        if capabilities is None or capabilities.supports("hover"):
            hover_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            hover_data = hover_result.to_dict()
            self._record_action(state, "uav.hover", {}, hover_data)
            self._debug_event("preflight_hover_completed", hover_data)
            if not hover_result.ok:
                raise RuntimeError(f"Preflight hover failed: {hover_result.message}")
        else:
            self._debug_event(
                "preflight_hover_skipped",
                {
                    "reason": "backend_capability",
                    "protocol": capabilities.protocol,
                    "control_mode": capabilities.control_mode,
                },
            )

    def _apply_initial_pose(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        initial_pose: InitialPose,
    ) -> None:
        """Place the UAV before the first episode observation, outside planning."""

        args = initial_pose.to_action_args()
        payload = {
            "scene_instance_id": scene.scene_instance_id,
            "agent_id": agent_id,
            "pose": initial_pose.to_dict(),
        }
        self._debug_event("episode_initial_pose_started", payload)
        try:
            result = self.tools.call(
                scene.scene_instance_id,
                agent_id,
                "uav.set_pose",
                args,
            )
        except Exception as exc:
            self._debug_event(
                "episode_initial_pose_failed",
                {
                    **payload,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        result_data = result.to_dict()
        if not result.ok:
            self._debug_event(
                "episode_initial_pose_failed",
                {**payload, "result": result_data},
            )
            raise RuntimeError(
                f"Episode initial pose failed: {result.message}"
            )
        # The placement is initialization, not Agent flight. Reset only recent
        # flight references so the planner starts from the post-placement
        # observation while append-only audit records remain intact.
        memory = self._memory_for_state(state)
        memory.reset_pose_epoch(active_instruction=state.user_instruction)
        self._debug_event(
            "episode_initial_pose_completed",
            {
                **payload,
                "result": result_data,
                "memory_reset_epoch": memory.reset_epoch,
            },
        )

    def _capture_instruction_exchange(
        self, decision_id: str, *, turn: AssistantTurn | None = None, error_type: str | None = None,
    ) -> None:
        """Preserve actual requests/media before control or physical side effects."""
        record = self._native_model_round_snapshot(self._decision_index, canonical_turn=turn)
        record["decision_id"] = decision_id
        if error_type:
            record["error_type"] = error_type
        record = self._persist_provider_exchange_media([record])[0]
        instruction_id = self.agent_transcript.instruction_id
        # Decision numbering restarts for interactive instructions sharing a
        # recorder. Namespace by instruction identity to keep prior captures
        # immutable; hash avoids treating arbitrary IDs as filesystem paths.
        namespace = hashlib.sha256(str(instruction_id).encode("utf-8")).hexdigest()[:16]
        debug_ref = f"debug/tool_loop/{namespace}/{decision_id}.json"
        snapshot = {
            "loop_schema": "unified_tool_loop_v2",
            "instruction_id": instruction_id,
            "decision_id": decision_id, "decision_index": self._decision_index,
            "checkpoint_observation_id": self.agent_transcript.latest_observation_id(),
            "model_context": redact_provider_value(
                self.agent_transcript.latest_checkpoint().to_dict()
                if self.agent_transcript.latest_checkpoint() is not None
                else {}
            ),
            "model_rounds": [record],
        }
        self.recorder.write_json(debug_ref, snapshot)
        # Freezeable append-only association, not a glob over future artifacts.
        # This records a captured proposal, never proof of applied control.
        self._event("model_turn_recorded", {
            "decision_id": decision_id, "debug_ref": debug_ref,
            "instruction_id": instruction_id,
            "loop_schema": "unified_tool_loop_v2",
        })

    def _native_model_round_snapshot(
        self,
        round_index: int,
        *,
        canonical_turn: AssistantTurn | None = None,
    ) -> dict[str, Any]:
        """Copy one public provider exchange without changing planner state."""

        exchange = getattr(self.planner, "last_exchange", None)
        exchange = copy.deepcopy(exchange) if isinstance(exchange, dict) else {}
        record: dict[str, Any] = {
            "round_index": round_index,
            "request": exchange.get("request"),
            "response": exchange.get("response"),
        }
        transport = exchange.get("transport")
        projection = exchange.get("history_projection")
        if isinstance(projection, dict):
            record["history_projection"] = projection
        if isinstance(transport, dict):
            record["transport"] = transport
        media = exchange.get("media")
        if isinstance(media, list):
            record["request_media"] = media
        if canonical_turn is not None:
            record["canonical_turn"] = canonical_turn.to_dict()
        elif not exchange:
            record["capture_status"] = "provider_exchange_unavailable"
        return record

    def _maybe_recharge_visual_search_budget(
        self,
        state: MissionState,
        measurements: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Recharge yaw search only after a verified viewpoint translation.

        Command success alone is insufficient: a blocked vehicle can return a
        successful command without changing the camera viewpoint. The
        post-action observation supplies canonical planar and vertical motion,
        so both horizontal relocation and altitude changes can start a fresh
        stationary search segment.
        """

        try:
            planar_m = abs(float(measurements.get("planar_displacement_m", 0.0) or 0.0))
            vertical_m = abs(float(measurements.get("vertical_delta_m", 0.0) or 0.0))
        except (TypeError, ValueError):
            return None
        translation_3d_m = math.hypot(planar_m, vertical_m)
        threshold_m = float(
            self.safety.safety_config.yaw_scan_recharge_distance_m
        )
        if translation_3d_m < threshold_m:
            return None

        recharge = state.recharge_yaw_scan()
        payload = {
            "phase_key": None,
            "window_index": state.physical_window_count,
            "planar_displacement_m": round(planar_m, 6),
            "vertical_displacement_m": round(vertical_m, 6),
            "translation_3d_m": round(translation_3d_m, 6),
            "recharge_distance_m": threshold_m,
            **recharge,
        }
        self._event("visual_search_budget_recharged", payload)
        return payload

    def _pose_from_observation(self, observation: dict) -> dict:
        pose = observation.get("pose") if isinstance(observation, dict) else None
        return dict(pose) if isinstance(pose, dict) else {}

    def _apply_automatic_observation_policy(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
    ) -> dict:
        modalities = list(state.observation_modalities)
        source = "automatic_physical_window"
        policy = {
            "mode": "automatic",
            "modalities": modalities,
            "sample_timing": "after_each_physical_window",
        }
        self._debug_event("automatic_observation_policy_applied", policy)
        data = self._record_observation(scene, agent_id, state, modalities, source=source)
        self._debug_event("automatic_observation_updated", data)
        self._event("observation_updated", self._observation_summary_event(data, source=source))
        self._write_state(state)
        return data

    def _validate_step_capability(
        self,
        scene: SceneInstance,
        tool_name: str,
    ) -> None:
        """Reject unsupported primitive actions before ToolDispatcher.

        Model planners normally encounter the same restriction in
        ToolContractValidator. This final guard also covers deterministic or
        custom planners that do not use the model contract machinery.
        """

        spec = self.tools.registry.get(tool_name)
        action = spec.metadata.get("backend_action")
        if not isinstance(action, str) or not action or scene.capabilities is None:
            return
        if scene.capabilities.supports(action):
            return
        payload = {
            "tool": tool_name,
            "action": action,
            "protocol": scene.capabilities.protocol,
            "control_mode": scene.capabilities.control_mode,
            "supported_actions": list(
                scene.capabilities.supported_actions
            ),
        }
        self._debug_event("runtime_action_capability_rejected", payload)
        raise RuntimeError(
            f"{scene.capabilities.protocol}/"
            f"{scene.capabilities.control_mode} does not support {action}"
        )

    def _record_observation(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        modalities: list[str],
        source: str,
        overwrite_task_initial: bool = False,
    ) -> dict:
        observation = self.tools.observe(scene.scene_instance_id, agent_id, modalities)
        raw = RawObservation.from_env_observation(observation, source=source)
        data = raw.to_dict()
        started = time.monotonic()
        try:
            write_result = self.memory_writer.write_observation(data)
        finally:
            self._debug_event("artifact_write_timing", {"kind": "observation", "elapsed_sec": time.monotonic() - started})
        observation_id = write_result["observation_ref"]["id"]
        data["observation_id"] = observation_id
        frames = data.get("frames")
        if isinstance(frames, dict):
            for frame_ref in write_result.get("frame_refs", []):
                modality = frame_ref.get("modality")
                frame = frames.get(modality)
                if isinstance(frame, dict):
                    frame["id"] = frame_ref.get("id")
                    frame["frame_id"] = frame_ref.get("id")
                    frame["artifact_ref"] = frame_ref.get("ref")
                    frame["observation_id"] = observation_id
        state.record_observation(data)
        memory = self._memory_for_state(state)
        memory.append_observation_ref(write_result["observation_ref"])
        memory.append_pose_ref(write_result.get("pose_ref"))
        memory.append_frame_refs(write_result.get("frame_refs"))
        if source in {"preflight", "instruction_start"}:
            memory.set_observation_anchor(
                "task_initial",
                write_result["observation_ref"],
                overwrite=True,
            )
        self.recorder.write_json("debug/latest_observation.json", data)
        return data

    def _mark_before_last_action_window(
        self,
        state: MissionState,
        observation_before: dict[str, Any],
    ) -> None:
        """Anchor one observation for the whole ordered physical action batch."""

        memory = self._memory_for_state(state)
        if not memory.recent_observation_refs:
            return
        expected_id = observation_before.get("observation_id")
        latest_ref = memory.recent_observation_refs[-1]
        if expected_id and latest_ref.get("id") != expected_id:
            matching = next(
                (
                    ref
                    for ref in reversed(memory.recent_observation_refs)
                    if ref.get("id") == expected_id
                ),
                None,
            )
            if matching is None:
                raise RuntimeError(
                    "cannot anchor before_last_action_window: observation is not in active memory"
                )
            latest_ref = matching
        memory.set_observation_anchor(
            "before_last_action_window", latest_ref, overwrite=True
        )

    def _try_hover(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        if scene.capabilities and not scene.capabilities.supports("hover"):
            self._debug_event(
                "safety_hover_skipped",
                {"reason": "backend_capability"},
            )
            return
        try:
            result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            self._record_action(state, "uav.hover", {}, result.to_dict())
            self._debug_event("safety_hover_completed", result.to_dict())
            if result.execution_state.get("requires_scene_reset"):
                raise SceneRecoveryRequired(result.message)
        except SceneRecoveryRequired:
            raise
        except Exception as exc:
            self._debug_event("safety_hover_failed", {"error_type": type(exc).__name__, "error": str(exc)})

    def _run_shutdown(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        if self.scene_sessions.is_unusable(scene):
            self._debug_event("shutdown_skipped", {"reason": "scene_recovery_required"})
            return
        if self.shutdown_policy == "none" or not state.connected:
            self._debug_event("shutdown_skipped", {"policy": self.shutdown_policy, "connected": state.connected})
            return

        self._debug_event("shutdown_started", {"policy": self.shutdown_policy})
        capabilities = scene.capabilities
        if capabilities is None or capabilities.supports("hover"):
            hover_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            hover_data = hover_result.to_dict()
            self._record_action(state, "uav.hover", {}, hover_data)
            self._debug_event("shutdown_hover_completed", hover_data)
            if not hover_result.ok:
                raise RuntimeError(f"Shutdown hover failed: {hover_result.message}")
        else:
            self._debug_event(
                "shutdown_hover_skipped",
                {
                    "reason": "backend_capability",
                    "protocol": capabilities.protocol,
                    "control_mode": capabilities.control_mode,
                },
            )

        if self.shutdown_policy == "land":
            if capabilities is None or capabilities.supports("land"):
                self._debug_event("shutdown_land_started", {"scene_instance_id": scene.scene_instance_id})
                land_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.land", {})
                land_data = land_result.to_dict()
                self._record_action(state, "uav.land", {}, land_data)
                self._debug_event("shutdown_land_completed", land_data)
                if not land_result.ok:
                    raise RuntimeError(f"Shutdown land failed: {land_result.message}")
            else:
                self._debug_event(
                    "shutdown_land_skipped",
                    {
                        "reason": "backend_capability",
                        "protocol": capabilities.protocol,
                        "control_mode": capabilities.control_mode,
                    },
                )

        self._write_state(state)

    def _write_state(self, state: MissionState) -> None:
        self.journal.write_state(
            state, self._memory_for_state(state), self._state_summary(state)
        )

    def _event(self, event: str, payload: dict) -> None:
        self.journal.event(event, payload)

    def _debug_event(self, event: str, payload: dict) -> None:
        self.journal.debug_event(event, payload)

    def _emit_runtime_event(self, event: dict[str, Any]) -> None:
        """Push a live event to CLI/TUI consumers without changing run history.

        The sink is deliberately best-effort: user interface rendering should not
        be able to fail a flight task. Durable replay and debugging still use the
        recorder-backed public/debug events.
        """

        self.journal.emit_live(event)

    def _persist_provider_exchange_media(
        self, model_rounds: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Write request media once per digest and return base64-free rounds."""
        return self.journal.persist_provider_exchange_media(model_rounds)

    @staticmethod
    def _stabilization_summary(stabilization: dict | None) -> dict:
        data = stabilization if isinstance(stabilization, dict) else {}
        return {
            "status": data.get("status"),
            "elapsed_sec": data.get("elapsed_sec"),
            "last_velocity_norm_mps": data.get("last_velocity_norm_mps"),
            "threshold": data.get("threshold"),
            "reason": data.get("reason"),
        }

    @staticmethod
    def _observation_summary_event(observation: dict, source: str | None = None) -> dict:
        frames = observation.get("frames") if isinstance(observation, dict) else {}
        frame_refs: dict[str, str] = {}
        if isinstance(frames, dict):
            for modality, frame in frames.items():
                if isinstance(frame, dict):
                    artifact_path = frame.get("artifact_path") or frame.get("path")
                    if artifact_path:
                        frame_refs[str(modality)] = str(artifact_path)
        return {
            "source": source or observation.get("source"),
            "timestamp": observation.get("timestamp") if isinstance(observation, dict) else None,
            "scene_instance_id": observation.get("scene_instance_id") if isinstance(observation, dict) else None,
            "agent_id": observation.get("agent_id") if isinstance(observation, dict) else None,
            "modalities": list(frames.keys()) if isinstance(frames, dict) else [],
            "has_pose": bool(observation.get("pose")) if isinstance(observation, dict) else False,
            "frame_artifacts": frame_refs,
        }
    @staticmethod
    def _state_summary(state: MissionState) -> dict:
        return {
            "run_id": state.run_id, "completed": state.completed,
            "stopped": state.stopped, "stopped_reason": state.stopped_reason,
            "stop_source": state.stop_source, "connected": state.connected,
            "airborne": state.airborne, "physical_window_count": state.physical_window_count,
            "plan_revision": state.task_plan.revision if state.task_plan else None,
        }

    def _forget_scene_memory(self, scene: SceneInstance) -> None:
        for key, memory in list(self._active_flight_memories.items()):
            if memory.scene_instance_id == scene.scene_instance_id:
                self._active_flight_memories.pop(key, None)

    def _resolve_flight_memory(
        self,
        map_id: str,
        scene: SceneInstance,
        agent_id: str,
        scene_policy: str,
        instruction: str,
        state: MissionState,
    ) -> FlightMemory:
        key = (map_id, scene.scene_id, agent_id)
        if scene_policy == "reuse_running" and key in self._active_flight_memories:
            memory = self._active_flight_memories[key]
        else:
            memory = FlightMemory(
                run_id=state.run_id,
                map_id=map_id,
                scene_instance_id=scene.scene_instance_id,
                agent_id=agent_id,
                memory_reader=self.memory_reader,
            )
            if scene_policy == "reuse_running":
                self._active_flight_memories[key] = memory
        instruction_ref = self.memory_writer.write_instruction(instruction, {})
        memory.update_instruction_ref(instruction_ref, instruction)
        memory.observation_anchors.clear()
        return memory

    def _memory_for_state(self, state: MissionState) -> FlightMemory:
        memory = self._state_flight_memories.get(state.run_id)
        if memory is None:
            raise RuntimeError(f"FlightMemory missing for run: {state.run_id}")
        return memory

    def _record_action(self, state: MissionState, tool: str, args: dict, result: dict) -> dict[str, Any]:
        state.record_action(tool, args, result)
        started = time.monotonic()
        try:
            write_result = self.memory_writer.write_action(tool, args, result)
        finally:
            self._debug_event("artifact_write_timing", {"kind": "action", "elapsed_sec": time.monotonic() - started})
        memory = self._memory_for_state(state)
        memory.append_action_ref(write_result["action_ref"])
        memory.append_pose_ref(write_result.get("pose_ref"))
        return write_result


def _attach_cleanup_error(primary: Exception, cleanup: Exception, stage: str) -> None:
    """Retain secondary cleanup evidence without replacing the original failure."""
    errors = list(getattr(primary, "cleanup_errors", []))
    errors.append({"stage": stage, "error_type": type(cleanup).__name__, "error": str(cleanup)})
    primary.cleanup_errors = errors
