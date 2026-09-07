"""Run semantic tasks through short-horizon planning windows."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.messages import AgentMessage
from src.agent.phase import (
    ADVANCE_TO_NEXT_PHASE,
    COMPLETE_TASK,
    CONTINUE_CURRENT_PHASE,
    STOP_TASK_INCOMPLETE,
    PhaseTransition,
)
from src.agent.planning.model_action_planner import AgentProtocolError, ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.agent.safety import SafetyDecision
from src.agent.planning.schemas import (
    ActionWindow,
    CompletionAssessment,
    CoupledActionPhasePlan,
    PhaseStrategy,
    PlannedStep,
    RuntimeControlAction,
    ShortHorizonPlan,
)
from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn, ToolResult
from src.agent.semantic.plan_updater import (
    SemanticPlanUpdater,
    validate_semantic_plan_patch_numeric_provenance,
)
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.task_parser import TaskParser
from src.agent.semantic.validator import apply_semantic_plan_patch
from src.env_client import EnvClient, SceneInstance
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.plugins.skills import ModelSkillSelector
from src.perception.raw_observation import RawObservation
from src.runtime.control import RuntimeControlExecutor, UavStabilizer
from src.runtime.control.actions import SemanticPlanUpdateJob
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.model_config import ModelConfig
from src.runtime.configs.runtime_control_config import RuntimeControlConfig, load_runtime_control_config
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.configs.task_execution_options import InitialPose, TaskExecutionOptions
from src.runtime.context import PhaseWindowContextBuilder
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import default_registry
from src.tools.validation import (
    ToolArgumentError,
    validate_runtime_bound_tool_args,
    validate_tool_args,
)
from src.runtime.middleware import ToolTelemetryMiddleware
from src.uav import clean_pose_for_model, pose_measurements


class TaskRuntime:
    SHUTDOWN_POLICIES = {"hover", "land", "none"}
    SCENE_POLICIES = {"isolated", "reuse_running"}

    def __init__(
        self,
        env_client: EnvClient,
        recorder,
        task_parser: TaskParser,
        planner=None,
        safety: SafetySupervisor | None = None,
        shutdown_policy: str = "hover",
        action_config: UavActionConfig | None = None,
        stabilization_config: UavStabilizationConfig | None = None,
        runtime_control_config: RuntimeControlConfig | None = None,
        semantic_plan_updater: Any | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        runtime_loop_guard_windows: int = 40,
        enable_action_planner_debug: bool = False,
        context_builder: ObservationContextBuilder | None = None,
        model_config: ModelConfig | None = None,
        plugin_registry: PluginRegistry | None = None,
    ) -> None:
        if shutdown_policy not in self.SHUTDOWN_POLICIES:
            raise ValueError(f"Unsupported shutdown policy: {shutdown_policy}")
        if int(runtime_loop_guard_windows) <= 0:
            raise ValueError("runtime_loop_guard_windows must be positive")
        self.env_client = env_client
        self.recorder = recorder
        self.task_parser = task_parser
        if planner is None and model_config is None:
            raise ValueError(
                "TaskRuntime requires planner or explicit model_config; "
                "runtime components do not load .env implicitly"
            )
        self.planner = planner or ModelActionPlanner(model_config)
        self.model_config = model_config
        self.safety = safety or SafetySupervisor()
        self.shutdown_policy = shutdown_policy
        self.runtime_loop_guard_windows = int(runtime_loop_guard_windows)
        self.enable_action_planner_debug = bool(enable_action_planner_debug)
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()
        self.tools = ToolDispatcher(
            env_client,
            registry=default_registry(self.plugin_registry, self.safety.safety_config),
            plugin_registry=self.plugin_registry,
        )
        self.stabilizer = UavStabilizer(self.tools, recorder, stabilization_config)
        resolved_runtime_control_config = (
            runtime_control_config or load_runtime_control_config()
        )
        actual_model_planner = isinstance(self.planner, ModelActionPlanner)
        model_planner_enabled = (
            actual_model_planner
            and getattr(self.planner, "planner_mode", "") == "model"
        )
        self._model_planner_enabled = model_planner_enabled
        if model_planner_enabled:
            if self.model_config is None:
                raise ValueError(
                    "Model-backed TaskRuntime requires explicit model_config"
                )
            if not self.model_config.api_key:
                raise ValueError(
                    "Model-backed TaskRuntime requires model_config.api_key"
                )
        if (
            semantic_plan_updater is None
            and resolved_runtime_control_config.is_enabled("semantic_plan_update")
        ):
            if self.model_config is not None and self.model_config.api_key:
                semantic_plan_updater = SemanticPlanUpdater(self.model_config)
            elif model_planner_enabled:
                raise ValueError(
                    "Enabled semantic_plan_update requires explicit model_config with api_key"
                )
        self.runtime_control = RuntimeControlExecutor(
            resolved_runtime_control_config,
            semantic_plan_updater=semantic_plan_updater,
        )
        # Optional synchronous stream for CLI/TUI frontends. The recorder remains
        # the historical source of truth; this sink is only for live user display.
        self.event_sink = event_sink
        # Evaluation and experimental integrations may replace one context
        # provider without changing global plugin configuration.  Runtime owns
        # when context is built; the injected builder still owns its contents.
        self.context_builder = context_builder or ObservationContextBuilder(
            action_config=action_config,
            safety_config=self.safety.safety_config,
            plugin_registry=self.plugin_registry,
            skill_selector=(
                ModelSkillSelector(self.model_config)
                if model_planner_enabled
                else ModelSkillSelector(enabled=False)
            ),
        )
        self.semantic_evidence_capabilities: list[str] = []
        # Initialize compatibility for callers that provide an already-parsed
        # plan, then refresh again before every model parse. Context builders
        # may be reconfigured between tasks, so constructor-time discovery is
        # not a durable source of truth for later instructions.
        self._refresh_semantic_evidence_capabilities()
        self.phase_window_context_builder = PhaseWindowContextBuilder()
        self.tool_telemetry = ToolTelemetryMiddleware(self.tools, self._debug_event)
        self.memory_writer = MemoryEventWriter(self.recorder.run_dir)
        self.memory_reader = MemoryRecordReader(self.recorder.run_dir)
        self.agent_transcript = AgentTranscript(self.memory_writer)
        self._preflight_scene_ids: set[str] = set()
        self._active_reused_scenes: dict[tuple[str, str | None], SceneInstance] = {}
        self._active_flight_memories: dict[tuple[str, str | None, str], FlightMemory] = {}
        self._state_flight_memories: dict[str, FlightMemory] = {}
        # In-memory user-facing messages for the current instruction. Public
        # events are still written for history, but CLI/TUI summaries should not
        # depend on the async logger flushing before they can show the final reply.
        self._current_turn_agent_messages: list[dict] = []
        self._pending_task_parser_trace: dict[str, Any] = {}
        # A decision is one state-checkpoint reasoning cycle. It is deliberately
        # independent from physical window numbering because terminal decisions
        # and read-only rounds consume no flight window.
        self._decision_index = 0
        self._pending_failed_model_rounds: list[dict[str, Any]] = []
        self._persisted_provider_media: set[str] = set()

    def run_task(
        self,
        map_id: str,
        instruction: str,
        agent_id: str = "uav_1",
        scene_policy: str = "isolated",
        stop_scene_on_finish: bool = True,
        scene_id: str | None = None,
        execution_options: TaskExecutionOptions | None = None,
    ) -> MissionState:
        # Execute one short-horizon window per semantic phase.
        options = self._resolve_execution_options(execution_options)
        self._current_turn_agent_messages = []
        self._decision_index = 0
        self._pending_failed_model_rounds = []
        self._persisted_provider_media = set()
        self._refresh_semantic_evidence_capabilities()
        parse_result = self.task_parser.parse(instruction)
        parse_trace = getattr(self.task_parser, "last_parse_trace", {})
        self._pending_task_parser_trace = (
            dict(parse_trace) if isinstance(parse_trace, dict) else {}
        )
        try:
            return self.run_parsed_task(
                map_id,
                parse_result,
                agent_id=agent_id,
                scene_policy=scene_policy,
                stop_scene_on_finish=stop_scene_on_finish,
                scene_id=scene_id,
                execution_options=options,
            )
        finally:
            self._pending_task_parser_trace = {}

    def run_parsed_task(
        self,
        map_id: str,
        parse_result: TaskParseResult,
        agent_id: str = "uav_1",
        scene_policy: str = "isolated",
        stop_scene_on_finish: bool = True,
        scene_id: str | None = None,
        execution_options: TaskExecutionOptions | None = None,
    ) -> MissionState:
        # Execute a task from an already-created semantic plan.
        if scene_policy not in self.SCENE_POLICIES:
            raise ValueError(f"Unsupported scene policy: {scene_policy}")
        options = self._resolve_execution_options(execution_options)
        self._current_turn_agent_messages = []
        self._decision_index = 0
        self._pending_failed_model_rounds = []
        self._persisted_provider_media = set()
        self._debug_event("task_execution_options_resolved", options.to_dict())
        self.recorder.write_json("state/semantic_plan.json", parse_result.to_dict())
        self._debug_event("task_parsed", parse_result.to_dict())
        self._debug_event(
            "model_dependency_status",
            self._model_dependency_status(),
        )
        if self._pending_task_parser_trace:
            self._debug_event(
                "task_parser_phase_skill_trace",
                self._pending_task_parser_trace,
            )
            self._pending_task_parser_trace = {}
        self._event(
            "instruction_started",
            {
                "instruction": parse_result.instruction,
                "phase_count": len(parse_result.plan.phases),
                "phase_names": [phase.name for phase in parse_result.plan.phases],
            },
        )
        scene = self._resolve_scene(map_id, scene_policy, scene_id)
        state = MissionState(
            run_id=self.recorder.run_dir.name,
            user_instruction=parse_result.instruction,
            semantic_plan=parse_result.plan,
            connected=scene.scene_instance_id in self._preflight_scene_ids,
            airborne=(
                scene.scene_instance_id in self._preflight_scene_ids
                and bool(
                    scene.capabilities
                    and scene.capabilities.requires_takeoff
                )
            ),
            observation_modalities=options.observation_modalities,
            max_phase_window_count=self.runtime_loop_guard_windows,
        )
        flight_memory = self._resolve_flight_memory(map_id, scene, agent_id, scene_policy, parse_result, state)
        self._state_flight_memories[state.run_id] = flight_memory
        self.agent_transcript.start_instruction(
            parse_result.instruction,
            flight_memory.active_instruction_id,
        )
        try:
            self._validate_execution_options(scene, options)
            self._prepare_scene_for_task(scene, agent_id, state, options)
            self._execute_phase_loop(scene, agent_id, state)
            if stop_scene_on_finish:
                self._run_shutdown(scene, agent_id, state)
            self._event(
                "instruction_completed",
                {
                    "instruction": state.user_instruction,
                    "completed": state.completed,
                    "stopped": state.stopped,
                    "stopped_reason": state.stopped_reason,
                    "stop_source": state.stop_source,
                    "current_phase": state.current_phase_name,
                    "current_phase_index": state.current_phase_index,
                },
            )
            return state
        except Exception as exc:
            state.record_error(type(exc).__name__, str(exc))
            self._write_state(state)
            self._event("instruction_failed", {"error_type": type(exc).__name__, "error": str(exc)})
            self._debug_event("task_failed", {"error_type": type(exc).__name__, "error": str(exc)})
            if state.connected:
                self._try_hover(scene, agent_id, state)
            raise
        finally:
            if stop_scene_on_finish:
                self.env_client.stop_scene(scene.scene_instance_id)
                self._preflight_scene_ids.discard(scene.scene_instance_id)
                self._forget_scene(scene)
                self._debug_event("scene_stopped", scene.to_dict())

    def finish_reused_scene(
        self,
        map_id: str,
        state: MissionState | None,
        agent_id: str = "uav_1",
        scene_id: str | None = None,
    ) -> None:
        # Apply final shutdown policy and stop a reused scene owned by this runtime.
        scene = self._active_reused_scenes.get(self._scene_key(map_id, scene_id))
        if scene is None:
            self._debug_event("scene_stop_skipped", {"map_id": map_id, "scene_id": scene_id, "reason": "not_running"})
            return
        try:
            if state is not None:
                self._run_shutdown(scene, agent_id, state)
        finally:
            self.env_client.stop_scene(scene.scene_instance_id)
            self._preflight_scene_ids.discard(scene.scene_instance_id)
            self._forget_scene(scene)
            self._debug_event("scene_stopped", scene.to_dict())

    def safe_hover_reused_scene(self, map_id: str, agent_id: str = "uav_1", scene_id: str | None = None) -> None:
        # Best-effort hover after an instruction failure in a reused scene.
        scene = self._active_reused_scenes.get(self._scene_key(map_id, scene_id))
        if scene is None:
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

        return self._resolve_scene(map_id, "reuse_running", scene_id)

    def _resolve_scene(self, map_id: str, scene_policy: str, scene_id: str | None) -> SceneInstance:
        if scene_policy == "reuse_running":
            key = self._scene_key(map_id, scene_id)
            known_scene = self._active_reused_scenes.get(key)
            scene = self.env_client.get_or_start_scene(map_id, scene_id)
            self._active_reused_scenes[key] = scene
            self._record_scene_contract(scene)
            if known_scene is not None and known_scene.scene_instance_id == scene.scene_instance_id:
                self._debug_event("scene_reused", scene.to_dict())
            else:
                self._debug_event("scene_started", scene.to_dict())
            return scene

        scene = self.env_client.start_scene(map_id, scene_id)
        self._record_scene_contract(scene)
        self._debug_event("scene_started", scene.to_dict())
        return scene

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
        first_scene_use = scene.scene_instance_id not in self._preflight_scene_ids
        if first_scene_use:
            self._run_preflight(scene, agent_id, state)
        if options.initial_pose is not None:
            self._apply_initial_pose(scene, agent_id, state, options.initial_pose)
        if first_scene_use:
            # Hold only after an episode pose has replaced the packaged scene's
            # default pose. This prevents a visible/default-pose hover followed
            # by a large camera jump at the start of an evaluation episode.
            self._run_preflight_hover(scene, agent_id, state)
            self._preflight_scene_ids.add(scene.scene_instance_id)

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

    def _execute_phase_loop(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        while not state.completed and not state.stopped:
            self._apply_completed_semantic_plan_update(state)
            # Native read-only tool rounds are reasoning within one observed
            # state. Only a committed physical batch consumes a flight window.
            if not callable(getattr(self.planner, "plan_turn", None)):
                state.record_phase_window_started()
            plan, safety_decision, transition = self._run_phase_window(scene, agent_id, state)
            self._apply_phase_transition(scene, agent_id, state, plan, safety_decision, transition)
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

    def _run_phase_window(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
    ) -> tuple[ShortHorizonPlan, SafetyDecision, PhaseTransition]:
        return self._run_coupled_phase_window(scene, agent_id, state)

    def _run_coupled_phase_window(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
    ) -> tuple[ShortHorizonPlan, SafetyDecision, PhaseTransition]:
        """Execute one state decision containing one or more model rounds."""

        self._decision_index += 1
        decision_index = self._decision_index
        decision_id = f"decision_{decision_index:04d}"
        phase_name = state.current_phase_name
        phase_key = state.current_phase_key
        semantic_plan_version = state.semantic_plan_version
        observation_before = dict(state.latest_observation)
        checkpoint_observation_id = observation_before.get("observation_id")
        completed_window_count = state.current_phase_window_count
        state.ensure_current_phase_progress(observation_before)
        state.update_current_phase_progress(observation_before)
        context = self.context_builder.build(state, self._memory_for_state(state))
        skill_trace = self._planner_skill_selection_trace(context)
        if skill_trace:
            self._debug_event(
                "planner_phase_skill_trace",
                {
                    "phase": phase_name,
                    "phase_key": phase_key,
                    "window_index": state.current_phase_window_count,
                    **skill_trace,
                },
            )
        runtime_control_context = self.runtime_control.model_context(
            self._runtime_control_recommendation_context(state)
        )
        phase_window_context = self.phase_window_context_builder.build(
            state,
            context,
            observation_before,
            runtime_control_context=runtime_control_context,
            scene_capabilities=scene.capabilities,
        )
        coupled_context = phase_window_context.planner_payload
        model_context = phase_window_context.model_context
        self.recorder.write_json("contexts/latest_context.json", coupled_context)
        self.recorder.write_json("contexts/latest_action_planner_model_context.json", model_context)
        native_meta: dict[str, Any] = {}
        if callable(getattr(self.planner, "plan_turn", None)):
            self.agent_transcript.append_state(
                coupled_context,
                observation_before.get("observation_id"),
            )
            try:
                coupled_plan, native_meta = self._run_native_agent_tool_loop(
                    scene,
                    agent_id,
                    state,
                )
            except Exception:
                self._write_failed_action_planner_debug_artifact(
                    state,
                    coupled_context,
                    model_context,
                    {
                        "decision_id": decision_id,
                        "decision_index": decision_index,
                        "checkpoint_observation_id": checkpoint_observation_id,
                        "completed_window_count": completed_window_count,
                        "physical_window_index": None,
                    },
                    self._pending_failed_model_rounds,
                )
                raise
            completion_result = {
                "status": "valid",
                "native_tool_calls": True,
                "assessment": (
                    coupled_plan.completion_assessment.to_dict()
                    if coupled_plan.completion_assessment is not None
                    else None
                ),
            }
            if coupled_plan.phase_decision.decision == CONTINUE_CURRENT_PHASE:
                state.record_phase_window_started()
        else:
            # Deterministic injected planners remain useful for simulator and
            # safety tests. The production model path above has no legacy JSON
            # provider fallback.
            planner = getattr(self.planner, "plan_next_with_phase_decision", None)
            if not callable(planner):
                raise RuntimeError("TaskRuntime planner must implement plan_turn")
            coupled_plan = planner(state, coupled_context)
            completion_result = dict(
                getattr(self.planner, "last_completion_result", {}) or {}
            )
        physical_window_index = (
            state.current_phase_window_count
            if coupled_plan.phase_decision.decision == CONTINUE_CURRENT_PHASE
            else None
        )
        decision_meta = {
            "decision_id": decision_id,
            "decision_index": decision_index,
            "checkpoint_observation_id": checkpoint_observation_id,
            "completed_window_count": completed_window_count,
            "physical_window_index": physical_window_index,
        }
        completion_payload = {
            "phase": coupled_plan.phase,
            "phase_key": phase_key,
            "window_index": physical_window_index,
            **decision_meta,
            **completion_result,
        }
        self._event("phase_completion_assessed", completion_payload)
        if completion_result.get("fallback"):
            self._debug_event("phase_completion_contract_fallback", completion_payload)
        transition = coupled_plan.phase_decision
        strategy_update: dict[str, Any] = {}
        if (
            transition.decision == CONTINUE_CURRENT_PHASE
            and coupled_plan.phase_strategy is not None
        ):
            strategy_update = state.update_current_phase_strategy(
                coupled_plan.phase_strategy.to_dict()
            )
            self._event(
                "phase_strategy_updated",
                {
                    "phase": coupled_plan.phase,
                    "phase_key": phase_key,
                    "window_index": physical_window_index,
                    **decision_meta,
                    **strategy_update,
                },
            )
        raw_plan = coupled_plan.to_short_horizon_plan()
        self._debug_event("model_action_planned", coupled_plan.to_dict())
        control_results = self.runtime_control.validate_all(coupled_plan.control_actions)
        self._record_runtime_control_results(coupled_plan.control_actions, control_results)
        self._append_native_control_results(native_meta, control_results)
        safety_decision = SafetyDecision(raw_plan, [])
        planner_debug_ref = self._write_action_planner_debug_artifact(
            state,
            coupled_context,
            model_context,
            raw_plan,
            raw_plan,
            safety_decision,
            coupled_plan=coupled_plan,
            decision_meta=decision_meta,
            native_meta=native_meta,
        )
        decision_payload = {
            "phase": coupled_plan.phase,
            **decision_meta,
            "decision": transition.decision,
            "reason": transition.reason,
            "step_count": len(coupled_plan.steps),
            "planner_debug_ref": planner_debug_ref,
            "phase_strategy_revision": state.current_phase_strategy_revision,
        }
        self._event("coupled_planner_phase_decided", decision_payload)
        self._agent_messages(coupled_plan.agent_messages, source="coupled_planner", phase=coupled_plan.phase)
        if transition.decision == STOP_TASK_INCOMPLETE:
            self._finish_agent_reported_blocked(
                scene,
                agent_id,
                state,
                native_meta,
                decision_payload,
            )
            planner_debug_ref = self._write_action_planner_debug_artifact(
                state,
                coupled_context,
                model_context,
                raw_plan,
                raw_plan,
                safety_decision,
                coupled_plan=coupled_plan,
                decision_meta=decision_meta,
                native_meta=native_meta,
            )
            self._event(
                "phase_plan_ready",
                {
                    "phase": raw_plan.phase,
                    **decision_meta,
                    "action_window": raw_plan.action_window.to_dict(),
                    "step_count": 0,
                    "tools": [],
                    "planner_debug_ref": planner_debug_ref,
                    "coupled_phase_decision": transition.to_dict(),
                    "note": "Agent reported a terminal blocker; no model-requested UAV tool executed.",
                },
            )
            return raw_plan, safety_decision, transition
        if transition.decision != CONTINUE_CURRENT_PHASE:
            self._append_native_commit_result(
                native_meta,
                status="ok",
                message="phase assessment committed without a physical action window",
            )
            planner_debug_ref = self._write_action_planner_debug_artifact(
                state,
                coupled_context,
                model_context,
                raw_plan,
                raw_plan,
                safety_decision,
                coupled_plan=coupled_plan,
                decision_meta=decision_meta,
                native_meta=native_meta,
            )
            self._event(
                "phase_plan_ready",
                {
                    "phase": raw_plan.phase,
                    **decision_meta,
                    "action_window": raw_plan.action_window.to_dict(),
                    "step_count": 0,
                    "tools": [],
                    "planner_debug_ref": planner_debug_ref,
                    "coupled_phase_decision": transition.to_dict(),
                    "note": "No UAV tool executed because coupled planner requested a phase transition.",
                },
            )
            return raw_plan, safety_decision, transition

        safety_decision = self.safety.review(raw_plan, state, registry=self.tools.registry)
        plan = safety_decision.plan
        planner_debug_ref = self._write_action_planner_debug_artifact(
            state,
            coupled_context,
            model_context,
            raw_plan,
            plan,
            safety_decision,
            coupled_plan=coupled_plan,
            decision_meta=decision_meta,
            native_meta=native_meta,
        )
        self._debug_event("short_horizon_planned", plan.to_dict())
        self._event(
            "phase_plan_ready",
            {
                "phase": plan.phase,
                **decision_meta,
                "action_window": plan.action_window.to_dict(),
                "step_count": len(plan.steps),
                "tools": [step.tool for step in plan.steps if step.kind == "tool"],
                "planner_debug_ref": planner_debug_ref,
                "coupled_phase_decision": transition.to_dict(),
            },
        )
        if safety_decision.interventions:
            self._debug_event(
                "safety_intervened",
                {"phase": plan.phase, "interventions": safety_decision.interventions},
            )
            self._agent_message(
                AgentMessage(
                    type="warning",
                    content="为保证安全，我调整了本轮动作。",
                    evidence_refs=[{"kind": "event", "event": "safety_intervened"}],
                    confidence="high",
                    requires_user_reply=False,
                ),
                source="safety",
                phase=plan.phase,
            )
        self._mark_before_last_action_window(state, observation_before)
        executed_steps = []
        native_physical_calls = list(native_meta.get("physical_calls", []))
        failed_native_index: int | None = None
        for index, step in enumerate(plan.steps):
            payload = {"phase": plan.phase, "step_index": index, "step": step.to_dict()}
            if step.kind != "tool" or not step.tool:
                raise ValueError("runtime action plan requires tool steps; observation is automatic")
            self._validate_step_capability(scene, step.tool)
            try:
                result = self.tool_telemetry.call_step(
                    scene_instance_id=scene.scene_instance_id,
                    agent_id=agent_id,
                    phase=plan.phase,
                    window_index=state.current_phase_window_count,
                    step_index=index,
                    tool=step.tool,
                    args=step.args,
                    pose_before=self._pose_from_observation(state.latest_observation),
                    runtime_context={
                        "latest_observation": dict(state.latest_observation),
                        "observation_before": dict(observation_before),
                        "phase": plan.phase,
                        "run_dir": str(self.recorder.run_dir),
                    },
                )
            except Exception as exc:
                if not native_physical_calls:
                    raise
                from src.env_client import ActionResult

                result = ActionResult(
                    ok=False,
                    action=step.tool,
                    message=f"{type(exc).__name__}: {exc}",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )
            data = result.to_dict()
            write_result = self._record_action(state, step.tool, step.args, data)
            native_call = native_physical_calls[index] if index < len(native_physical_calls) else None
            if native_call is not None:
                requested_tool = native_call.tool_name
                relevant_interventions = [
                    dict(item)
                    for item in safety_decision.interventions
                    if item.get("tool") == requested_tool
                ]
                transcript_result = ToolResult(
                        tool_call_id=native_call.tool_call_id,
                        tool_name=requested_tool,
                        status="ok" if result.ok else "error",
                        message=result.message,
                        data=dict(result.data),
                        result_ref=write_result["action_ref"].get("ref"),
                        requested_args=dict(native_call.args),
                        executed_args=dict(step.args),
                        safety_interventions=relevant_interventions,
                    )
                self.agent_transcript.append_tool_result(transcript_result)
                self._append_native_round_tool_result(native_meta, transcript_result)
            if not result.ok:
                executed_steps.append(
                    {
                        **payload,
                        "result": data,
                        "stabilization": {
                            "status": "skipped",
                            "reason": "tool execution failed",
                        },
                    }
                )
                if not native_physical_calls:
                    raise RuntimeError(f"Action failed: {step.tool}: {result.message}")
                failed_native_index = index
                break
            stabilization = self.stabilizer.wait_after_step(
                scene.scene_instance_id,
                agent_id,
                plan.phase,
                step,
                capabilities=scene.capabilities,
            )
            executed_steps.append({**payload, "result": data, "stabilization": stabilization})
            self._event(
                "tool_step_completed",
                {
                    "decision_id": decision_id,
                    "physical_window_index": physical_window_index,
                    "phase": plan.phase,
                    "step_index": index,
                    "tool": step.tool,
                    "ok": data.get("ok"),
                    "stabilization": self._stabilization_summary(stabilization),
                },
            )
            self._write_state(state)
        if failed_native_index is not None:
            for skipped_index, call in enumerate(
                native_physical_calls[failed_native_index + 1 :],
                start=failed_native_index + 1,
            ):
                transcript_result = ToolResult(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        status="skipped",
                        message="skipped because an earlier physical call failed",
                        requested_args=dict(call.args),
                    )
                self.agent_transcript.append_tool_result(transcript_result)
                self._append_native_round_tool_result(native_meta, transcript_result)
                skipped_step = (
                    plan.steps[skipped_index]
                    if skipped_index < len(plan.steps)
                    else None
                )
                if skipped_step is not None:
                    executed_steps.append(
                        {
                            "phase": plan.phase,
                            "step_index": skipped_index,
                            "step": skipped_step.to_dict(),
                            "result": {
                                "ok": False,
                                "status": "skipped",
                                "message": "skipped because an earlier physical call failed",
                            },
                            "stabilization": {
                                "status": "skipped",
                                "reason": "physical call was not executed",
                            },
                        }
                    )
            self._try_hover(scene, agent_id, state)
        observation_after = self._apply_automatic_observation_policy(scene, agent_id, state, plan.phase)
        window_measurements = pose_measurements(
            self._pose_from_observation(observation_before),
            self._pose_from_observation(observation_after),
        )
        self._append_native_commit_result(
            native_meta,
            status="error" if failed_native_index is not None else "ok",
            message=(
                "physical action window completed with a failed call"
                if failed_native_index is not None
                else "physical action window completed and was observed"
            ),
            data={
                "observation_before_id": observation_before.get("observation_id"),
                "observation_after_id": observation_after.get("observation_id"),
                "observed_effect": window_measurements,
            },
        )
        planner_debug_ref = self._write_action_planner_debug_artifact(
            state,
            coupled_context,
            model_context,
            raw_plan,
            plan,
            safety_decision,
            coupled_plan=coupled_plan,
            decision_meta=decision_meta,
            native_meta=native_meta,
        )
        visual_search_recharge = self._maybe_recharge_visual_search_budget(
            state,
            window_measurements,
            phase=phase_name,
        )
        window_record = {
            **decision_meta,
            "phase_key": phase_key,
            "phase": phase_name,
            "window_index": state.current_phase_window_count,
            "action_window": plan.action_window.to_dict(),
            "coupled_phase_decision": transition.to_dict(),
            "executed_steps": list(executed_steps),
            "observation_before": observation_before,
            "observation_after": observation_after,
            "measurements": window_measurements,
            "safety_interventions": list(safety_decision.interventions),
            "phase_strategy": dict(state.current_phase_strategy),
            "phase_strategy_revision": state.current_phase_strategy_revision,
        }
        if visual_search_recharge is not None:
            window_record["visual_search_recharge"] = visual_search_recharge
        window_ref = self._write_phase_window_debug_artifact(phase_key, state.current_phase_window_count, window_record)
        window_summary = {
            **decision_meta,
            "phase": phase_name,
            "phase_key": phase_key,
            "window_index": state.current_phase_window_count,
            "action_window": plan.action_window.to_dict(),
            "coupled_phase_decision": transition.to_dict(),
            "executed_step_count": len(executed_steps),
            "executed_tools": [
                executed.get("step", {}).get("tool")
                for executed in executed_steps
                if isinstance(executed.get("step", {}), dict)
            ],
            "measurements": window_measurements,
            "safety_intervention_count": len(safety_decision.interventions),
            "phase_strategy_revision": state.current_phase_strategy_revision,
            "phase_strategy_update_kind": strategy_update.get("update_kind"),
            "phase_strategy_active_stage": state.current_phase_strategy.get("active_stage"),
            "debug_ref": window_ref,
        }
        self._event("phase_window_recorded", window_summary)
        state.update_current_phase_progress(
            observation_after,
            last_window_measurements=window_measurements,
            tool_facts_delta=self._tool_facts_from_executed_steps(executed_steps),
        )
        self._submit_semantic_plan_updates_after_window(
            state,
            coupled_plan.control_actions,
            control_results,
            phase_key=phase_key,
            semantic_plan_version=semantic_plan_version,
            observation_before=observation_before,
            observation_after=observation_after,
            selected_skill_guidance=self._selected_semantic_update_guidance(coupled_context),
        )
        return plan, safety_decision, PhaseTransition.continue_current_phase(transition.reason)

    def _run_native_agent_tool_loop(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
    ) -> tuple[CoupledActionPhasePlan, dict[str, Any]]:
        """Run immediate tools until the model commits one phase boundary.

        Every rejected or completed read-only call is appended to the same
        transcript before the next model request, allowing the model to repair
        its own protocol and action choices without a separate repair model.
        """

        allowed_names = self._native_allowed_tool_names(scene)
        max_rounds = int(getattr(self.model_config, "max_tool_rounds", 6) or 6)
        model_rounds: list[dict[str, Any]] = []
        self._pending_failed_model_rounds = []
        for round_index in range(1, max_rounds + 1):
            try:
                turn = self.planner.plan_turn(
                    self.agent_transcript,
                    self.tools.registry,
                    allowed_tool_names=allowed_names,
                )
            except AgentProtocolError as exc:
                round_record = self._native_model_round_snapshot(round_index)
                feedback = (
                    f"Protocol error: {exc}. Return a corrected native tool-call turn."
                )
                round_record["runtime_outcome"] = {
                    "status": "protocol_rejected",
                    "feedback": feedback,
                }
                model_rounds.append(round_record)
                if exc.rejected_turn:
                    self.agent_transcript.append_provider_rejection(
                        exc.rejected_turn
                    )
                self.agent_transcript.append_protocol_feedback(feedback)
                self._debug_event(
                    "agent_tool_protocol_rejected",
                    {"round": round_index, "error": str(exc)},
                )
                continue
            except Exception as exc:
                round_record = self._native_model_round_snapshot(round_index)
                round_record["runtime_outcome"] = {
                    "status": "provider_failure",
                    "feedback": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                model_rounds.append(round_record)
                # Preserve failure evidence for the caller even though no
                # CoupledActionPhasePlan can be constructed.
                self._pending_failed_model_rounds = model_rounds
                raise

            round_record = self._native_model_round_snapshot(
                round_index,
                canonical_turn=turn,
            )
            self.agent_transcript.append_assistant(turn)
            specs = [self.tools.registry.get(call.tool_name) for call in turn.tool_calls]
            try:
                for call, spec in zip(turn.tool_calls, specs):
                    if call.tool_name not in allowed_names:
                        raise ToolArgumentError(f"tool is not exposed: {call.tool_name}")
                    validate_tool_args(call.args, spec.input_schema)
                    validate_runtime_bound_tool_args(
                        call.args,
                        spec.input_schema,
                        state.latest_observation,
                    )
            except ToolArgumentError as exc:
                feedback = (
                    "The previous tool-call turn was rejected: "
                    f"{exc}. Correct it using the available tools."
                )
                round_record["runtime_outcome"] = {
                    "status": "arguments_rejected",
                    "feedback": feedback,
                    "tool_results": self._rejected_tool_results(
                        turn.tool_calls, str(exc)
                    ),
                }
                model_rounds.append(round_record)
                self._reject_native_turn(turn.tool_calls, str(exc))
                continue

            read_calls = [
                call
                for call, spec in zip(turn.tool_calls, specs)
                if spec.runtime.effect == "read_only"
            ]
            if read_calls:
                if len(read_calls) != len(turn.tool_calls) or any(
                    self.tools.registry.get(call.tool_name).runtime.result_timing != "immediate"
                    for call in read_calls
                ):
                    reason = "read-only calls cannot be mixed with a phase commit or physical calls"
                    round_record["runtime_outcome"] = {
                        "status": "tool_set_rejected",
                        "feedback": reason,
                        "tool_results": self._rejected_tool_results(
                            turn.tool_calls, reason
                        ),
                    }
                    model_rounds.append(round_record)
                    self._reject_native_turn(
                        turn.tool_calls,
                        reason,
                    )
                    continue
                if len(read_calls) > 1 and any(
                    self.tools.registry.get(call.tool_name).runtime.batching
                    != "parallel_safe"
                    for call in read_calls
                ):
                    reason = "multiple read-only calls require parallel_safe batching"
                    round_record["runtime_outcome"] = {
                        "status": "tool_set_rejected",
                        "feedback": reason,
                        "tool_results": self._rejected_tool_results(
                            turn.tool_calls, reason
                        ),
                    }
                    model_rounds.append(round_record)
                    self._reject_native_turn(
                        turn.tool_calls,
                        reason,
                    )
                    continue
                read_results = self._execute_native_read_calls(
                    scene,
                    agent_id,
                    state,
                    turn.public_intent,
                    read_calls,
                )
                round_record["runtime_outcome"] = {
                    "status": "read_only_executed",
                    "feedback": None,
                    "tool_results": read_results,
                }
                model_rounds.append(round_record)
                continue

            commit_calls = [
                call for call in turn.tool_calls if call.tool_name == "runtime.commit_phase_turn"
            ]
            if len(commit_calls) != 1 or turn.tool_calls[0].tool_name != "runtime.commit_phase_turn":
                reason = "exactly one runtime.commit_phase_turn must be the first call"
                round_record["runtime_outcome"] = {
                    "status": "commit_rejected",
                    "feedback": reason,
                    "tool_results": self._rejected_tool_results(
                        turn.tool_calls, reason
                    ),
                }
                model_rounds.append(round_record)
                self._reject_native_turn(
                    turn.tool_calls,
                    reason,
                )
                continue

            commit = commit_calls[0]
            physical_calls = [
                call
                for call in turn.tool_calls
                if self.tools.registry.get(call.tool_name).runtime.effect == "physical"
            ]
            control_calls = [
                call
                for call in turn.tool_calls
                if self.tools.registry.get(call.tool_name).runtime.effect == "runtime_control"
                and call.tool_name != "runtime.commit_phase_turn"
            ]
            report_blocked_calls = [
                call
                for call in control_calls
                if call.tool_name == "runtime.report_blocked"
            ]
            recognized = 1 + len(physical_calls) + len(control_calls)
            if recognized != len(turn.tool_calls):
                reason = "turn contains a non-executable tool class"
                round_record["runtime_outcome"] = {
                    "status": "tool_set_rejected",
                    "feedback": reason,
                    "tool_results": self._rejected_tool_results(
                        turn.tool_calls, reason
                    ),
                }
                model_rounds.append(round_record)
                self._reject_native_turn(turn.tool_calls, reason)
                continue
            try:
                plan = self._native_committed_plan(
                    state,
                    turn.public_intent,
                    commit,
                    physical_calls,
                    control_calls,
                )
            except (TypeError, ValueError) as exc:
                round_record["runtime_outcome"] = {
                    "status": "commit_rejected",
                    "feedback": str(exc),
                    "tool_results": self._rejected_tool_results(
                        turn.tool_calls, str(exc)
                    ),
                }
                model_rounds.append(round_record)
                self._reject_native_turn(turn.tool_calls, str(exc))
                continue
            round_record["runtime_outcome"] = {
                "status": "commit_accepted",
                "feedback": None,
            }
            model_rounds.append(round_record)
            return plan, {
                "commit_call": commit,
                "physical_calls": physical_calls,
                "control_calls": control_calls,
                "report_blocked_call": (
                    report_blocked_calls[0] if len(report_blocked_calls) == 1 else None
                ),
                "model_rounds": model_rounds,
            }

        self._pending_failed_model_rounds = model_rounds
        raise RuntimeError(
            f"Agent native tool loop exhausted {max_rounds} rounds without a valid phase commit"
        )

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
        media = exchange.get("media")
        if isinstance(media, list):
            record["request_media"] = media
        if canonical_turn is not None:
            record["canonical_turn"] = canonical_turn.to_dict()
        elif not exchange:
            record["capture_status"] = "provider_exchange_unavailable"
        return record

    @staticmethod
    def _rejected_tool_results(
        calls: list[AgentToolCall], reason: str
    ) -> list[dict[str, Any]]:
        """Describe Runtime-generated rejection results for decision review."""

        return [
            ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="rejected",
                message=reason,
                requested_args=dict(call.args),
            ).to_dict()
            for call in calls
        ]

    def _native_allowed_tool_names(self, scene: SceneInstance) -> set[str]:
        allowed: set[str] = set()
        for spec in self.tools.registry.model_specs():
            backend_action = spec.metadata.get("backend_action")
            if (
                spec.runtime.effect == "physical"
                and scene.capabilities is not None
                and isinstance(backend_action, str)
                and not scene.capabilities.supports(backend_action)
            ):
                continue
            allowed.add(spec.name)
        exposed_runtime_actions = set(
            self.runtime_control.config.exposed_actions(
                self.runtime_control.registry.names()
            )
        )
        for canonical in self.tools.registry.names():
            spec = self.tools.registry.get(canonical)
            runtime_action = spec.metadata.get("runtime_action")
            if runtime_action in exposed_runtime_actions:
                allowed.add(canonical)
        return allowed

    def _execute_native_read_calls(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        public_intent: str,
        calls: list[AgentToolCall],
    ) -> list[dict[str, Any]]:
        recorded_results: list[dict[str, Any]] = []
        for call in calls:
            result = self.tools.call(
                scene.scene_instance_id,
                agent_id,
                call.tool_name,
                call.args,
                runtime_context={
                    "latest_observation": dict(state.latest_observation),
                    "phase": state.current_phase_name,
                    "run_dir": str(self.recorder.run_dir),
                    "public_intent": public_intent,
                },
            )
            data = result.to_dict()
            write_result = self._record_action(state, call.tool_name, call.args, data)
            transcript_result = ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="ok" if result.ok else "error",
                    message=result.message,
                    data=dict(result.data),
                    artifact_refs=_result_artifact_refs(result.data),
                    result_ref=write_result["action_ref"].get("ref"),
                    requested_args=dict(call.args),
                    executed_args=dict(call.args),
                )
            self.agent_transcript.append_tool_result(transcript_result)
            recorded_results.append(transcript_result.to_dict())
        return recorded_results

    def _native_committed_plan(
        self,
        state: MissionState,
        public_intent: str,
        commit: AgentToolCall,
        physical_calls: list[AgentToolCall],
        control_calls: list[AgentToolCall],
    ) -> CoupledActionPhasePlan:
        data = commit.args
        phase = str(data.get("phase", "")).strip()
        if phase != state.current_phase_name:
            raise ValueError(
                f"commit phase mismatch: expected {state.current_phase_name}, got {phase}"
            )
        assessment = CompletionAssessment.from_dict(
            dict(data.get("completion_assessment") or {})
        )
        current_phase = state.semantic_plan.phases[state.current_phase_index]
        assessment.validate_coverage(len(current_phase.done_when))
        complete = all(item.status == "supported" for item in assessment.done_when)
        message = AgentMessage(
            type="final" if complete and state.is_final_phase else "progress",
            content=public_intent,
            confidence="medium",
        )
        if complete:
            if physical_calls or control_calls:
                raise ValueError("a completed phase cannot emit physical or runtime-control side effects")
            decision = COMPLETE_TASK if state.is_final_phase else ADVANCE_TO_NEXT_PHASE
            transition = PhaseTransition(decision, "all done_when conditions are supported")
            return CoupledActionPhasePlan(
                phase=phase,
                phase_decision=transition,
                agent_messages=[message],
                completion_assessment=assessment,
            )

        report_blocked_calls = [
            call for call in control_calls if call.tool_name == "runtime.report_blocked"
        ]
        if report_blocked_calls:
            if len(report_blocked_calls) != 1 or len(control_calls) != 1:
                raise ValueError(
                    "runtime.report_blocked must be the only runtime-control call"
                )
            if physical_calls:
                raise ValueError(
                    "runtime.report_blocked cannot be mixed with physical tool calls"
                )
            report = report_blocked_calls[0]
            reason = str(report.args.get("reason", "")).strip()
            evidence_refs = report.args.get("evidence_refs")
            if not reason:
                raise ValueError("runtime.report_blocked reason must not be empty")
            if not isinstance(evidence_refs, list) or not evidence_refs:
                raise ValueError(
                    "runtime.report_blocked requires at least one evidence reference"
                )
            normalized_refs = [str(item).strip() for item in evidence_refs]
            if any(not item for item in normalized_refs):
                raise ValueError(
                    "runtime.report_blocked evidence references must not be empty"
                )
            available_refs = self.agent_transcript.prior_evidence_reference_ids()
            unknown_refs = sorted(set(normalized_refs) - available_refs)
            if unknown_refs:
                raise ValueError(
                    "runtime.report_blocked references evidence outside the current "
                    f"instruction transcript: {', '.join(unknown_refs)}"
                )
            return CoupledActionPhasePlan(
                phase=phase,
                phase_decision=PhaseTransition(
                    STOP_TASK_INCOMPLETE,
                    f"agent reported blocker: {report.args['blocking_condition']}",
                ),
                agent_messages=[message],
                completion_assessment=assessment,
            )

        if not 1 <= len(physical_calls) <= 3:
            raise ValueError("an incomplete phase requires one to three physical tool calls")
        if any(
            self.tools.registry.get(call.tool_name).runtime.batching != "ordered"
            for call in physical_calls
        ):
            raise ValueError("physical action-window tools must declare ordered batching")
        strategy_data = data.get("phase_strategy")
        window_data = data.get("action_window")
        if not isinstance(strategy_data, dict) or not isinstance(window_data, dict):
            raise ValueError("an incomplete phase requires phase_strategy and action_window")
        strategy = PhaseStrategy.from_dict(dict(strategy_data))
        window = ActionWindow(
            intent=str(window_data.get("intent", "")).strip(),
            expected_effect=str(window_data.get("expected_effect", "")).strip(),
        )
        if not window.intent or not window.expected_effect:
            raise ValueError("action_window intent and expected_effect must not be empty")
        steps = [
            PlannedStep.tool_step(call.tool_name, call.args, public_intent)
            for call in physical_calls
        ]
        controls = [
            RuntimeControlAction(
                name=str(self.tools.registry.get(call.tool_name).metadata["runtime_action"]),
                reason=public_intent,
                payload=dict(call.args),
            )
            for call in control_calls
        ]
        return CoupledActionPhasePlan(
            phase=phase,
            phase_decision=PhaseTransition.continue_current_phase(
                "completion evidence still requires progress"
            ),
            steps=steps,
            action_window=window,
            agent_messages=[message],
            control_actions=controls,
            completion_assessment=assessment,
            phase_strategy=strategy,
        )

    def _reject_native_turn(self, calls: list[AgentToolCall], reason: str) -> None:
        for call in calls:
            self.agent_transcript.append_tool_result(
                ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="rejected",
                    message=reason,
                    requested_args=dict(call.args),
                )
            )
        self.agent_transcript.append_protocol_feedback(
            f"The previous tool-call turn was rejected: {reason}. Correct it using the available tools."
        )

    def _append_native_control_results(
        self,
        native_meta: dict[str, Any],
        results: list[Any],
    ) -> None:
        calls = native_meta.get("control_calls", [])
        for call, result in zip(calls, results):
            transcript_result = ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="ok" if result.status in {"accepted", "pending"} else "error",
                    message=result.reason,
                    data=result.to_dict(),
                    requested_args=dict(call.args),
                    executed_args=dict(call.args),
                )
            self.agent_transcript.append_tool_result(transcript_result)
            self._append_native_round_tool_result(native_meta, transcript_result)

    def _append_native_commit_result(
        self,
        native_meta: dict[str, Any],
        *,
        status: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        call = native_meta.get("commit_call")
        if not isinstance(call, AgentToolCall):
            return
        transcript_result = ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status=status,
                message=message,
                data=dict(data or {}),
                requested_args=dict(call.args),
                executed_args=dict(call.args),
            )
        self.agent_transcript.append_tool_result(transcript_result)
        self._append_native_round_tool_result(native_meta, transcript_result)

    @staticmethod
    def _append_native_round_tool_result(
        native_meta: dict[str, Any], result: ToolResult
    ) -> None:
        """Attach post-acceptance ToolResults to the accepted model round."""

        rounds = native_meta.get("model_rounds")
        if not isinstance(rounds, list) or not rounds:
            return
        final_round = rounds[-1]
        if not isinstance(final_round, dict):
            return
        outcome = final_round.get("runtime_outcome")
        if not isinstance(outcome, dict):
            outcome = {"status": "commit_accepted", "feedback": None}
            final_round["runtime_outcome"] = outcome
        results = outcome.setdefault("tool_results", [])
        if isinstance(results, list):
            results.append(result.to_dict())

    def _finish_agent_reported_blocked(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        native_meta: dict[str, Any],
        decision_payload: dict[str, Any],
    ) -> None:
        """Safely close a model-declared terminal blocker and its ToolCalls."""

        call = native_meta.get("report_blocked_call")
        if not isinstance(call, AgentToolCall):
            raise RuntimeError(
                "stop_task_incomplete requires one runtime.report_blocked call"
            )
        blocking_condition = str(call.args.get("blocking_condition", "")).strip()
        observation_before_id = state.latest_observation.get("observation_id")
        self._try_hover(scene, agent_id, state)
        final_observation: dict[str, Any] | None = None
        observation_error: dict[str, str] | None = None
        try:
            final_observation = self._record_observation(
                scene,
                agent_id,
                state,
                list(state.observation_modalities),
                source="agent_reported_blocked",
            )
            self._event(
                "observation_updated",
                self._observation_summary_event(
                    final_observation, source="agent_reported_blocked"
                ),
            )
        except Exception as exc:
            observation_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._debug_event(
                "agent_reported_blocked_final_observation_failed",
                observation_error,
            )

        stopped_reason = f"agent_reported_blocked:{blocking_condition}"
        state.mark_stopped(stopped_reason, source="runtime.report_blocked")
        report_data: dict[str, Any] = {
            "blocking_condition": blocking_condition,
            "reason": str(call.args.get("reason", "")).strip(),
            "evidence_refs": list(call.args.get("evidence_refs", [])),
            "suggested_user_action": call.args.get("suggested_user_action"),
            "observation_before_id": observation_before_id,
            "observation_after_id": (
                final_observation.get("observation_id")
                if isinstance(final_observation, dict)
                else None
            ),
        }
        if observation_error is not None:
            report_data["final_observation_error"] = observation_error
        report_result = ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="ok",
                message="current instruction stopped as incomplete",
                data=report_data,
                requested_args=dict(call.args),
                executed_args=dict(call.args),
            )
        self.agent_transcript.append_tool_result(report_result)
        self._append_native_round_tool_result(native_meta, report_result)
        self._append_native_commit_result(
            native_meta,
            status="ok",
            message="blocked phase assessment committed without a physical action window",
            data=report_data,
        )
        try:
            self._append_terminal_native_state_checkpoint(scene, state)
        except Exception as exc:
            checkpoint_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            report_data["terminal_checkpoint_error"] = checkpoint_error
            self._debug_event(
                "agent_reported_blocked_terminal_checkpoint_failed",
                checkpoint_error,
            )
        terminal_payload = {
            **decision_payload,
            "completed": False,
            "stopped": True,
            "stopped_reason": stopped_reason,
            "stop_source": "runtime.report_blocked",
            **report_data,
        }
        self.agent_transcript.append_task_terminal(terminal_payload)
        self._event("agent_reported_blocked", terminal_payload)
        self._write_state(state)

    def _runtime_control_recommendation_context(self, state: MissionState) -> dict[str, Any]:
        """Return compact runtime evidence used only by runtime-control recommendations."""

        return {
            "phase_execution_summary": state.current_phase_execution_summary,
            "completed": state.completed,
            "stopped": state.stopped,
        }

    def _apply_phase_transition(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        plan: ShortHorizonPlan,
        safety_decision: SafetyDecision,
        transition: PhaseTransition,
    ) -> None:
        transition_source = "coupled_planner"
        payload = {
            "phase": plan.phase,
            "decision": transition.decision,
            "reason": transition.reason,
            "transition_source": transition_source,
            "current_phase_window_count": state.current_phase_window_count,
            "max_phase_window_count": state.max_phase_window_count,
            "phase_progress_evidence": self._phase_progress_evidence(state),
        }
        self._agent_messages(transition.agent_messages, source=transition_source, phase=plan.phase)
        if safety_decision.interventions and not self._safety_allows_phase_transition(safety_decision):
            self._debug_event(
                "phase_transition_suppressed_by_safety",
                {**payload, "interventions": safety_decision.interventions},
            )
            if state.current_phase_window_count >= state.max_phase_window_count:
                self._stop_for_runtime_loop_guard(scene, agent_id, state, payload, "safety_intervention_loop_guard")
            return
        if (
            transition.decision == CONTINUE_CURRENT_PHASE
            and self._visual_phase_budget_exhausted(payload, safety_decision)
        ):
            self._stop_for_visual_search_budget(scene, agent_id, state, payload, safety_decision)
            return
        if transition.decision == CONTINUE_CURRENT_PHASE:
            self._event("phase_transition_continued", payload)
            if state.current_phase_window_count >= state.max_phase_window_count:
                self._stop_for_runtime_loop_guard(scene, agent_id, state, payload, "continue_current_phase_loop_guard")
            return
        if transition.decision == ADVANCE_TO_NEXT_PHASE:
            state.advance_phase()
            self._mark_phase_start_observation(state)
            self._event("phase_transition_applied", payload)
            return
        if transition.decision == STOP_TASK_INCOMPLETE:
            if not state.stopped or state.stop_source != "runtime.report_blocked":
                raise RuntimeError(
                    "stop_task_incomplete transition was not closed by runtime.report_blocked"
                )
            self._event("phase_transition_stopped", payload)
            return
        if transition.decision == COMPLETE_TASK:
            if not state.is_final_phase:
                downgraded_payload = {
                    **payload,
                    "original_decision": COMPLETE_TASK,
                    "applied_decision": ADVANCE_TO_NEXT_PHASE,
                    "guard_reason": "complete_task_before_final_phase",
                }
                self._debug_event("phase_complete_task_downgraded", downgraded_payload)
                state.advance_phase()
                self._mark_phase_start_observation(state)
                self._event("phase_transition_applied", downgraded_payload)
                return
            state.mark_completed()
            self._event("phase_transition_applied", payload)
            return
        raise ValueError(f"Unsupported phase transition decision: {transition.decision}")

    def _stop_for_runtime_loop_guard(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        payload: dict[str, Any],
        reason: str,
    ) -> None:
        guard_payload = {
            **payload,
            "reason": reason,
            "runtime_loop_guard_windows": state.max_phase_window_count,
            "applied_decision": "stop_task_incomplete",
        }
        self._event("runtime_loop_guard_reached", guard_payload)
        self._event("phase_transition_window_limit_reached", guard_payload)
        self._try_hover(scene, agent_id, state)
        self._append_terminal_native_state_checkpoint(scene, state)
        state.mark_stopped("runtime_loop_guard_reached", source=reason)

    def _stop_for_visual_search_budget(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        payload: dict[str, Any],
        safety_decision: SafetyDecision,
    ) -> None:
        stop_payload = {
            **payload,
            "reason": "visual_search_budget_exhausted",
            "applied_decision": "stop_task_incomplete",
            "safety_intervention_summary": self._safety_intervention_summary(safety_decision, state),
        }
        self._event("visual_search_budget_exhausted", stop_payload)
        self._try_hover(scene, agent_id, state)
        self._append_terminal_native_state_checkpoint(scene, state)
        state.mark_stopped("visual_search_budget_exhausted", source="safety_yaw_scan_budget")

    def _append_terminal_native_state_checkpoint(
        self,
        scene: SceneInstance,
        state: MissionState,
    ) -> None:
        """Close a native physical batch when no later model turn will run.

        Ordinarily the next phase-window iteration appends the checkpoint for
        the automatic post-action observation. A runtime stop has no next
        model iteration, so persist that same current-state payload here. The
        observation ID guard prevents duplicate checkpoints if a future stop
        path has already recorded the latest state.
        """

        if not callable(getattr(self.planner, "plan_turn", None)):
            return
        observation = dict(state.latest_observation)
        observation_id = observation.get("observation_id")
        if (
            not observation_id
            or self.agent_transcript.latest_observation_id() == observation_id
        ):
            return
        context = self.context_builder.build(state, self._memory_for_state(state))
        runtime_control_context = self.runtime_control.model_context(
            self._runtime_control_recommendation_context(state)
        )
        phase_window_context = self.phase_window_context_builder.build(
            state,
            context,
            observation,
            runtime_control_context=runtime_control_context,
            scene_capabilities=scene.capabilities,
        )
        self.agent_transcript.append_state(
            phase_window_context.planner_payload,
            str(observation_id),
        )

    def _phase_progress_evidence(self, state: MissionState) -> dict[str, Any]:
        return self.phase_window_context_builder.phase_progress_evidence(state)

    def _safety_allows_phase_transition(self, safety_decision: SafetyDecision) -> bool:
        if not safety_decision.interventions:
            return True
        intervention_items = [item for item in safety_decision.interventions if isinstance(item, dict)]
        if len(intervention_items) != len(safety_decision.interventions):
            return False
        return all(
            intervention.get("reason") == "yaw_scan_budget_exceeded"
            and intervention.get("replacement") == "uav.hover"
            for intervention in intervention_items
        )

    def _visual_phase_budget_exhausted(
        self,
        payload: dict[str, Any],
        safety_decision: SafetyDecision,
    ) -> bool:
        return any(reason == "yaw_scan_budget_exceeded" for reason in self._safety_intervention_reasons(safety_decision))

    def _safety_intervention_summary(
        self,
        safety_decision: SafetyDecision,
        state: MissionState,
        executed_steps: list[dict] | None = None,
    ) -> dict[str, Any]:
        reasons = self._safety_intervention_reasons(safety_decision)
        replacements = [
            str(item.get("replacement", ""))
            for item in safety_decision.interventions
            if isinstance(item, dict) and item.get("replacement")
        ]
        hover_replacement_count = sum(1 for replacement in replacements if replacement == "uav.hover")
        return {
            "intervention_count": len(safety_decision.interventions),
            "reasons": reasons,
            "replacements": replacements,
            "hover_replacement_count": hover_replacement_count,
            "all_replacements_hover": bool(replacements) and hover_replacement_count == len(replacements),
            "yaw_scan_budget_exhausted": "yaw_scan_budget_exceeded" in reasons,
            "yaw_scan_offset_deg": float(getattr(state, "yaw_scan_offset_deg", 0.0) or 0.0),
            "yaw_scan_extent_deg": float(getattr(state, "yaw_scan_extent_deg", 0.0) or 0.0),
            "yaw_scan_remaining_deg": self._yaw_scan_remaining_deg(state),
            "executed_tools": self._executed_tool_names(executed_steps or []),
        }

    def _safety_intervention_reasons(self, safety_decision: SafetyDecision) -> list[str]:
        return [
            str(item.get("reason", ""))
            for item in safety_decision.interventions
            if isinstance(item, dict) and item.get("reason")
        ]

    def _yaw_scan_remaining_deg(self, state: MissionState) -> float:
        max_yaw_scan = float(self.safety.safety_config.max_yaw_scan_extent_deg)
        yaw_scan_extent = float(getattr(state, "yaw_scan_extent_deg", 0.0) or 0.0)
        return max(0.0, max_yaw_scan - yaw_scan_extent)

    def _maybe_recharge_visual_search_budget(
        self,
        state: MissionState,
        measurements: dict[str, Any],
        *,
        phase: str,
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
            "phase": phase,
            "phase_key": state.current_phase_key,
            "window_index": state.current_phase_window_count,
            "planar_displacement_m": round(planar_m, 6),
            "vertical_displacement_m": round(vertical_m, 6),
            "translation_3d_m": round(translation_3d_m, 6),
            "recharge_distance_m": threshold_m,
            **recharge,
        }
        self._event("visual_search_budget_recharged", payload)
        return payload

    def _executed_tool_names(self, executed_steps: list[dict]) -> list[str]:
        names = []
        for executed in executed_steps:
            step = executed.get("step", {}) if isinstance(executed, dict) else {}
            if isinstance(step, dict) and step.get("tool"):
                names.append(str(step["tool"]))
        return names

    def _tool_facts_from_executed_steps(self, executed_steps: list[dict]) -> dict:
        facts: dict[str, object] = {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "turn_yaw_attempted_sum_deg": 0.0,
            "turn_yaw_success_sum_deg": 0.0,
            "move_duration_attempted_sum_sec": 0.0,
            "move_duration_success_sum_sec": 0.0,
            "move_duration_by_direction_success": {},
            "tools_executed": {},
        }
        directions: dict[str, float] = {}
        tools: dict[str, int] = {}
        for executed in executed_steps:
            step = executed.get("step", {}) if isinstance(executed, dict) else {}
            result = executed.get("result", {}) if isinstance(executed, dict) else {}
            tool = str(step.get("tool", ""))
            args = step.get("args", {}) if isinstance(step.get("args", {}), dict) else {}
            ok = result.get("ok") is True
            facts["total_steps"] = int(facts["total_steps"]) + 1
            facts["successful_steps" if ok else "failed_steps"] = int(facts["successful_steps" if ok else "failed_steps"]) + 1
            tools[tool] = tools.get(tool, 0) + 1
            if tool == "uav.turn_relative":
                yaw = float(args.get("yaw_deg", 0.0) or 0.0)
                facts["turn_yaw_attempted_sum_deg"] = round(float(facts["turn_yaw_attempted_sum_deg"]) + yaw, 3)
                if ok:
                    facts["turn_yaw_success_sum_deg"] = round(float(facts["turn_yaw_success_sum_deg"]) + yaw, 3)
            if tool == "uav.move_body":
                duration = float(args.get("duration", 0.0) or 0.0)
                facts["move_duration_attempted_sum_sec"] = round(float(facts["move_duration_attempted_sum_sec"]) + duration, 3)
                if ok:
                    facts["move_duration_success_sum_sec"] = round(float(facts["move_duration_success_sum_sec"]) + duration, 3)
                    direction = str(args.get("direction", "velocity_body"))
                    directions[direction] = round(directions.get(direction, 0.0) + duration, 3)
        facts["move_duration_by_direction_success"] = directions
        facts["tools_executed"] = tools
        return facts

    def _pose_from_observation(self, observation: dict) -> dict:
        pose = observation.get("pose") if isinstance(observation, dict) else None
        return dict(pose) if isinstance(pose, dict) else {}

    def _apply_automatic_observation_policy(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        phase: str,
    ) -> dict:
        modalities = list(state.observation_modalities)
        policy = {
            "mode": "automatic",
            "modalities": modalities,
            "sample_timing": "after_each_phase_window",
            "phase": phase,
        }
        self._debug_event("automatic_observation_policy_applied", policy)
        data = self._record_observation(scene, agent_id, state, modalities, source="automatic_phase_window")
        self._debug_event("automatic_observation_updated", data)
        self._event("observation_updated", self._observation_summary_event(data, source="automatic_phase_window"))
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
        write_result = self.memory_writer.write_observation(data)
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
                overwrite=overwrite_task_initial,
            )
            memory.set_observation_anchor("phase_start", write_result["observation_ref"], overwrite=True)
        self.recorder.write_json("debug/latest_observation.json", data)
        self.recorder.write_json("contexts/latest_context.json", self.context_builder.build(state, self._memory_for_state(state)))
        return data

    def _mark_phase_start_observation(self, state: MissionState) -> None:
        """Anchor the latest observation as the start of the newly active phase."""

        memory = self._memory_for_state(state)
        memory.clear_observation_anchor("before_last_action_window")
        if not memory.recent_observation_refs:
            return
        memory.set_observation_anchor("phase_start", memory.recent_observation_refs[-1], overwrite=True)

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
        except Exception as exc:
            self._debug_event("safety_hover_failed", {"error_type": type(exc).__name__, "error": str(exc)})

    def _run_shutdown(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
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
        state_data = state.to_dict()
        self.recorder.write_json("state/latest_state.json", state_data)
        self.recorder.write_json("state/flight_memory.json", self._memory_for_state(state).to_dict())
        self._debug_event("mission_state_updated", self._state_summary(state))

    def _apply_completed_semantic_plan_update(self, state: MissionState) -> None:
        result = self.runtime_control.take_completed_semantic_plan_update()
        if result is None:
            return
        payload = result.to_dict()
        if result.status != "completed":
            self._debug_event("runtime_control_action_completed", {"name": "semantic_plan_update", **payload})
            return
        discard_reason = ""
        if not self.runtime_control.config.is_enabled("semantic_plan_update"):
            discard_reason = "runtime_control_action_disabled"
        elif not self.runtime_control.config.is_exposed("semantic_plan_update"):
            discard_reason = "runtime_control_action_not_exposed"
        elif result.job.run_id != state.run_id:
            discard_reason = "run_id_mismatch"
        elif result.job.phase_key != state.current_phase_key:
            discard_reason = "phase_key_mismatch"
        elif result.job.semantic_plan_version != state.semantic_plan_version:
            discard_reason = "semantic_plan_version_mismatch"
        if discard_reason:
            self._debug_event("semantic_plan_update_discarded", {**payload, "discard_reason": discard_reason})
            return
        try:
            current_phase = state.semantic_plan.phases[state.current_phase_index].to_dict()
            validate_semantic_plan_patch_numeric_provenance(
                result.patch,
                instruction=state.user_instruction,
                current_phase=current_phase,
            )
            updated_plan = apply_semantic_plan_patch(
                state.semantic_plan,
                state.current_phase_index,
                result.patch,
            )
        except Exception as exc:
            self._debug_event(
                "semantic_plan_update_discarded",
                {**payload, "discard_reason": "invalid_patch", "error_type": type(exc).__name__, "error": str(exc)},
            )
            return
        state.apply_semantic_plan_update(updated_plan)
        self.recorder.write_json(
            "state/semantic_plan.json",
            {
                "instruction": state.user_instruction,
                "parser_mode": "runtime_updated",
                "validation_ok": True,
                "plan": state.semantic_plan.to_dict(),
                "semantic_plan_version": state.semantic_plan_version,
            },
        )
        self._event(
            "runtime_control_action_completed",
            {
                "name": "semantic_plan_update",
                "status": "applied",
                "phase_key": state.current_phase_key,
                "semantic_plan_version": state.semantic_plan_version,
            },
        )
        self._debug_event("semantic_plan_update_applied", payload)

    def _submit_semantic_plan_updates_after_window(
        self,
        state: MissionState,
        actions,
        results,
        *,
        phase_key: str,
        semantic_plan_version: int,
        observation_before: dict[str, Any],
        observation_after: dict[str, Any],
        selected_skill_guidance: list[dict[str, str]],
    ) -> None:
        for action, result in zip(actions, results):
            if action.name != "semantic_plan_update" or result.status != "accepted":
                continue
            job = SemanticPlanUpdateJob(
                run_id=state.run_id,
                phase_key=phase_key,
                semantic_plan_version=semantic_plan_version,
                request_payload=dict(action.payload),
                updater_context=self._semantic_plan_update_context(
                    state,
                    action,
                    phase_key=phase_key,
                    semantic_plan_version=semantic_plan_version,
                    observation_before=observation_before,
                    observation_after=observation_after,
                    selected_skill_guidance=selected_skill_guidance,
                ),
            )
            submit_result = self.runtime_control.submit_semantic_plan_update(job)
            self._debug_event(
                "runtime_control_action_started",
                {
                    "name": "semantic_plan_update",
                    "status": submit_result.status,
                    "reason": submit_result.reason,
                    "phase_key": phase_key,
                    "semantic_plan_version": semantic_plan_version,
                },
            )

    def _semantic_plan_update_context(
        self,
        state: MissionState,
        action,
        *,
        phase_key: str,
        semantic_plan_version: int,
        observation_before: dict[str, Any],
        observation_after: dict[str, Any],
        selected_skill_guidance: list[dict[str, str]],
    ) -> dict[str, Any]:
        current_phase = {}
        if state.current_phase_index < len(state.semantic_plan.phases):
            current_phase = state.semantic_plan.phases[state.current_phase_index].to_dict()
        context = {
            "run_id": state.run_id,
            "user_instruction": state.user_instruction,
            "semantic_plan": state.semantic_plan.to_dict(),
            "semantic_plan_version": semantic_plan_version,
            "current_phase_key": phase_key,
            "current_phase_index": state.current_phase_index,
            "current_phase": current_phase,
            "planner_request": {
                "reason": action.reason,
                "payload": dict(action.payload),
            },
            "phase_execution_summary": state.current_phase_execution_summary,
            "phase_progress_evidence": self._phase_progress_evidence(state),
            "observation_before": _clean_observation_for_semantic_update(observation_before),
            "observation_after": _clean_observation_for_semantic_update(observation_after),
        }
        if selected_skill_guidance:
            context["selected_skill_guidance"] = selected_skill_guidance
        if self.semantic_evidence_capabilities:
            context["configured_evidence"] = list(self.semantic_evidence_capabilities)
        return context

    def _configured_semantic_evidence_capabilities(self) -> list[str]:
        """Read configured context capabilities without requiring an observation."""

        provider = getattr(self.context_builder, "semantic_evidence_capabilities", None)
        if not callable(provider):
            return []
        try:
            capabilities = provider()
        except Exception:
            return []
        if not isinstance(capabilities, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in capabilities:
            if not isinstance(item, str):
                continue
            value = " ".join(item.split())
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
        return normalized

    def _refresh_semantic_evidence_capabilities(self) -> list[str]:
        """Rediscover configured semantic evidence and update the parser.

        The registry owns which enabled context modules can contribute model
        evidence. Runtime only refreshes and forwards that bounded declaration;
        it does not hardcode modality, simulator, or benchmark knowledge.
        """

        capabilities = self._configured_semantic_evidence_capabilities()
        self.semantic_evidence_capabilities = capabilities
        configure_evidence = getattr(
            self.task_parser,
            "configure_evidence_capabilities",
            None,
        )
        if callable(configure_evidence):
            configure_evidence(capabilities)
        return list(capabilities)

    @staticmethod
    def _selected_semantic_update_guidance(planner_context: dict[str, Any]) -> list[dict[str, str]]:
        fragments = planner_context.get("fragments", []) if isinstance(planner_context, dict) else []
        if not isinstance(fragments, list):
            return []
        selected: list[dict[str, str]] = []
        seen: set[str] = set()
        for fragment in fragments:
            if not isinstance(fragment, dict) or fragment.get("name") != "skills":
                continue
            content = fragment.get("content", {})
            skills = content.get("selected_skills", []) if isinstance(content, dict) else []
            if not isinstance(skills, list):
                continue
            for skill in skills:
                if not isinstance(skill, dict):
                    continue
                name = str(skill.get("name", "")).strip()
                skill_text = str(skill.get("skill_text", "")).strip()
                if not name or not skill_text or name in seen:
                    continue
                selected.append({"name": name, "skill_text": skill_text})
                seen.add(name)
        return selected

    @staticmethod
    def _planner_skill_selection_trace(planner_context: dict[str, Any]) -> dict[str, Any]:
        """Extract a compact, non-secret Skill audit record from context."""

        fragments = planner_context.get("fragments", []) if isinstance(planner_context, dict) else []
        if not isinstance(fragments, list):
            return {}
        for fragment in fragments:
            if not isinstance(fragment, dict) or fragment.get("name") != "skills":
                continue
            metadata = fragment.get("metadata", {})
            selection = metadata.get("selection", {}) if isinstance(metadata, dict) else {}
            content = fragment.get("content", {})
            loaded = content.get("selected_skills", []) if isinstance(content, dict) else []
            loaded_names = [
                str(item.get("name", "")).strip()
                for item in loaded
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            ] if isinstance(loaded, list) else []
            return {
                "selection_status": selection.get("status"),
                "requested_skills": list(selection.get("requested_skill_names", [])),
                "accepted_skills": list(selection.get("accepted_skill_names", [])),
                "loaded_skills": loaded_names,
                "selection_source": metadata.get("selection_source"),
            }
        return {}

    def _record_runtime_control_results(self, actions, results) -> None:
        for action, result in zip(actions, results):
            request_payload = {
                "name": action.name,
                "reason": action.reason,
                "status": result.status,
            }
            self._event("runtime_control_action_requested", request_payload)
            self._debug_event(
                "runtime_control_action_completed",
                {
                    "request": action.to_dict(),
                    "result": result.to_dict(),
                },
            )

    def _event(self, event: str, payload: dict) -> None:
        self.recorder.event(event, payload)
        if event in {
            "instruction_started",
            "phase_plan_ready",
            "coupled_planner_phase_decided",
            "tool_step_completed",
            "observation_updated",
            "phase_transition_applied",
            "phase_transition_continued",
            "phase_transition_window_limit_reached",
            "runtime_loop_guard_reached",
        }:
            self._emit_runtime_event({"kind": event, "payload": dict(payload)})

    def _debug_event(self, event: str, payload: dict) -> None:
        debug_event = getattr(self.recorder, "debug_event", None)
        if callable(debug_event):
            debug_event(event, payload)
        else:
            self.recorder.event(event, payload)

    def _model_dependency_status(self) -> dict[str, Any]:
        """Return redacted wiring facts for human audit.

        This deliberately reports dependency identity rather than credential
        values or filesystem paths.  It makes a worktree configuration split
        visible without leaking secrets into planner context or run artifacts.
        """

        composer = getattr(self.context_builder, "composer", None)
        registry = getattr(composer, "registry", None)
        modules = getattr(registry, "modules", {})
        skills_module = modules.get("skills") if isinstance(modules, dict) else None
        selector = getattr(skills_module, "skill_selector", None)
        updater = self.runtime_control.semantic_plan_update_manager.updater
        return {
            "model_config_source": (
                "task_runtime_injected" if self.model_config is not None else "none"
            ),
            "model_config_has_api_key": bool(
                self.model_config and self.model_config.api_key
            ),
            "model_provider": (
                self.model_config.provider if self.model_config is not None else None
            ),
            "model_name": (
                self.model_config.model_name if self.model_config is not None else None
            ),
            "planner_skill_selection_enabled": bool(
                selector is not None and getattr(selector, "enabled", False)
            ),
            "planner_skill_selector_uses_runtime_config": bool(
                selector is not None
                and self.model_config is not None
                and getattr(selector, "model_config", None) is self.model_config
            ),
            "semantic_updater_configured": updater is not None,
            "semantic_updater_uses_runtime_config": bool(
                updater is not None
                and self.model_config is not None
                and getattr(updater, "model_config", None) is self.model_config
            ),
        }

    def _agent_messages(self, messages: list[AgentMessage], *, source: str, phase: str | None = None) -> None:
        for message in messages:
            self._agent_message(message, source=source, phase=phase)

    def _agent_message(self, message: AgentMessage, *, source: str, phase: str | None = None) -> None:
        message_data = message.to_dict()
        message_ref = self.memory_writer.write_agent_message(source, phase, message_data)
        memory = self._state_flight_memories.get(self.recorder.run_dir.name)
        if memory is not None:
            memory.append_agent_message_ref(message_ref)
        snapshot = {
            "source": source,
            "phase": phase,
            "message": message_data,
            "message_ref": dict(message_ref),
        }
        self.recorder.write_json("state/latest_agent_message.json", snapshot)
        if message.type == "final":
            self.recorder.write_json("state/final_answer.json", snapshot)
        payload = {
            "source": source,
            "phase": phase,
            "message": message_data,
            "message_ref": dict(message_ref),
        }
        self._current_turn_agent_messages.append(
            {
                "source": source,
                "phase": phase,
                **payload["message"],
            }
        )
        self._emit_runtime_event(
            {
                "kind": "agent_message",
                "source": source,
                "phase": phase,
                "message": payload["message"],
            }
        )
        self._event("agent_message", payload)

    def latest_turn_agent_messages(self) -> list[dict]:
        """Return user-facing messages produced during the current instruction.

        This is the synchronous counterpart to the public ``agent_message``
        events. Interactive shells use it for immediate summaries, while
        offline tools continue to read events.jsonl.
        """

        return [dict(item) for item in self._current_turn_agent_messages]

    def _emit_runtime_event(self, event: dict[str, Any]) -> None:
        """Push a live event to CLI/TUI consumers without changing run history.

        The sink is deliberately best-effort: user interface rendering should not
        be able to fail a flight task. Durable replay and debugging still use the
        recorder-backed public/debug events.
        """

        if self.event_sink is None:
            return
        try:
            self.event_sink(dict(event))
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._debug_event(
                "runtime_event_sink_failed",
                {"error_type": type(exc).__name__, "error": str(exc), "event_kind": event.get("kind")},
            )

    def _write_phase_window_debug_artifact(self, phase_key: str, window_index: int, payload: dict) -> str:
        name = f"debug/phase_windows/{self._safe_file_part(phase_key)}_window_{window_index:03d}.json"
        self.recorder.write_json(name, payload)
        return name

    def _write_action_planner_debug_artifact(
        self,
        state: MissionState,
        planner_context: dict,
        model_context: dict,
        raw_plan: ShortHorizonPlan,
        safe_plan: ShortHorizonPlan,
        safety_decision: SafetyDecision,
        coupled_plan: CoupledActionPhasePlan | None = None,
        decision_meta: dict[str, Any] | None = None,
        native_meta: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.enable_action_planner_debug:
            return None
        meta = dict(decision_meta or {})
        decision_id = str(meta.get("decision_id") or "").strip()
        if decision_id:
            name = (
                "debug/action_planner/"
                f"{self._safe_file_part(state.current_phase_key)}_"
                f"{self._safe_file_part(decision_id)}.json"
            )
        else:
            # Compatibility for deterministic tests and callers that invoke
            # this helper outside a decision cycle.
            name = f"debug/action_planner/{self._safe_file_part(state.current_phase_key)}_window_{state.current_phase_window_count:03d}.json"
        payload = {
            "schema_version": "agent_decision_debug_v2",
            **meta,
            "phase": state.current_phase_name,
            "phase_key": state.current_phase_key,
            "window_index": meta.get(
                "physical_window_index", state.current_phase_window_count
            ),
            # Retained as a legacy replay aid. New reviews use model_rounds,
            # which are the actual provider exchanges rather than a context
            # reconstruction.
            "planner_context": _redact_model_input_context(planner_context),
            "model_context": model_context,
            "model_rounds": self._persist_provider_exchange_media(
                list((native_meta or {}).get("model_rounds", []))
            ),
            "raw_plan": raw_plan.to_dict(),
            "safe_plan": safe_plan.to_dict(),
            "safety_decision": {
                "interventions": list(safety_decision.interventions),
                "plan_changed": bool(safety_decision.interventions),
            },
        }
        if coupled_plan is not None:
            payload["coupled_plan"] = coupled_plan.to_dict()
            payload["coupled_phase_decision"] = coupled_plan.phase_decision.to_dict()
        self.recorder.write_json(name, payload)
        return name

    def _write_failed_action_planner_debug_artifact(
        self,
        state: MissionState,
        planner_context: dict[str, Any],
        model_context: dict[str, Any],
        decision_meta: dict[str, Any],
        model_rounds: list[dict[str, Any]],
    ) -> str | None:
        """Persist provider failure/protocol exhaustion without fabricating a plan."""

        if not self.enable_action_planner_debug:
            return None
        decision_id = self._safe_file_part(decision_meta.get("decision_id"))
        name = (
            "debug/action_planner/"
            f"{self._safe_file_part(state.current_phase_key)}_{decision_id}.json"
        )
        self.recorder.write_json(
            name,
            {
                "schema_version": "agent_decision_debug_v2",
                **decision_meta,
                "phase": state.current_phase_name,
                "phase_key": state.current_phase_key,
                "window_index": None,
                "planner_context": _redact_model_input_context(planner_context),
                "model_context": model_context,
                "model_rounds": self._persist_provider_exchange_media(model_rounds),
                "runtime_failure": True,
            },
        )
        return name

    def _persist_provider_exchange_media(
        self, model_rounds: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Write request media once per digest and return base64-free rounds."""

        persisted = copy.deepcopy(model_rounds)
        save_bytes = getattr(self.recorder, "save_bytes", None)
        for model_round in persisted:
            media = model_round.get("request_media")
            if not isinstance(media, list):
                continue
            for item in media:
                if not isinstance(item, dict):
                    continue
                data = item.pop("data", None)
                digest = str(item.get("sha256") or "").strip().lower()
                suffix = str(item.get("suffix") or ".bin")
                if (
                    isinstance(data, bytes)
                    and digest
                    and callable(save_bytes)
                ):
                    relative = f"debug/provider_media/{digest}{suffix}"
                    if digest not in self._persisted_provider_media:
                        save_bytes(relative, data)
                        self._persisted_provider_media.add(digest)
                    item["asset_ref"] = relative
                elif data is not None:
                    item["unavailable_reason"] = "recorder_cannot_persist_media"
        return persisted

    @staticmethod
    def _safe_file_part(value: str | None) -> str:
        text = str(value or "phase").strip().lower()
        safe = re.sub(r"[^a-z0-9_.-]+", "_", text)
        return safe.strip("._-") or "phase"

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
            "run_id": state.run_id,
            "current_phase": state.current_phase_name,
            "current_phase_index": state.current_phase_index,
            "completed": state.completed,
            "stopped": state.stopped,
            "stopped_reason": state.stopped_reason,
            "stop_source": state.stop_source,
            "connected": state.connected,
            "airborne": state.airborne,
            "phase_window_count": state.current_phase_window_count,
            "phase_strategy_revision": state.current_phase_strategy_revision,
            "phase_strategy_active_stage": state.current_phase_strategy.get("active_stage"),
        }

    @staticmethod
    def _scene_key(map_id: str, scene_id: str | None) -> tuple[str, str | None]:
        return (map_id, str(scene_id) if scene_id is not None else None)

    def _forget_scene(self, scene: SceneInstance) -> None:
        for key, active_scene in list(self._active_reused_scenes.items()):
            if active_scene.scene_instance_id == scene.scene_instance_id:
                self._active_reused_scenes.pop(key, None)
        for key, memory in list(self._active_flight_memories.items()):
            if memory.scene_instance_id == scene.scene_instance_id:
                self._active_flight_memories.pop(key, None)

    def _resolve_flight_memory(
        self,
        map_id: str,
        scene: SceneInstance,
        agent_id: str,
        scene_policy: str,
        parse_result: TaskParseResult,
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
        instruction_ref = self.memory_writer.write_instruction(parse_result.instruction, parse_result.plan.to_dict())
        memory.update_instruction_ref(instruction_ref, parse_result.instruction)
        return memory

    def _memory_for_state(self, state: MissionState) -> FlightMemory:
        memory = self._state_flight_memories.get(state.run_id)
        if memory is None:
            raise RuntimeError(f"FlightMemory missing for run: {state.run_id}")
        return memory

    def _record_action(self, state: MissionState, tool: str, args: dict, result: dict) -> dict[str, Any]:
        state.record_action(tool, args, result)
        write_result = self.memory_writer.write_action(tool, args, result)
        memory = self._memory_for_state(state)
        memory.append_action_ref(write_result["action_ref"])
        memory.append_pose_ref(write_result.get("pose_ref"))
        return write_result


def _clean_observation_for_semantic_update(observation: dict[str, Any]) -> dict[str, Any]:
    """Return a compact observation copy for the asynchronous semantic updater."""

    data = dict(observation) if isinstance(observation, dict) else {}
    pose = data.get("pose")
    if isinstance(pose, dict):
        data["pose"] = clean_pose_for_model(pose)
    return data


def _result_artifact_refs(value: Any) -> list[str]:
    """Collect explicit artifact references from a tool's structured result."""

    refs: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            artifact_ref = item.get("artifact_ref")
            if isinstance(artifact_ref, str) and artifact_ref and artifact_ref not in refs:
                refs.append(artifact_ref)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return refs


def _redact_model_input_context(value: Any) -> Any:
    """Keep v2 planner context readable; exact images live in provider_media."""

    if isinstance(value, dict):
        return {
            key: (
                "[inline image omitted from planner context]"
                if key == "data"
                and isinstance(item, str)
                and item.startswith("data:image/")
                else _redact_model_input_context(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_model_input_context(item) for item in value]
    return value
