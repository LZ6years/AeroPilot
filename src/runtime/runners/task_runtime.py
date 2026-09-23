"""Run semantic tasks through short-horizon planning windows."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections.abc import Callable
from typing import Any
from src.agent.mission_state import MissionState
from src.agent.planning.tool_contract_validator import NativeTurnValidationError, read_call_signature, validate_native_turn
from src.agent.planning.model_action_planner import AgentProtocolError, ModelActionPlanner
from src.agent.planning.schemas import PlannedStep
from src.agent.safety import SafetySupervisor
from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn, ToolResult
from src.env_client import ActionResult, EnvClient, SceneInstance
from src.llm.audit import redact_provider_value
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
from src.perception.raw_observation import RawObservation
from src.plugins import PluginRegistry
from src.plugins.skills.knowledge_service import KnowledgeService
from src.plugins.contracts.skill import SKILL_ARTIFACT_REFS_KEY, SKILL_MODEL_INPUTS_KEY
from src.runtime.control.executor import execute_builtin_control
from src.runtime.control.stabilization import UavStabilizer
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.model_config import ModelConfig
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.configs.task_execution_options import InitialPose, TaskExecutionOptions
from src.runtime.middleware import ToolTelemetryMiddleware
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import default_registry
from src.uav import clean_pose_for_model, pose_measurements

PHYSICAL_TRACE_SCHEMA_VERSION = "physical_window_v2"

class TaskRuntime:
    SHUTDOWN_POLICIES = {"hover", "land", "none"}
    SCENE_POLICIES = {"isolated", "reuse_running"}

    def __init__(
        self, env_client: EnvClient, recorder, planner=None,
        safety: SafetySupervisor | None = None, shutdown_policy: str = "hover",
        action_config: UavActionConfig | None = None,
        stabilization_config: UavStabilizationConfig | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        runtime_loop_guard_windows: int = 300, enable_action_planner_debug: bool = False,
        context_builder: ObservationContextBuilder | None = None,
        model_config: ModelConfig | None = None, plugin_registry: PluginRegistry | None = None,
    ) -> None:
        if shutdown_policy not in self.SHUTDOWN_POLICIES:
            raise ValueError(f"Unsupported shutdown policy: {shutdown_policy}")
        if isinstance(runtime_loop_guard_windows, bool) or not isinstance(runtime_loop_guard_windows, int) or runtime_loop_guard_windows < 1:
            raise ValueError("runtime_loop_guard_windows must be a positive integer")
        if planner is None and model_config is None:
            raise ValueError("TaskRuntime requires planner or explicit model_config")
        self.env_client, self.recorder = env_client, recorder
        self.model_config = model_config or ModelConfig()
        self.planner = planner or ModelActionPlanner(self.model_config)
        self.safety = safety or SafetySupervisor()
        self.shutdown_policy = shutdown_policy
        self.runtime_loop_guard_windows = runtime_loop_guard_windows
        self.enable_action_planner_debug = enable_action_planner_debug
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()
        self.tools = ToolDispatcher(env_client, registry=default_registry(self.plugin_registry, self.safety.safety_config),
                                    plugin_registry=self.plugin_registry)
        self.stabilizer = UavStabilizer(self.tools, recorder, stabilization_config)
        self.event_sink = event_sink
        self.context_builder = context_builder or ObservationContextBuilder(
            safety_config=self.safety.safety_config,
            plugin_registry=self.plugin_registry)
        if isinstance(self.planner, ModelActionPlanner):
            from src.agent.history_projection import HistoryProjector
            registry = getattr(getattr(self.context_builder, "composer", None), "registry", None)
            if registry is None or not callable(getattr(registry, "history_bindings", None)):
                raise ValueError("Model context builder must expose history bindings")
            self.planner.history_projector = HistoryProjector(registry.history_bindings())
        self.tool_telemetry = ToolTelemetryMiddleware(self.tools, self._debug_event)
        self.memory_writer = MemoryEventWriter(recorder.run_dir)
        self.memory_reader = MemoryRecordReader(recorder.run_dir)
        self.agent_transcript = AgentTranscript(self.memory_writer)
        self._preflight_scene_ids = set()
        self._active_reused_scenes = {}
        self._active_flight_memories = {}
        self._state_flight_memories = {}
        self._current_turn_agent_messages = []
        self._decision_index = 0
        self._persisted_provider_media = set()

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
            max_nonphysical_rounds=self.model_config.max_tool_rounds)
        self._decision_index = 0
        self._current_turn_agent_messages = []
        self._persisted_provider_media = set()
        self.tools.knowledge_service = KnowledgeService(self.plugin_registry.skill_catalog)
        scene = self._resolve_scene(map_id, scene_policy, scene_id)
        primary_error = None
        instruction_ready = False
        try:
            state.connected = scene.scene_instance_id in self._preflight_scene_ids
            state.airborne = bool(state.connected and scene.capabilities and scene.capabilities.requires_takeoff)
            memory = self._resolve_flight_memory(map_id, scene, agent_id, scene_policy, instruction, state)
            self._state_flight_memories[state.run_id] = memory
            self.agent_transcript.start_instruction(instruction, memory.active_instruction_id)
            self._event("instruction_started", {
                "instruction": instruction, "instruction_id": memory.active_instruction_id,
                "protocol_context": protocol_context,
                "loop_schema": "unified_tool_loop_v1", "physical_trace_schema": PHYSICAL_TRACE_SCHEMA_VERSION})
            instruction_ready = True
            self._validate_execution_options(scene, options)
            self._prepare_scene_for_task(scene, agent_id, state, options)
            self._execute_instruction_loop(scene, agent_id, state)
            # Terminal evidence precedes any optional land/scene cleanup. A
            # refresh is not a formal window and never changes motion accounting.
            self._finalize_instruction(scene, agent_id, state)
            if stop_scene_on_finish:
                self._run_shutdown(scene, agent_id, state)
            self._event("instruction_completed", self._state_summary(state))
            return state
        except Exception as exc:
            primary_error = exc
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
                    self._finalize_instruction(scene, agent_id, state)
                except Exception as final_error:
                    _attach_cleanup_error(exc, final_error, "final_observation")
            raise
        finally:
            if stop_scene_on_finish:
                try:
                    self.env_client.stop_scene(scene.scene_instance_id)
                    self._preflight_scene_ids.discard(scene.scene_instance_id)
                    self._forget_scene(scene)
                    self._debug_event("scene_stopped", scene.to_dict())
                except Exception as cleanup_error:
                    try:
                        self._event("scene_cleanup_failed", {"error_type": type(cleanup_error).__name__, "error": str(cleanup_error)})
                    except Exception as audit_error:
                        _attach_cleanup_error(cleanup_error, audit_error, "cleanup_audit")
                    if primary_error is None:
                        raise
                    _attach_cleanup_error(primary_error, cleanup_error, "scene_stop")

    def _finalize_instruction(self, scene, agent_id, state) -> None:
        """Close the instruction with real final evidence; preserve cleanup errors."""
        failures = []
        if state.connected:
            if scene.capabilities is None or scene.capabilities.supports("hover"):
                try:
                    result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
                    self._record_action(state, "uav.hover", {}, result.to_dict())
                    if not result.ok:
                        raise RuntimeError(result.message)
                except Exception as exc:
                    failures.append(exc)
            try:
                self._record_observation(scene, agent_id, state, list(state.observation_modalities), "task_terminal")
                context = self.context_builder.build(state, self._memory_for_state(state))
                self.agent_transcript.append_state(context, state.latest_observation.get("observation_id"))
            except Exception as exc:
                failures.append(exc)
        terminal = {**self._state_summary(state),
                    "cleanup_errors": [{"error_type": type(e).__name__, "error": str(e)} for e in failures]}
        self.agent_transcript.append_task_terminal(terminal)
        self._event("task_terminal", terminal)
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
        scene = self._active_reused_scenes.get(self._scene_key(map_id, scene_id))
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
                self.env_client.stop_scene(scene.scene_instance_id)
                self._preflight_scene_ids.discard(scene.scene_instance_id)
                self._forget_scene(scene)
                self._debug_event("scene_stopped", scene.to_dict())
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                _attach_cleanup_error(primary_error, cleanup_error, "scene_stop")

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
            self._record_acquired_scene(
                scene,
                reused=known_scene is not None and known_scene.scene_instance_id == scene.scene_instance_id,
            )
            return scene

        scene = self.env_client.start_scene(map_id, scene_id)
        self._record_acquired_scene(scene, reused=False)
        return scene

    def _record_acquired_scene(self, scene: SceneInstance, *, reused: bool) -> None:
        """Rollback acquisition if audit persistence fails before task startup.

        An already reused scene stays owned by its existing caller. Preserve
        the original persistence error even when rollback also fails.
        """
        try:
            self._record_scene_contract(scene)
            self._debug_event("scene_reused" if reused else "scene_started", scene.to_dict())
        except Exception as original:
            if not reused:
                try:
                    self.env_client.stop_scene(scene.scene_instance_id)
                except Exception as cleanup_error:
                    _attach_cleanup_error(original, cleanup_error, "scene_acquisition_rollback")
                else:
                    self._preflight_scene_ids.discard(scene.scene_instance_id)
                    self._forget_scene(scene)
            raise

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

    def _execute_physical_batch(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: Any,
        *,
        steps: list[PlannedStep],
        interventions: list[dict[str, Any]],
        native_meta: dict[str, Any],
        observation_before: dict[str, Any],
        decision_id: str,
        physical_window_index: int,
        window_index: int,
    ) -> tuple[list[dict[str, Any]], int | None, dict[str, Any], dict[str, Any]]:
        """Execute one prevalidated Safety-reviewed batch, without planning.

        The caller owns counters, terminal decisions and the window event.
        This method owns action acknowledgements, failed/skipped ToolResults,
        stabilization and the single after-observation. Shared telemetry receives
        a null phase label solely for the versioned physical trace envelope.
        """
        if not 1 <= len(steps) <= 3:
            raise ValueError("physical batch requires 1–3 steps")
        submitted_calls = native_meta.get("physical_calls", [])
        if submitted_calls and len(submitted_calls) != len(steps):
            raise ValueError("physical calls and Safety-reviewed steps must match")
        # Preflight every reviewed step before the first backend side effect;
        # Safety may replace a turn with hover, so validate replacements too.
        for step in steps:
            if step.kind != "tool" or not step.tool:
                raise ValueError("physical batches accept only tool steps")
            if self.tools.registry.get(step.tool).runtime.effect != "physical":
                raise ValueError("non-physical step in a physical batch")
            self._validate_step_capability(scene, step.tool)
        self._mark_before_last_action_window(state, observation_before)
        executed_steps = []
        native_physical_calls = list(native_meta.get("physical_calls", []))
        failed_native_index: int | None = None
        batch_error: Exception | None = None
        for index, step in enumerate(steps):
            payload = {"step_index": index, "step": step.to_dict()}
            if step.kind != "tool" or not step.tool:
                raise ValueError("runtime action plan requires tool steps; observation is automatic")
            native_call = native_physical_calls[index] if index < len(native_physical_calls) else None
            requested_tool = native_call.tool_name if native_call is not None else step.tool
            relevant_interventions = [
                dict(item)
                for item in interventions
                if item.get("tool") == requested_tool
                and ("step_index" not in item or item["step_index"] == index)
            ]
            try:
                result = self.tool_telemetry.call_step(
                    scene_instance_id=scene.scene_instance_id,
                    agent_id=agent_id,
                    phase=None,
                    window_index=window_index,
                    step_index=index,
                    tool=step.tool,
                    args=step.args,
                    pose_before=self._pose_from_observation(state.latest_observation),
                    runtime_context={
                        "latest_observation": dict(state.latest_observation),
                        "observation_before": dict(observation_before),
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
            action_ref = write_result.get("action_ref") or {}
            pose_ref = write_result.get("pose_ref") or {}
            step_tracking = {
                "step_index": index,
                "tool_call_id": (
                    native_call.tool_call_id if native_call is not None else None
                ),
                "requested_tool": requested_tool,
                "tool": step.tool,
                "effect": self.tools.registry.get(step.tool).runtime.effect,
                "status": "ok" if result.ok else "error",
                "action_ref": action_ref.get("ref"),
                "result_pose_ref": pose_ref.get("ref"),
                "requested_args": dict(
                    native_call.args if native_call is not None else step.args
                ),
                "executed_args": dict(step.args),
                "safety_interventions": relevant_interventions,
            }
            if native_call is not None:
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
                        "tracking": step_tracking,
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
            try:
                stabilization = self.stabilizer.wait_after_step(
                    scene.scene_instance_id,
                    agent_id,
                    None,
                    step,
                    capabilities=scene.capabilities,
                )
            except Exception as exc:
                # The backend already acknowledged this action. Preserve that
                # fact rather than rewriting its ToolResult as failed. Stop the
                # batch, close unexecuted calls, and propagate the infrastructure
                # error after capturing the best available final observation.
                batch_error = exc
                failed_native_index = index
                stabilization = {"status": "error", "error_type": type(exc).__name__,
                                 "error": str(exc)}
            executed_steps.append(
                {
                    **payload,
                    "tracking": step_tracking,
                    "result": data,
                    "stabilization": stabilization,
                }
            )
            self._event(
                "tool_step_completed",
                {
                    "decision_id": decision_id,
                    "physical_window_index": physical_window_index,
                    "step_index": index,
                    "tool": step.tool,
                    "ok": data.get("ok"),
                    "stabilization": self._stabilization_summary(stabilization),
                },
            )
            self._write_state(state)
            if batch_error is not None:
                break
        if failed_native_index is not None:
            for skipped_index, call in enumerate(
                native_physical_calls[failed_native_index + 1 :],
                start=failed_native_index + 1,
            ):
                transcript_result = ToolResult(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        status="skipped",
                        message="skipped because an earlier physical call or stabilization failed",
                        requested_args=dict(call.args),
                    )
                self.agent_transcript.append_tool_result(transcript_result)
                self._append_native_round_tool_result(native_meta, transcript_result)
                skipped_step = (
                    steps[skipped_index]
                    if skipped_index < len(steps)
                    else None
                )
                if skipped_step is not None:
                    executed_steps.append(
                        {
                            "step_index": skipped_index,
                            "step": skipped_step.to_dict(),
                            "tracking": {
                                "step_index": skipped_index,
                                "tool_call_id": call.tool_call_id,
                                "requested_tool": call.tool_name,
                                "tool": skipped_step.tool,
                                "effect": self.tools.registry.get(
                                    skipped_step.tool
                                ).runtime.effect,
                                "status": "skipped",
                                "action_ref": None,
                                "result_pose_ref": None,
                                "requested_args": dict(call.args),
                                "executed_args": None,
                                "safety_interventions": [],
                            },
                            "result": {
                                "ok": False,
                                "status": "skipped",
                                "message": "skipped because an earlier physical call or stabilization failed",
                            },
                            "stabilization": {
                                "status": "skipped",
                                "reason": "physical call was not executed",
                            },
                        }
                    )
            self._try_hover(scene, agent_id, state)
        observation_after: dict[str, Any] = {}
        observation_error: Exception | None = None
        try:
            observation_after = self._apply_automatic_observation_policy(scene, agent_id, state)
        except Exception as exc:
            observation_error = exc
            if batch_error is None:
                batch_error = exc
                self._try_hover(scene, agent_id, state)
        if batch_error is not None:
            # The normal caller cannot publish a window after this method raises.
            # Publish the actual partial evidence here; a missing after frame is
            # explicitly unavailable, never replaced by the before observation.
            self._event("physical_window_recorded", {
                "schema_version": PHYSICAL_TRACE_SCHEMA_VERSION,
                "decision_id": decision_id, "phase_key": None,
                "window_index": physical_window_index,
                "observation_before_id": observation_before.get("observation_id"),
                "observation_after_id": observation_after.get("observation_id"),
                "status": "error",
                "ordered_steps": [item["tracking"] for item in executed_steps],
                "error_type": type(batch_error).__name__, "error": str(batch_error),
                "stabilization": [item["stabilization"] for item in executed_steps],
                "observation_error": None if observation_error is None else {
                    "error_type": type(observation_error).__name__, "error": str(observation_error)},
            })
            raise batch_error
        window_measurements = pose_measurements(
            self._pose_from_observation(observation_before),
            self._pose_from_observation(observation_after),
        )
        return executed_steps, failed_native_index, observation_after, window_measurements

    def _execute_instruction_loop(
        self, scene: SceneInstance, agent_id: str, state: MissionState,
    ) -> None:
        """Run the phase-free tool protocol with caller-prepared scene/context.

        Scene acquisition and final shutdown remain outside this method. Every
        model round uses the same transcript; no parser, selector, phase commit
        or synthetic one-phase plan is invoked. The caller must bind the new
        ToolRegistry, instruction context and KnowledgeService before entry.
        """
        registry = self.tools.registry
        allowed = {
            spec.name for spec in registry.model_specs()
            if spec.runtime.effect != "lifecycle"
            and (not spec.metadata.get("backend_action") or scene.capabilities is None
                 or scene.capabilities.supports(spec.metadata["backend_action"]))
        }
        seen_call_ids: set[str] = set()
        executed_reads: set[tuple[str, str, str]] = set()
        consecutive_rejections = 0
        rejection_limit = self.model_config.max_consecutive_protocol_rejections
        while not state.completed and not state.stopped:
            context = self.context_builder.build(state, self._memory_for_state(state))
            self.agent_transcript.append_state(context, state.latest_observation.get("observation_id"))
            self._write_state(state)
            if state.physical_window_count >= state.max_physical_windows:
                state.mark_stopped("instruction_physical_window_limit", "runtime_loop_guard")
                break
            if state.nonphysical_round_count >= state.max_nonphysical_rounds:
                state.mark_stopped("nonphysical_round_limit", "runtime_loop_guard")
                break
            if consecutive_rejections >= rejection_limit:
                state.mark_stopped("consecutive_protocol_rejections", "runtime_loop_guard")
                break
            self._decision_index += 1
            decision_id = f"decision_{self._decision_index:04d}"
            try:
                turn = self.planner.plan_turn(
                    self.agent_transcript, registry, allowed_tool_names=allowed,
                    decision_index=self._decision_index,
                )
            except AgentProtocolError as exc:
                self._capture_instruction_exchange(decision_id, error_type=type(exc).__name__)
                if exc.rejected_turn:
                    self.agent_transcript.append_provider_rejection(exc.rejected_turn)
                self.agent_transcript.append_protocol_feedback(str(exc))
                state.record_nonphysical_round()
                consecutive_rejections += 1
                self._event("tool_turn_rejected", {"decision_id": decision_id, "error": str(exc)})
                continue
            except Exception as exc:
                self._capture_instruction_exchange(decision_id, error_type=type(exc).__name__)
                raise
            self._capture_instruction_exchange(decision_id, turn=turn)
            ids = [call.tool_call_id for call in turn.tool_calls]
            # Invalid identities cannot be persisted as canonical calls: that
            # would corrupt pairing even if rejected ToolResults were appended.
            if len(set(ids)) != len(ids) or set(ids) & seen_call_ids or any(not item for item in ids):
                self.agent_transcript.append_provider_rejection({
                    "error": "tool call IDs must be instruction-wide unique",
                    "content_present": bool(turn.public_intent),
                    "content": turn.public_intent or None,
                    "tool_calls": [call.to_dict() for call in turn.tool_calls],
                })
                self.agent_transcript.append_protocol_feedback("Use fresh non-empty tool_call_ids for every call.")
                state.record_nonphysical_round()
                consecutive_rejections += 1
                continue
            history_before_turn = self.agent_transcript.snapshot()
            self.agent_transcript.append_assistant(turn)
            try:
                effect = validate_native_turn(
                    turn, registry, allowed_names=allowed,
                    latest_observation=state.latest_observation,
                    prior_call_ids=seen_call_ids, executed_read_signatures=executed_reads,
                )
            except NativeTurnValidationError as exc:
                seen_call_ids.update(ids)
                for record in self._rejected_tool_results(turn.tool_calls, str(exc), rejection=exc.diagnostic):
                    self.agent_transcript.append_tool_result(ToolResult(
                        record["tool_call_id"], record["tool_name"], "rejected", str(exc),
                        data=record["data"], requested_args=record["requested_args"],
                    ))
                self.agent_transcript.append_protocol_feedback(str(exc) + ". No call in this turn executed.")
                state.record_nonphysical_round()
                consecutive_rejections += 1
                continue
            seen_call_ids.update(ids)
            if effect == "runtime_control":
                state.record_nonphysical_round()
                result = execute_builtin_control(turn.tool_calls[0], state, registry, history_before_turn)
                self.agent_transcript.append_tool_result(result)
                self._event("task_control_recorded", {
                    "decision_id": decision_id, "tool_result": result.to_dict(),
                })
                if result.status == "rejected":
                    self.agent_transcript.append_protocol_feedback(result.message)
                    consecutive_rejections += 1
                else:
                    consecutive_rejections = 0
                continue
            if effect == "read_only":
                state.record_nonphysical_round()
                observation_id = str(state.latest_observation.get("observation_id") or "")
                for call in turn.tool_calls:
                    executed_reads.add(read_call_signature(observation_id, call.tool_name, call.args))
                results = self._execute_native_read_calls(scene, agent_id, state, turn.public_intent, turn.tool_calls)
                if self.tools.knowledge_service is not None:
                    state.loaded_knowledge = self.tools.knowledge_service.loaded_index()
                self._event("read_only_turn_recorded", {"decision_id": decision_id, "tool_results": results})
                if all(result["status"] == "ok" for result in results):
                    consecutive_rejections = 0
                continue
            steps = [PlannedStep.tool_step(call.tool_name, dict(call.args), turn.public_intent)
                     for call in turn.tool_calls]
            safety = self.safety.review_steps(steps, state, registry=registry)
            before = copy.deepcopy(state.latest_observation)
            state.begin_physical_window()
            executed, failed, after, measurements = self._execute_physical_batch(
                scene, agent_id, state, steps=safety.steps, interventions=safety.interventions,
                native_meta={"physical_calls": turn.tool_calls}, observation_before=before,
                decision_id=decision_id, physical_window_index=state.physical_window_count,
                window_index=state.physical_window_count,
            )
            state.finish_physical_window(after, measurements)
            self._maybe_recharge_visual_search_budget(state, measurements)
            self._event("physical_window_recorded", {
                "schema_version": PHYSICAL_TRACE_SCHEMA_VERSION,
                "decision_id": decision_id, "phase_key": None,
                "window_index": state.physical_window_count,
                "observation_before_id": before.get("observation_id"),
                "observation_after_id": after.get("observation_id"),
                "status": "error" if failed is not None else "ok",
                "ordered_steps": [item["tracking"] for item in executed],
                "measurements": measurements,
            })
            consecutive_rejections = 0
        self._write_state(state)

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
            "loop_schema": "unified_tool_loop_v1",
            "instruction_id": instruction_id,
            "decision_id": decision_id, "decision_index": self._decision_index,
            "checkpoint_observation_id": self.agent_transcript.latest_observation_id(),
            "model_context": redact_provider_value(self.agent_transcript.latest_context()),
            "model_rounds": [record],
        }
        self.recorder.write_json(debug_ref, snapshot)
        # Freezeable append-only association, not a glob over future artifacts.
        # This records a captured proposal, never proof of applied control.
        self._event("model_turn_recorded", {
            "decision_id": decision_id, "debug_ref": debug_ref,
            "instruction_id": instruction_id,
            "loop_schema": "unified_tool_loop_v1",
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

    @staticmethod
    def _rejected_tool_results(
        calls: list[AgentToolCall],
        reason: str,
        *,
        rejection: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Describe Runtime-generated rejection results for decision review."""

        diagnosis = dict(rejection or {"code": "invalid_turn_combination", "scope": "turn"})
        source = diagnosis.get("source_tool_call_id")
        return [
            ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="rejected",
                message=reason,
                data={
                    "rejection": {
                        **diagnosis,
                        "role": (
                            ("source" if call.tool_call_id == source else "affected")
                            if source else "turn_member"
                        ),
                        "executed": False,
                    }
                },
                requested_args=dict(call.args),
            ).to_dict()
            for call in calls
        ]

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
            try:
                result = self.tools.call(
                    scene.scene_instance_id, agent_id, call.tool_name, call.args,
                    runtime_context={
                        "latest_observation": dict(state.latest_observation),
                        "run_dir": str(self.recorder.run_dir),
                        "public_intent": public_intent, "tool_call_id": call.tool_call_id,
                    },
                )
            except Exception as exc:
                # A failed independent read must still close its call and must
                # not suppress the results of other prevalidated reads.
                result = ActionResult(ok=False, action=call.tool_name, message=str(exc),
                                      data={"error_type": type(exc).__name__})
            result_data = dict(result.data)
            capability_artifact_refs = _pop_string_list(
                result_data,
                SKILL_ARTIFACT_REFS_KEY,
            )
            capability_model_inputs = _pop_dict_list(
                result_data,
                SKILL_MODEL_INPUTS_KEY,
            )
            data = result.to_dict()
            data["data"] = result_data
            write_result = self._record_action(state, call.tool_name, call.args, data)
            transcript_result = ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="ok" if result.ok else "error",
                    message=result.message,
                    data=result_data,
                    artifact_refs=list(
                        dict.fromkeys(
                            [
                                *capability_artifact_refs,
                                *_result_artifact_refs(result_data),
                            ]
                        )
                    ),
                    result_ref=write_result["action_ref"].get("ref"),
                    requested_args=dict(call.args),
                    executed_args=dict(call.args),
                    model_inputs=capability_model_inputs,
                )
            self.agent_transcript.append_tool_result(transcript_result)
            recorded_results.append(transcript_result.to_dict())
        return recorded_results

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
            outcome = {"status": "physical_batch_accepted", "feedback": None}
            final_round["runtime_outcome"] = outcome
        results = outcome.setdefault("tool_results", [])
        if isinstance(results, list):
            results.append(result.to_dict())

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
                overwrite=True,
            )
        self.recorder.write_json("debug/latest_observation.json", data)
        self.recorder.write_json("contexts/latest_context.json", self.context_builder.build(state, self._memory_for_state(state)))
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

    def _event(self, event: str, payload: dict) -> None:
        self.recorder.event(event, payload)
        if event in {
            "instruction_started",
            "model_turn_recorded",
            "task_control_recorded",
            "read_only_turn_recorded",
            "physical_window_recorded",
            "task_terminal",
            "instruction_failed",
            "scene_cleanup_failed",
            "tool_step_completed",
            "observation_updated",
            "runtime_loop_guard_reached",
        }:
            self._emit_runtime_event({"kind": event, "payload": dict(payload)})

    def _debug_event(self, event: str, payload: dict) -> None:
        debug_event = getattr(self.recorder, "debug_event", None)
        if callable(debug_event):
            debug_event(event, payload)
        else:
            self.recorder.event(event, payload)

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
        write_result = self.memory_writer.write_action(tool, args, result)
        memory = self._memory_for_state(state)
        memory.append_action_ref(write_result["action_ref"])
        memory.append_pose_ref(write_result.get("pose_ref"))
        return write_result


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


def _pop_string_list(data: dict[str, Any], key: str) -> list[str]:
    """Remove an internal skill envelope field and return safe string values."""

    value = data.pop(key, [])
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _pop_dict_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Remove an internal skill envelope field and return copied objects."""

    value = data.pop(key, [])
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


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


def _attach_cleanup_error(primary: Exception, cleanup: Exception, stage: str) -> None:
    """Retain secondary cleanup evidence without replacing the original failure."""
    errors = list(getattr(primary, "cleanup_errors", []))
    errors.append({"stage": stage, "error_type": type(cleanup).__name__, "error": str(cleanup)})
    primary.cleanup_errors = errors
