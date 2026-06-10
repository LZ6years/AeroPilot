"""Run semantic tasks through short-horizon planning windows."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.messages import AgentMessage
from src.agent.phase import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK, CONTINUE_CURRENT_PHASE, PhaseTransition, evaluate_phase_completion
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.planning.model_repair_planner import ModelRepairPlannerError
from src.agent.safety import SafetySupervisor
from src.agent.safety import SafetyDecision
from src.agent.planning.schemas import ShortHorizonPlan
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.task_parser import TaskParser
from src.env_client.base import EnvClient
from src.env_client.schemas import SceneInstance
from src.memory.flight_memory import FlightMemory
from src.memory.context.modules.image import (
    frame_debug_metadata,
    rgb_model_inputs_from_frames,
    selected_frame_summaries,
    selected_rgb_frames_from_observations,
)
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.perception.raw_observation import RawObservation
from src.runtime.control import UavStabilizer
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.tools.dispatcher import ToolDispatcher
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
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        runtime_loop_guard_windows: int = 100,
        enable_action_planner_debug: bool = False,
    ) -> None:
        if shutdown_policy not in self.SHUTDOWN_POLICIES:
            raise ValueError(f"Unsupported shutdown policy: {shutdown_policy}")
        if int(runtime_loop_guard_windows) <= 0:
            raise ValueError("runtime_loop_guard_windows must be positive")
        self.env_client = env_client
        self.recorder = recorder
        self.task_parser = task_parser
        self.planner = planner or ModelActionPlanner()
        self.safety = safety or SafetySupervisor()
        self.shutdown_policy = shutdown_policy
        self.runtime_loop_guard_windows = int(runtime_loop_guard_windows)
        self.enable_action_planner_debug = bool(enable_action_planner_debug)
        self.plugin_registry = PluginRegistry.from_config()
        self.tools = ToolDispatcher(env_client, plugin_registry=self.plugin_registry)
        self.stabilizer = UavStabilizer(self.tools, recorder, stabilization_config)
        # Optional synchronous stream for CLI/TUI frontends. The recorder remains
        # the historical source of truth; this sink is only for live user display.
        self.event_sink = event_sink
        self.context_builder = ObservationContextBuilder(
            action_config=action_config,
            safety_config=self.safety.safety_config,
            plugin_registry=self.plugin_registry,
            enable_model_skill_selection=getattr(self.planner, "planner_mode", "") == "model",
        )
        self.memory_writer = MemoryEventWriter(self.recorder.run_dir)
        self.memory_reader = MemoryRecordReader(self.recorder.run_dir)
        self._preflight_scene_ids: set[str] = set()
        self._active_reused_scenes: dict[tuple[str, str | None], SceneInstance] = {}
        self._active_flight_memories: dict[tuple[str, str | None, str], FlightMemory] = {}
        self._state_flight_memories: dict[str, FlightMemory] = {}
        # In-memory user-facing messages for the current instruction. Public
        # events are still written for history, but CLI/TUI summaries should not
        # depend on the async logger flushing before they can show the final reply.
        self._current_turn_agent_messages: list[dict] = []

    def run_task(
        self,
        map_id: str,
        instruction: str,
        agent_id: str = "uav_1",
        scene_policy: str = "isolated",
        stop_scene_on_finish: bool = True,
        scene_id: str | None = None,
    ) -> MissionState:
        # Execute one short-horizon window per semantic phase.
        self._current_turn_agent_messages = []
        parse_result = self.task_parser.parse(instruction)
        return self.run_parsed_task(
            map_id,
            parse_result,
            agent_id=agent_id,
            scene_policy=scene_policy,
            stop_scene_on_finish=stop_scene_on_finish,
            scene_id=scene_id,
        )

    def run_parsed_task(
        self,
        map_id: str,
        parse_result: TaskParseResult,
        agent_id: str = "uav_1",
        scene_policy: str = "isolated",
        stop_scene_on_finish: bool = True,
        scene_id: str | None = None,
    ) -> MissionState:
        # Execute a task from an already-created semantic plan.
        if scene_policy not in self.SCENE_POLICIES:
            raise ValueError(f"Unsupported scene policy: {scene_policy}")
        self._current_turn_agent_messages = []
        self.recorder.write_json("state/semantic_plan.json", parse_result.to_dict())
        self._debug_event("task_parsed", parse_result.to_dict())
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
            airborne=scene.scene_instance_id in self._preflight_scene_ids,
            max_phase_window_count=self.runtime_loop_guard_windows,
        )
        flight_memory = self._resolve_flight_memory(map_id, scene, agent_id, scene_policy, parse_result, state)
        self._state_flight_memories[state.run_id] = flight_memory
        try:
            self._prepare_scene_for_task(scene, agent_id, state)
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
            self.recorder.write_json("state/scene_instance.json", scene.to_dict())
            if known_scene is not None and known_scene.scene_instance_id == scene.scene_instance_id:
                self._debug_event("scene_reused", scene.to_dict())
            else:
                self._debug_event("scene_started", scene.to_dict())
            return scene

        scene = self.env_client.start_scene(map_id, scene_id)
        self.recorder.write_json("state/scene_instance.json", scene.to_dict())
        self._debug_event("scene_started", scene.to_dict())
        return scene

    def _prepare_scene_for_task(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        if scene.scene_instance_id not in self._preflight_scene_ids:
            self._run_preflight(scene, agent_id, state)
            self._preflight_scene_ids.add(scene.scene_instance_id)
            return

        observation_data = self._record_observation(
            scene,
            agent_id,
            state,
            ["rgb", "depth", "pose"],
            source="instruction_start",
        )
        self._debug_event("instruction_start_observation_updated", observation_data)
        self._event("observation_updated", self._observation_summary_event(observation_data, source="instruction_start"))
        self._write_state(state)

    def _execute_phase_loop(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        while not state.completed and not state.stopped:
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

        self._debug_event("preflight_takeoff_started", {"scene_instance_id": scene.scene_instance_id})
        takeoff_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.takeoff", {})
        takeoff_data = takeoff_result.to_dict()
        self._record_action(state, "uav.takeoff", {}, takeoff_data)
        self._debug_event("preflight_takeoff_completed", takeoff_data)
        if not takeoff_result.ok:
            raise RuntimeError(f"Preflight takeoff failed: {takeoff_result.message}")

        hover_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
        hover_data = hover_result.to_dict()
        self._record_action(state, "uav.hover", {}, hover_data)
        self._debug_event("preflight_hover_completed", hover_data)
        if not hover_result.ok:
            raise RuntimeError(f"Preflight hover failed: {hover_result.message}")

        stabilization = self.stabilizer.wait_after_preflight(scene.scene_instance_id, agent_id)
        self._debug_event(
            "preflight_stabilization_completed",
            stabilization,
        )
        self._event("preflight_completed", {"scene_instance_id": scene.scene_instance_id, "stabilization": self._stabilization_summary(stabilization)})
        observation_data = self._record_observation(
            scene,
            agent_id,
            state,
            ["rgb", "depth", "pose"],
            source="preflight",
        )
        self._debug_event("initial_observation_updated", observation_data)
        self._event("observation_updated", self._observation_summary_event(observation_data, source="preflight"))
        self._write_state(state)

    def _run_phase_window(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
    ) -> tuple[ShortHorizonPlan, SafetyDecision, PhaseTransition]:
        phase_name = state.current_phase_name
        phase_key = state.current_phase_key
        observation_before = dict(state.latest_observation)
        state.ensure_current_phase_progress(observation_before)
        state.update_current_phase_progress(observation_before)
        context = self.context_builder.build(state, self._memory_for_state(state))
        self.recorder.write_json("contexts/latest_context.json", context)
        try:
            raw_plan = self.planner.plan_next(state, context)
        except ModelRepairPlannerError as exc:
            self._record_model_action_contract_repair_events(exc.repair_attempts)
            raise
        self._record_model_action_contract_repair_events(getattr(self.planner, "last_repair_attempts", []))
        if getattr(self.planner, "planner_mode", "") == "model":
            self._debug_event("model_action_planned", raw_plan.to_dict())
        safety_decision = self.safety.review(raw_plan, state)
        plan = safety_decision.plan
        planner_debug_ref = self._write_action_planner_debug_artifact(
            state,
            context,
            raw_plan,
            plan,
            safety_decision,
        )
        self._debug_event("short_horizon_planned", plan.to_dict())
        self._agent_messages(plan.agent_messages, source="action_planner", phase=plan.phase)
        self._event(
            "phase_plan_ready",
            {
                "phase": plan.phase,
                "action_window": plan.action_window.to_dict(),
                "step_count": len(plan.steps),
                "tools": [step.tool for step in plan.steps if step.kind == "tool"],
                "planner_debug_ref": planner_debug_ref,
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
        executed_steps = []
        for index, step in enumerate(plan.steps):
            payload = {"phase": plan.phase, "step_index": index, "step": step.to_dict()}
            self._debug_event("step_started", payload)
            if step.kind != "tool" or not step.tool:
                raise ValueError("runtime action plan requires tool steps; observation is automatic")
            result = self.tools.call(
                scene.scene_instance_id,
                agent_id,
                step.tool,
                step.args,
                runtime_context={
                    "latest_observation": dict(state.latest_observation),
                    "observation_before": dict(observation_before),
                    "phase": plan.phase,
                    "run_dir": str(self.recorder.run_dir),
                },
            )
            data = result.to_dict()
            self._record_action(state, step.tool, step.args, data)
            self._debug_event("step_completed", {**payload, "result": data})
            if not result.ok:
                raise RuntimeError(f"Action failed: {step.tool}: {result.message}")
            stabilization = self.stabilizer.wait_after_step(scene.scene_instance_id, agent_id, plan.phase, step)
            executed_steps.append({**payload, "result": data, "stabilization": stabilization})
            self._event(
                "tool_step_completed",
                {
                    "phase": plan.phase,
                    "step_index": index,
                    "tool": step.tool,
                    "ok": data.get("ok"),
                    "stabilization": self._stabilization_summary(stabilization),
                },
            )
            self._write_state(state)
        observation_after = self._apply_automatic_observation_policy(scene, agent_id, state, plan.phase)
        window_measurements = pose_measurements(
            self._pose_from_observation(observation_before),
            self._pose_from_observation(observation_after),
        )
        window_record = {
            "phase_key": phase_key,
            "phase": phase_name,
            "window_index": state.current_phase_window_count,
            "action_window": plan.action_window.to_dict(),
            "executed_steps": list(executed_steps),
            "observation_before": observation_before,
            "observation_after": observation_after,
            "measurements": window_measurements,
            "safety_interventions": list(safety_decision.interventions),
        }
        window_ref = self._write_phase_window_debug_artifact(phase_key, state.current_phase_window_count, window_record)
        window_summary = {
            "phase": phase_name,
            "phase_key": phase_key,
            "window_index": state.current_phase_window_count,
            "action_window": plan.action_window.to_dict(),
            "executed_step_count": len(executed_steps),
            "executed_tools": [
                executed.get("step", {}).get("tool")
                for executed in executed_steps
                if isinstance(executed.get("step", {}), dict)
            ],
            "measurements": window_measurements,
            "safety_intervention_count": len(safety_decision.interventions),
            "debug_ref": window_ref,
        }
        self._debug_event("phase_window_recorded", window_summary)
        self._event("phase_window_recorded", window_summary)
        state.update_current_phase_progress(
            observation_after,
            last_window_measurements=window_measurements,
            tool_facts_delta=self._tool_facts_from_executed_steps(executed_steps),
        )
        transition = self._judge_phase_transition_after_execution(
            state,
            context,
            plan,
            safety_decision,
            observation_before,
            observation_after,
            executed_steps,
        )
        return plan, safety_decision, transition

    def _apply_phase_transition(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        plan: ShortHorizonPlan,
        safety_decision: SafetyDecision,
        transition: PhaseTransition,
    ) -> None:
        criteria = self._current_phase_completion_criteria(state)
        payload = {
            "phase": plan.phase,
            "decision": transition.decision,
            "reason": transition.reason,
            "current_phase_window_count": state.current_phase_window_count,
            "max_phase_window_count": state.max_phase_window_count,
            "phase_completion_evaluation": evaluate_phase_completion(
                criteria,
                state.current_phase_execution_summary,
            ),
        }
        self._agent_messages(transition.agent_messages, source="phase_judge", phase=plan.phase)
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
            self._event("phase_transition_applied", payload)
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
        state.mark_stopped("visual_search_budget_exhausted", source="safety_yaw_scan_budget")

    def _judge_phase_transition_after_execution(
        self,
        state: MissionState,
        planner_context: dict,
        plan: ShortHorizonPlan,
        safety_decision: SafetyDecision,
        observation_before: dict,
        observation_after: dict,
        executed_steps: list[dict],
    ) -> PhaseTransition:
        judge = getattr(self.planner, "judge_phase_transition", None)
        if not callable(judge):
            if safety_decision.interventions:
                return PhaseTransition.continue_current_phase("SafetySupervisor changed the action; post-action judgment is required")
            return plan.phase_transition
        phase_context = self._build_post_action_phase_context(
            state,
            planner_context,
            plan,
            safety_decision,
            observation_before,
            observation_after,
            executed_steps,
        )
        self.recorder.write_json("contexts/latest_phase_transition_context.json", phase_context)
        context_ref = self._write_phase_transition_debug_artifact(
            state.current_phase_key,
            state.current_phase_window_count,
            phase_context,
        )
        context_summary = {
            "phase": plan.phase,
            "phase_key": state.current_phase_key,
            "window_index": state.current_phase_window_count,
            "context_ref": context_ref,
            "action_window": phase_context.get("action_window", {}),
            "phase_completion_criteria": phase_context.get("phase_completion_criteria", []),
            "phase_execution_summary": phase_context.get("phase_execution_summary", {}),
        }
        self._debug_event("phase_transition_judge_context_built", context_summary)
        self._event("phase_transition_judge_context_built", context_summary)
        transition = judge(phase_context)
        self._event("model_phase_transition_judged", transition.to_dict())
        self._debug_event("model_phase_transition_judged", transition.to_dict())
        return transition

    def _build_post_action_phase_context(
        self,
        state: MissionState,
        planner_context: dict,
        plan: ShortHorizonPlan,
        safety_decision: SafetyDecision,
        observation_before: dict,
        observation_after: dict,
        executed_steps: list[dict],
    ) -> dict:
        before_pose = self._pose_from_observation(observation_before)
        after_pose = self._pose_from_observation(observation_after)
        clean_observation_before = self._clean_observation_for_model(observation_before)
        clean_observation_after = self._clean_observation_for_model(observation_after)
        memory = self._memory_for_state(state)
        phase_judge_fragments = self._post_action_phase_fragments(clean_observation_after, memory)
        phase_judge_model_input_count = sum(
            len(fragment.get("model_inputs", []))
            for fragment in phase_judge_fragments
            if isinstance(fragment, dict)
        )
        criteria = self._current_phase_completion_criteria(state)
        completion_evaluation = evaluate_phase_completion(
            criteria,
            state.current_phase_execution_summary,
        )
        return {
            "active_instruction": state.user_instruction,
            "semantic_goal": state.semantic_plan.goal.to_dict(),
            "current_phase": planner_context.get("current_phase"),
            "phase_completion_criteria": criteria,
            "phase_completion_evaluation": completion_evaluation,
            "phase_window_count": state.current_phase_window_count,
            "max_phase_window_count": state.max_phase_window_count,
            "action_window": plan.action_window.to_dict(),
            "planner_context_summary": {
                "fragment_names": [fragment.get("name") for fragment in planner_context.get("fragments", [])],
                "phase_judge_model_input_count": phase_judge_model_input_count,
                "safety_constraints": planner_context.get("safety_constraints", {}),
            },
            "fragments": phase_judge_fragments,
            "user_preferences": planner_context.get("user_preferences", {}),
            "safety_intervention_summary": self._safety_intervention_summary(safety_decision, state, executed_steps),
            "yaw_scan_remaining_deg": self._yaw_scan_remaining_deg(state),
            "visual_phase_budget_exhausted": self._visual_phase_budget_exhausted(
                {"phase_completion_evaluation": completion_evaluation},
                safety_decision,
            ),
            "recent_executed_tools": self._executed_tool_names(executed_steps)[-5:],
            "recent_safety_reasons": self._safety_intervention_reasons(safety_decision)[-5:],
            "phase_execution_summary": state.current_phase_execution_summary,
            "action_plan": plan.to_dict(),
            "executed_steps": list(executed_steps),
            "safety_interventions": list(safety_decision.interventions),
            "observation_before": clean_observation_before,
            "observation_after": clean_observation_after,
            "measurements": pose_measurements(before_pose, after_pose),
            "recent_history": {
                "recent_action_refs": list(memory.recent_action_refs[-5:]),
                "recent_observation_refs": list(memory.recent_observation_refs[-3:]),
                "note": "Full history is stored in memory/*.jsonl; MissionState keeps only latest state.",
            },
        }

    def _current_phase_completion_criteria(self, state: MissionState) -> list[dict[str, Any]]:
        if state.current_phase_index >= len(state.semantic_plan.phases):
            return []
        criteria = state.semantic_plan.phases[state.current_phase_index].completion_criteria
        if isinstance(criteria, list):
            return [dict(item) for item in criteria if isinstance(item, dict)]
        if isinstance(criteria, dict):
            return [dict(criteria)]
        return []

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
        evaluation = payload.get("phase_completion_evaluation", {})
        if not isinstance(evaluation, dict) or int(evaluation.get("model_judged_count", 0) or 0) <= 0:
            return False
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
            "yaw_scan_accum_deg": float(getattr(state, "yaw_scan_accum_deg", 0.0) or 0.0),
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
        max_yaw_scan = float(self.safety.safety_config.max_yaw_scan_accum_deg)
        yaw_scan_accum = float(getattr(state, "yaw_scan_accum_deg", 0.0) or 0.0)
        return max(0.0, max_yaw_scan - yaw_scan_accum)

    def _executed_tool_names(self, executed_steps: list[dict]) -> list[str]:
        names = []
        for executed in executed_steps:
            step = executed.get("step", {}) if isinstance(executed, dict) else {}
            if isinstance(step, dict) and step.get("tool"):
                names.append(str(step["tool"]))
        return names

    def _post_action_phase_fragments(
        self,
        observation_after: dict,
        memory: FlightMemory | None = None,
    ) -> list[dict[str, Any]]:
        frames = observation_after.get("frames") if isinstance(observation_after, dict) else {}
        rgb_frame = frames.get("rgb") if isinstance(frames, dict) else None
        latest_rgb = dict(rgb_frame) if isinstance(rgb_frame, dict) else None
        selected_rgb_frames = []
        if memory and memory.memory_reader:
            observations = memory.memory_reader.read_refs(memory.recent_observation_refs)
            selected_rgb_frames = selected_rgb_frames_from_observations(memory.memory_reader, observations)
        if not selected_rgb_frames and latest_rgb:
            latest_rgb["selection_role"] = "latest_observation"
            selected_rgb_frames = [latest_rgb]
        model_inputs = rgb_model_inputs_from_frames(selected_rgb_frames)
        has_rgb = latest_rgb is not None
        has_mock_frames = any(frame.get("mock") is True for frame in selected_rgb_frames) or bool(
            latest_rgb and latest_rgb.get("mock") is True
        )
        selected_frames = selected_frame_summaries(selected_rgb_frames)
        return [
            {
                "name": "image",
                "enabled": True,
                "source": "PostActionObservationImageContext",
                "content": {
                    "selected_frames": selected_frames,
                    "visual_evidence": {
                        "has_rgb": has_rgb,
                        "has_depth": False,
                        "has_mock_frames": has_mock_frames,
                        "latest_observation_source": observation_after.get("source"),
                        "attached_image_count": len(model_inputs),
                        "interpretation_status": (
                            "image_attached_for_direct_judge_inspection"
                            if model_inputs
                            else "metadata_only_not_interpreted"
                        ),
                    },
                },
                "model_inputs": model_inputs,
                "facts": [],
                "risks": [
                    {
                        "type": "visual_evidence_is_mock",
                        "value": "rgb",
                        "source": "PostActionObservationImageContext",
                    }
                ]
                if has_mock_frames
                else [],
                "artifact_refs": [
                    frame.get("artifact_ref") or frame.get("path")
                    for frame in selected_rgb_frames
                    if frame.get("artifact_ref") or frame.get("path")
                ],
                "confidence": "medium" if has_rgb else "low",
                "warnings": ["rgb_frame_is_mock"] if has_mock_frames else [],
                "metadata": {
                    "observation_source": observation_after.get("source"),
                    "selected_frame_metadata": frame_debug_metadata(selected_rgb_frames),
                    "recent_rgb_frame_metadata": frame_debug_metadata([latest_rgb] if latest_rgb else []),
                },
            }
        ]

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

    def _clean_observation_for_model(self, observation: dict) -> dict:
        data = dict(observation) if isinstance(observation, dict) else {}
        pose = data.get("pose")
        if isinstance(pose, dict):
            data["pose"] = clean_pose_for_model(pose)
        return data

    def _apply_automatic_observation_policy(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        phase: str,
    ) -> dict:
        modalities = ["rgb", "depth", "pose"]
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

    def _record_model_action_contract_repair_events(self, attempts: list[dict]) -> None:
        for attempt in attempts:
            requested_payload = {
                "attempt": attempt.get("attempt"),
                "repair_context": attempt.get("repair_context", {}),
            }
            self._debug_event("model_action_contract_repair_requested", requested_payload)
            status = attempt.get("status")
            if status == "repaired":
                self._debug_event("model_action_contract_repaired", attempt)
            elif status == "failed":
                self._debug_event("model_action_contract_repair_failed", attempt)

    def _record_observation(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        modalities: list[str],
        source: str,
    ) -> dict:
        observation = self.tools.observe(scene.scene_instance_id, agent_id, modalities)
        raw = RawObservation.from_env_observation(observation, source=source)
        data = raw.to_dict()
        state.record_observation(data)
        write_result = self.memory_writer.write_observation(data)
        memory = self._memory_for_state(state)
        memory.append_observation_ref(write_result["observation_ref"])
        memory.append_pose_ref(write_result.get("pose_ref"))
        memory.append_frame_refs(write_result.get("frame_refs"))
        self.recorder.write_json("debug/latest_observation.json", data)
        self.recorder.write_json("contexts/latest_context.json", self.context_builder.build(state, self._memory_for_state(state)))
        return data

    def _try_hover(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
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
        hover_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
        hover_data = hover_result.to_dict()
        self._record_action(state, "uav.hover", {}, hover_data)
        self._debug_event("shutdown_hover_completed", hover_data)
        if not hover_result.ok:
            raise RuntimeError(f"Shutdown hover failed: {hover_result.message}")

        if self.shutdown_policy == "land":
            self._debug_event("shutdown_land_started", {"scene_instance_id": scene.scene_instance_id})
            land_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.land", {})
            land_data = land_result.to_dict()
            self._record_action(state, "uav.land", {}, land_data)
            self._debug_event("shutdown_land_completed", land_data)
            if not land_result.ok:
                raise RuntimeError(f"Shutdown land failed: {land_result.message}")

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
            "phase_plan_ready",
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
        raw_plan: ShortHorizonPlan,
        safe_plan: ShortHorizonPlan,
        safety_decision: SafetyDecision,
    ) -> str | None:
        if not self.enable_action_planner_debug:
            return None
        name = f"debug/action_planner/{self._safe_file_part(state.current_phase_key)}_window_{state.current_phase_window_count:03d}.json"
        self.recorder.write_json(
            name,
            {
                "phase": state.current_phase_name,
                "phase_key": state.current_phase_key,
                "window_index": state.current_phase_window_count,
                "planner_context": _redact_model_input_data(planner_context),
                "raw_plan": raw_plan.to_dict(),
                "safe_plan": safe_plan.to_dict(),
                "safety_decision": {
                    "interventions": list(safety_decision.interventions),
                    "plan_changed": bool(safety_decision.interventions),
                },
            },
        )
        return name

    def _write_phase_transition_debug_artifact(self, phase_key: str, window_index: int, payload: dict) -> str:
        name = f"debug/phase_transition/{self._safe_file_part(phase_key)}_window_{window_index:03d}.json"
        self.recorder.write_json(name, payload)
        return name

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

    def _record_action(self, state: MissionState, tool: str, args: dict, result: dict) -> None:
        state.record_action(tool, args, result)
        write_result = self.memory_writer.write_action(tool, args, result)
        memory = self._memory_for_state(state)
        memory.append_action_ref(write_result["action_ref"])
        memory.append_pose_ref(write_result.get("pose_ref"))


def _redact_model_input_data(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key == "data" and isinstance(item, str) and item.startswith("data:image/"):
                redacted[key] = f"<omitted encoded image data: {len(item)} chars>"
            else:
                redacted[key] = _redact_model_input_data(item)
        return redacted
    if isinstance(value, list):
        return [_redact_model_input_data(item) for item in value]
    return value
