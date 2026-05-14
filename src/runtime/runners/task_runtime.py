"""Run semantic tasks through short-horizon planning windows."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.planning.phase_transition import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK, CONTINUE_CURRENT_PHASE
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.agent.safety import SafetyDecision
from src.agent.planning.schemas import ShortHorizonPlan
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.task_parser import TaskParser
from src.env_client.base import EnvClient
from src.env_client.schemas import SceneInstance
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
from src.perception.raw_observation import RawObservation
from src.tools.dispatcher import ToolDispatcher


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
    ) -> None:
        if shutdown_policy not in self.SHUTDOWN_POLICIES:
            raise ValueError(f"Unsupported shutdown policy: {shutdown_policy}")
        self.env_client = env_client
        self.recorder = recorder
        self.task_parser = task_parser
        self.planner = planner or ModelActionPlanner()
        self.safety = safety or SafetySupervisor()
        self.shutdown_policy = shutdown_policy
        self.tools = ToolDispatcher(env_client)
        self.context_builder = ObservationContextBuilder(safety_config=self.safety.safety_config)
        self.memory_writer = MemoryEventWriter(self.recorder.run_dir)
        self.memory_reader = MemoryRecordReader(self.recorder.run_dir)
        self._preflight_scene_ids: set[str] = set()
        self._active_reused_scenes: dict[tuple[str, str | None], SceneInstance] = {}
        self._active_flight_memories: dict[tuple[str, str | None, str], FlightMemory] = {}
        self._state_flight_memories: dict[str, FlightMemory] = {}

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
        self.recorder.write_json("semantic_plan.json", parse_result.to_dict())
        self.recorder.event("task_parsed", parse_result.to_dict())
        scene = self._resolve_scene(map_id, scene_policy, scene_id)
        state = MissionState(
            run_id=self.recorder.run_dir.name,
            user_instruction=parse_result.instruction,
            semantic_plan=parse_result.plan,
            connected=scene.scene_instance_id in self._preflight_scene_ids,
            airborne=scene.scene_instance_id in self._preflight_scene_ids,
        )
        flight_memory = self._resolve_flight_memory(map_id, scene, agent_id, scene_policy, parse_result, state)
        self._state_flight_memories[state.run_id] = flight_memory
        try:
            self._prepare_scene_for_task(scene, agent_id, state)
            self._execute_phase_loop(scene, agent_id, state)
            if stop_scene_on_finish:
                self._run_shutdown(scene, agent_id, state)
            return state
        except Exception as exc:
            state.record_error(type(exc).__name__, str(exc))
            self._write_state(state)
            self.recorder.event("task_failed", {"error_type": type(exc).__name__, "error": str(exc)})
            if state.connected:
                self._try_hover(scene, agent_id, state)
            raise
        finally:
            if stop_scene_on_finish:
                self.env_client.stop_scene(scene.scene_instance_id)
                self._preflight_scene_ids.discard(scene.scene_instance_id)
                self._forget_scene(scene)
                self.recorder.event("scene_stopped", scene.to_dict())

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
            self.recorder.event("scene_stop_skipped", {"map_id": map_id, "scene_id": scene_id, "reason": "not_running"})
            return
        try:
            if state is not None:
                self._run_shutdown(scene, agent_id, state)
        finally:
            self.env_client.stop_scene(scene.scene_instance_id)
            self._preflight_scene_ids.discard(scene.scene_instance_id)
            self._forget_scene(scene)
            self.recorder.event("scene_stopped", scene.to_dict())

    def safe_hover_reused_scene(self, map_id: str, agent_id: str = "uav_1", scene_id: str | None = None) -> None:
        # Best-effort hover after an instruction failure in a reused scene.
        scene = self._active_reused_scenes.get(self._scene_key(map_id, scene_id))
        if scene is None:
            self.recorder.event("continuous_error_hover_skipped", {"reason": "not_running"})
            return
        try:
            result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            self.recorder.event("continuous_error_hover_completed", result.to_dict())
        except Exception as exc:
            self.recorder.event("continuous_error_hover_failed", {"error_type": type(exc).__name__, "error": str(exc)})

    def _resolve_scene(self, map_id: str, scene_policy: str, scene_id: str | None) -> SceneInstance:
        if scene_policy == "reuse_running":
            key = self._scene_key(map_id, scene_id)
            known_scene = self._active_reused_scenes.get(key)
            scene = self.env_client.get_or_start_scene(map_id, scene_id)
            self._active_reused_scenes[key] = scene
            self.recorder.write_json("scene_instance.json", scene.to_dict())
            if known_scene is not None and known_scene.scene_instance_id == scene.scene_instance_id:
                self.recorder.event("scene_reused", scene.to_dict())
            else:
                self.recorder.event("scene_started", scene.to_dict())
            return scene

        scene = self.env_client.start_scene(map_id, scene_id)
        self.recorder.write_json("scene_instance.json", scene.to_dict())
        self.recorder.event("scene_started", scene.to_dict())
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
        self.recorder.event("instruction_start_observation_updated", observation_data)
        self._write_state(state)

    def _execute_phase_loop(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        while not state.completed:
            state.record_phase_window_started()
            plan, safety_decision = self._run_phase_window(scene, agent_id, state)
            self._apply_phase_transition(scene, agent_id, state, plan, safety_decision)
            self._write_state(state)

    def _run_preflight(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        connect_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.connect", {})
        connect_data = connect_result.to_dict()
        self._record_action(state, "uav.connect", {}, connect_data)
        self.recorder.event("control_connected", connect_data)
        if not connect_result.ok:
            raise RuntimeError(f"Preflight connect failed: {connect_result.message}")

        self.recorder.event("preflight_takeoff_started", {"scene_instance_id": scene.scene_instance_id})
        takeoff_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.takeoff", {})
        takeoff_data = takeoff_result.to_dict()
        self._record_action(state, "uav.takeoff", {}, takeoff_data)
        self.recorder.event("preflight_takeoff_completed", takeoff_data)
        if not takeoff_result.ok:
            raise RuntimeError(f"Preflight takeoff failed: {takeoff_result.message}")

        hover_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
        hover_data = hover_result.to_dict()
        self._record_action(state, "uav.hover", {}, hover_data)
        self.recorder.event("preflight_hover_completed", hover_data)
        if not hover_result.ok:
            raise RuntimeError(f"Preflight hover failed: {hover_result.message}")

        observation_data = self._record_observation(
            scene,
            agent_id,
            state,
            ["rgb", "depth", "pose"],
            source="preflight",
        )
        self.recorder.event("initial_observation_updated", observation_data)
        self._write_state(state)

    def _run_phase_window(self, scene: SceneInstance, agent_id: str, state: MissionState) -> tuple[ShortHorizonPlan, SafetyDecision]:
        context = self.context_builder.build(state, self._memory_for_state(state))
        self.recorder.write_json("contexts/latest_context.json", context)
        raw_plan = self.planner.plan_next(state, context)
        if getattr(self.planner, "planner_mode", "") == "model":
            self.recorder.event("model_action_planned", raw_plan.to_dict())
        safety_decision = self.safety.review(raw_plan, state)
        plan = safety_decision.plan
        self.recorder.event("short_horizon_planned", plan.to_dict())
        if safety_decision.interventions:
            self.recorder.event(
                "safety_intervened",
                {"phase": plan.phase, "interventions": safety_decision.interventions},
            )
        for index, step in enumerate(plan.steps):
            payload = {"phase": plan.phase, "step_index": index, "step": step.to_dict()}
            self.recorder.event("step_started", payload)
            if step.kind == "observe":
                data = self._record_observation(scene, agent_id, state, step.modalities, source="manual_observe_step")
                self.recorder.event("observation_updated", data)
                self.recorder.event("step_completed", {**payload, "result": data})
            else:
                if not step.tool:
                    raise ValueError("tool step requires tool name")
                result = self.tools.call(scene.scene_instance_id, agent_id, step.tool, step.args)
                data = result.to_dict()
                self._record_action(state, step.tool, step.args, data)
                self.recorder.event("step_completed", {**payload, "result": data})
                if not result.ok:
                    raise RuntimeError(f"Action failed: {step.tool}: {result.message}")
            self._write_state(state)
        self._apply_automatic_observation_policy(scene, agent_id, state, plan.phase)
        return plan, safety_decision

    def _apply_phase_transition(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        plan: ShortHorizonPlan,
        safety_decision: SafetyDecision,
    ) -> None:
        transition = plan.phase_transition
        payload = {
            "phase": plan.phase,
            "decision": transition.decision,
            "reason": transition.reason,
            "current_phase_window_count": state.current_phase_window_count,
            "max_phase_window_count": state.max_phase_window_count,
        }
        if safety_decision.interventions:
            self.recorder.event(
                "phase_transition_suppressed_by_safety",
                {**payload, "interventions": safety_decision.interventions},
            )
            if state.current_phase_window_count >= state.max_phase_window_count:
                self.recorder.event("phase_transition_window_limit_reached", {**payload, "reason": "safety_intervention_loop_guard"})
                self._try_hover(scene, agent_id, state)
                state.advance_phase()
                self.recorder.event("phase_transition_applied", {**payload, "applied_decision": ADVANCE_TO_NEXT_PHASE})
            return
        if transition.decision == CONTINUE_CURRENT_PHASE:
            self.recorder.event("phase_transition_continued", payload)
            if state.current_phase_window_count >= state.max_phase_window_count:
                self.recorder.event("phase_transition_window_limit_reached", payload)
                self._try_hover(scene, agent_id, state)
                state.advance_phase()
                self.recorder.event("phase_transition_applied", {**payload, "applied_decision": ADVANCE_TO_NEXT_PHASE})
            return
        if transition.decision == ADVANCE_TO_NEXT_PHASE:
            state.advance_phase()
            self.recorder.event("phase_transition_applied", payload)
            return
        if transition.decision == COMPLETE_TASK:
            state.mark_completed()
            self.recorder.event("phase_transition_applied", payload)
            return
        raise ValueError(f"Unsupported phase transition decision: {transition.decision}")

    def _apply_automatic_observation_policy(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: MissionState,
        phase: str,
    ) -> None:
        modalities = ["rgb", "depth", "pose"]
        policy = {
            "mode": "automatic",
            "modalities": modalities,
            "sample_timing": "after_each_phase_window",
            "phase": phase,
        }
        self.recorder.event("automatic_observation_policy_applied", policy)
        data = self._record_observation(scene, agent_id, state, modalities, source="automatic_phase_window")
        self.recorder.event("automatic_observation_updated", data)
        self._write_state(state)

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
        self.recorder.write_json("observations/latest_observation.json", data)
        self.recorder.write_json("contexts/latest_context.json", self.context_builder.build(state, self._memory_for_state(state)))
        return data

    def _try_hover(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        try:
            result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            self._record_action(state, "uav.hover", {}, result.to_dict())
            self.recorder.event("safety_hover_completed", result.to_dict())
        except Exception as exc:
            self.recorder.event("safety_hover_failed", {"error_type": type(exc).__name__, "error": str(exc)})

    def _run_shutdown(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        if self.shutdown_policy == "none" or not state.connected:
            self.recorder.event("shutdown_skipped", {"policy": self.shutdown_policy, "connected": state.connected})
            return

        self.recorder.event("shutdown_started", {"policy": self.shutdown_policy})
        hover_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
        hover_data = hover_result.to_dict()
        self._record_action(state, "uav.hover", {}, hover_data)
        self.recorder.event("shutdown_hover_completed", hover_data)
        if not hover_result.ok:
            raise RuntimeError(f"Shutdown hover failed: {hover_result.message}")

        if self.shutdown_policy == "land":
            self.recorder.event("shutdown_land_started", {"scene_instance_id": scene.scene_instance_id})
            land_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.land", {})
            land_data = land_result.to_dict()
            self._record_action(state, "uav.land", {}, land_data)
            self.recorder.event("shutdown_land_completed", land_data)
            if not land_result.ok:
                raise RuntimeError(f"Shutdown land failed: {land_result.message}")

        self._write_state(state)

    def _write_state(self, state: MissionState) -> None:
        self.recorder.write_json("mission_state.json", state.to_dict())
        self.recorder.write_json("flight_memory.json", self._memory_for_state(state).to_dict())
        self.recorder.event("mission_state_updated", state.to_dict())

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
