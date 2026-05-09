"""Run semantic tasks through short-horizon planning windows."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.safety import SafetySupervisor
from src.agent.short_horizon_planner import ShortHorizonPlanner
from src.agent.task_parser import TaskParser
from src.env_client.base import EnvClient
from src.env_client.schemas import SceneInstance
from src.tools.dispatcher import ToolDispatcher


class TaskRuntime:
    def __init__(
        self,
        env_client: EnvClient,
        recorder,
        task_parser: TaskParser,
        planner: ShortHorizonPlanner | None = None,
        safety: SafetySupervisor | None = None,
    ) -> None:
        self.env_client = env_client
        self.recorder = recorder
        self.task_parser = task_parser
        self.planner = planner or ShortHorizonPlanner()
        self.safety = safety or SafetySupervisor()
        self.tools = ToolDispatcher(env_client)

    def run_task(self, map_id: str, instruction: str, agent_id: str = "uav_1") -> MissionState:
        # Execute one short-horizon window per semantic phase.
        parse_result = self.task_parser.parse(instruction)
        self.recorder.write_json("semantic_plan.json", parse_result.to_dict())
        self.recorder.event("task_parsed", parse_result.to_dict())
        state = MissionState(
            run_id=self.recorder.run_dir.name,
            user_instruction=instruction,
            semantic_plan=parse_result.plan,
        )
        scene: SceneInstance | None = None
        try:
            scene = self.env_client.start_scene(map_id)
            self.recorder.write_json("scene_instance.json", scene.to_dict())
            self.recorder.event("scene_started", scene.to_dict())
            self._run_preflight(scene, agent_id, state)
            while not state.completed:
                self._run_phase_window(scene, agent_id, state)
                state.advance_phase()
                self._write_state(state)
            return state
        except Exception as exc:
            state.record_error(type(exc).__name__, str(exc))
            self._write_state(state)
            self.recorder.event("task_failed", {"error_type": type(exc).__name__, "error": str(exc)})
            if scene is not None and state.connected:
                self._try_hover(scene, agent_id, state)
            raise
        finally:
            if scene is not None:
                self.env_client.stop_scene(scene.scene_instance_id)
                self.recorder.event("scene_stopped", scene.to_dict())

    def _run_preflight(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        connect_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.connect", {})
        connect_data = connect_result.to_dict()
        state.record_action("uav.connect", {}, connect_data)
        self.recorder.event("control_connected", connect_data)
        if not connect_result.ok:
            raise RuntimeError(f"Preflight connect failed: {connect_result.message}")

        self.recorder.event("preflight_takeoff_started", {"scene_instance_id": scene.scene_instance_id})
        takeoff_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.takeoff", {})
        takeoff_data = takeoff_result.to_dict()
        state.record_action("uav.takeoff", {}, takeoff_data)
        self.recorder.event("preflight_takeoff_completed", takeoff_data)
        if not takeoff_result.ok:
            raise RuntimeError(f"Preflight takeoff failed: {takeoff_result.message}")

        hover_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
        hover_data = hover_result.to_dict()
        state.record_action("uav.hover", {}, hover_data)
        self.recorder.event("preflight_hover_completed", hover_data)
        if not hover_result.ok:
            raise RuntimeError(f"Preflight hover failed: {hover_result.message}")

        observation = self.tools.observe(scene.scene_instance_id, agent_id, ["rgb", "depth", "pose"])
        observation_data = observation.to_dict()
        state.record_observation(observation_data)
        self.recorder.event("initial_observation_updated", observation_data)
        self._write_state(state)

    def _run_phase_window(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        raw_plan = self.planner.plan_next(state)
        safety_decision = self.safety.review(raw_plan)
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
                observation = self.tools.observe(scene.scene_instance_id, agent_id, step.modalities)
                data = observation.to_dict()
                state.record_observation(data)
                self.recorder.event("observation_updated", data)
                self.recorder.event("step_completed", {**payload, "result": data})
            else:
                if not step.tool:
                    raise ValueError("tool step requires tool name")
                result = self.tools.call(scene.scene_instance_id, agent_id, step.tool, step.args)
                data = result.to_dict()
                state.record_action(step.tool, step.args, data)
                self.recorder.event("step_completed", {**payload, "result": data})
                if not result.ok:
                    raise RuntimeError(f"Action failed: {step.tool}: {result.message}")
            self._write_state(state)

    def _try_hover(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        try:
            result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            state.record_action("uav.hover", {}, result.to_dict())
            self.recorder.event("safety_hover_completed", result.to_dict())
        except Exception as exc:
            self.recorder.event("safety_hover_failed", {"error_type": type(exc).__name__, "error": str(exc)})

    def _write_state(self, state: MissionState) -> None:
        self.recorder.write_json("mission_state.json", state.to_dict())
        self.recorder.event("mission_state_updated", state.to_dict())
