"""Run semantic tasks through short-horizon planning windows."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.safety import SafetySupervisor
from src.agent.schemas import TaskParseResult
from src.agent.short_horizon_planner import ShortHorizonPlanner
from src.agent.task_parser import TaskParser
from src.env_client.base import EnvClient
from src.env_client.schemas import SceneInstance
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
        self.planner = planner or ShortHorizonPlanner()
        self.safety = safety or SafetySupervisor()
        self.shutdown_policy = shutdown_policy
        self.tools = ToolDispatcher(env_client)
        self.context_builder = ObservationContextBuilder(self.safety.action_config)
        self._preflight_scene_ids: set[str] = set()
        self._active_reused_scenes: dict[tuple[str, str | None], SceneInstance] = {}

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
            self._run_phase_window(scene, agent_id, state)
            state.advance_phase()
            self._write_state(state)

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

        observation_data = self._record_observation(
            scene,
            agent_id,
            state,
            ["rgb", "depth", "pose"],
            source="preflight",
        )
        self.recorder.event("initial_observation_updated", observation_data)
        self._write_state(state)

    def _run_phase_window(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        context = self.context_builder.build(state)
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
                data = self._record_observation(scene, agent_id, state, step.modalities, source="phase_step")
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
        self.recorder.write_json("observations/latest_observation.json", data)
        self.recorder.write_json("contexts/latest_context.json", self.context_builder.build(state))
        return data

    def _try_hover(self, scene: SceneInstance, agent_id: str, state: MissionState) -> None:
        try:
            result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            state.record_action("uav.hover", {}, result.to_dict())
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
        state.record_action("uav.hover", {}, hover_data)
        self.recorder.event("shutdown_hover_completed", hover_data)
        if not hover_result.ok:
            raise RuntimeError(f"Shutdown hover failed: {hover_result.message}")

        if self.shutdown_policy == "land":
            self.recorder.event("shutdown_land_started", {"scene_instance_id": scene.scene_instance_id})
            land_result = self.tools.call(scene.scene_instance_id, agent_id, "uav.land", {})
            land_data = land_result.to_dict()
            state.record_action("uav.land", {}, land_data)
            self.recorder.event("shutdown_land_completed", land_data)
            if not land_result.ok:
                raise RuntimeError(f"Shutdown land failed: {land_result.message}")

        self._write_state(state)

    def _write_state(self, state: MissionState) -> None:
        self.recorder.write_json("mission_state.json", state.to_dict())
        self.recorder.event("mission_state_updated", state.to_dict())

    @staticmethod
    def _scene_key(map_id: str, scene_id: str | None) -> tuple[str, str | None]:
        return (map_id, str(scene_id) if scene_id is not None else None)

    def _forget_scene(self, scene: SceneInstance) -> None:
        for key, active_scene in list(self._active_reused_scenes.items()):
            if active_scene.scene_instance_id == scene.scene_instance_id:
                self._active_reused_scenes.pop(key, None)
