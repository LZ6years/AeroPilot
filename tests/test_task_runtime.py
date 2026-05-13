"""Test M4 task runtime dry-run behavior."""

from __future__ import annotations

import json

from src.agent.schemas import PlannedStep, ShortHorizonPlan
from src.agent.task_parser import TaskParser
from src.env_client.local_client import LocalEnvClient
from src.env_client.schemas import MapPackage
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import LocalSimulationServer


LAMP_TASK = "前面右转会有一个黑色的路灯，右转面向路灯后停下"
ENTRANCE_TASK = "左转寻找建筑入口，并停在入口前"


def make_package() -> MapPackage:
    return MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "packaged_unreal",
            "scene_id": "1",
            "root": None,
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )


def load_event_records(run_dir):
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def test_task_runtime_completes_dry_run_and_writes_records(tmp_path):
    run_dir = tmp_path / "task_run"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(LocalEnvClient(server), recorder, TaskParser(mode="rule_based"))

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    assert state.completed
    assert state.connected
    assert state.airborne
    assert state.recent_observations
    assert (run_dir / "semantic_plan.json").exists()
    assert (run_dir / "mission_state.json").exists()
    assert (run_dir / "observations/latest_observation.json").exists()
    assert (run_dir / "contexts/latest_context.json").exists()

    event_records = load_event_records(run_dir)
    event_names = [record["event"] for record in event_records]
    short_horizon_tools = [
        step.get("tool")
        for record in event_records
        if record["event"] == "short_horizon_planned"
        for step in record["payload"]["steps"]
        if step["kind"] == "tool"
    ]
    mission_state = json.loads((run_dir / "mission_state.json").read_text(encoding="utf-8"))
    latest_observation = json.loads((run_dir / "observations/latest_observation.json").read_text(encoding="utf-8"))
    latest_context = json.loads((run_dir / "contexts/latest_context.json").read_text(encoding="utf-8"))

    assert "control_connected" in event_names
    assert "preflight_takeoff_started" in event_names
    assert "preflight_takeoff_completed" in event_names
    assert "preflight_hover_completed" in event_names
    assert "initial_observation_updated" in event_names
    assert "short_horizon_planned" in event_names
    assert "step_completed" in event_names
    assert "observation_updated" in event_names
    assert "shutdown_started" in event_names
    assert "shutdown_hover_completed" in event_names
    assert "scene_stopped" in event_names
    assert "uav.connect" not in short_horizon_tools
    assert "uav.takeoff" not in short_horizon_tools
    assert mission_state["connected"] is True
    assert mission_state["airborne"] is True
    assert mission_state["recent_observations"]
    assert mission_state["recent_observations"][-1]["source"] == "phase_step"
    assert "pose" in mission_state["recent_observations"][-1]
    assert "frames" in mission_state["recent_observations"][-1]
    assert latest_observation["source"] == "phase_step"
    assert latest_context["active_instruction"]
    assert latest_context["current_phase"] is not None
    fragments = {fragment["name"]: fragment for fragment in latest_context["fragments"]}
    assert fragments["image"]["content"]["recent_observation_sources"][-1] == "phase_step"
    assert "uav.takeoff" in latest_context["allowed_action_constraints"]["lifecycle_tools_excluded"]


def test_task_runtime_can_land_on_shutdown(tmp_path):
    run_dir = tmp_path / "task_run_land"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        shutdown_policy="land",
    )

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    mission_state = json.loads((run_dir / "mission_state.json").read_text(encoding="utf-8"))

    assert state.completed
    assert state.connected
    assert state.airborne is False
    assert mission_state["airborne"] is False
    assert mission_state["recent_actions"][-1]["tool"] == "uav.land"
    assert "shutdown_land_started" in event_names
    assert "shutdown_land_completed" in event_names


def test_task_runtime_can_reuse_existing_parse_result(tmp_path):
    class ExplodingParser:
        def parse(self, instruction: str):
            raise AssertionError("parser should not be called")

    parse_result = TaskParser(mode="rule_based").parse(ENTRANCE_TASK)
    run_dir = tmp_path / "parsed_task_run"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(LocalEnvClient(server), recorder, ExplodingParser())

    state = runtime.run_parsed_task("m1", parse_result)
    recorder.close()

    semantic_plan = json.loads((run_dir / "semantic_plan.json").read_text(encoding="utf-8"))
    assert state.completed
    assert state.user_instruction == ENTRANCE_TASK
    assert semantic_plan["instruction"] == ENTRANCE_TASK
    assert semantic_plan["plan"] == parse_result.plan.to_dict()


def test_task_runtime_model_planner_path_records_model_action_and_safety(tmp_path):
    class FakeModelPlanner:
        planner_mode = "model"

        def __init__(self) -> None:
            self.contexts = []

        def plan_next(self, state, context=None):
            self.contexts.append(context)
            return ShortHorizonPlan(
                state.current_phase_name,
                [
                    PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 999.0}, "oversized model suggestion"),
                    PlannedStep.observe_step(),
                ],
            )

    run_dir = tmp_path / "model_planner_task_run"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    planner = FakeModelPlanner()
    runtime = TaskRuntime(LocalEnvClient(server), recorder, TaskParser(mode="rule_based"), planner=planner)

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    planned_records = [record for record in events if record["event"] == "short_horizon_planned"]

    assert state.completed
    assert planner.contexts
    assert planner.contexts[0]["current_phase"]["name"] == "orient_right"
    assert "fragments" in planner.contexts[0]
    assert "model_action_planned" in event_names
    assert "safety_intervened" in event_names
    assert planned_records[0]["payload"]["steps"][0]["args"]["yaw_deg"] == 30.0


def test_task_runtime_reuse_running_scene_does_not_repeat_preflight(tmp_path):
    run_dir = tmp_path / "continuous_task_run"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(LocalEnvClient(server), recorder, TaskParser(mode="rule_based"), shutdown_policy="none")

    first_state = runtime.run_task("m1", LAMP_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    second_state = runtime.run_task("m1", ENTRANCE_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    runtime.finish_reused_scene("m1", second_state)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    scene_started = [record for record in events if record["event"] == "scene_started"]
    scene_reused = [record for record in events if record["event"] == "scene_reused"]

    assert first_state.completed
    assert second_state.completed
    assert second_state.connected
    assert second_state.airborne
    assert second_state.recent_observations[0]["source"] == "instruction_start"
    flight_memory = json.loads((run_dir / "flight_memory.json").read_text(encoding="utf-8"))
    assert len(flight_memory["instruction_history"]) >= 2
    assert flight_memory["active_instruction"] == ENTRANCE_TASK
    assert flight_memory["recent_actions"]
    assert event_names.count("control_connected") == 1
    assert event_names.count("preflight_takeoff_completed") == 1
    assert "instruction_start_observation_updated" in event_names
    assert len(scene_started) == 1
    assert len(scene_reused) == 1
    assert scene_started[0]["payload"]["scene_instance_id"] == scene_reused[0]["payload"]["scene_instance_id"]
    assert event_names.count("scene_stopped") == 1


def test_task_runtime_can_leave_reused_scene_running(tmp_path):
    run_dir = tmp_path / "leave_scene_running"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(LocalEnvClient(server), recorder, TaskParser(mode="rule_based"), shutdown_policy="none")

    state = runtime.run_task("m1", LAMP_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    running_scene = server.get_or_start_scene("m1")

    assert state.completed
    assert "scene_stopped" not in event_names
    assert running_scene.status == "running"
