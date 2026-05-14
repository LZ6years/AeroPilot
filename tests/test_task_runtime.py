"""Test M4 task runtime dry-run behavior."""

from __future__ import annotations

import json

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.planning.phase_transition import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK, CONTINUE_CURRENT_PHASE, PhaseTransition
from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.agent.semantic.task_parser import TaskParser
from src.env_client.local_client import LocalEnvClient
from src.env_client.schemas import MapPackage
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.configs.model_config import ModelConfig
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


class HoverOnlyPlanner:
    planner_mode = "test"

    def plan_next(self, state, context=None):
        return ShortHorizonPlan(
            state.current_phase_name,
            [PlannedStep.tool_step("uav.hover", {}, "test planner holds position")],
        )


def test_task_runtime_completes_dry_run_and_writes_records(tmp_path):
    run_dir = tmp_path / "task_run"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(LocalEnvClient(server), recorder, TaskParser(mode="rule_based"), planner=HoverOnlyPlanner())

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
    assert (run_dir / "flight_memory.json").exists()
    assert (run_dir / "memory/instructions.jsonl").exists()
    assert (run_dir / "memory/actions.jsonl").exists()
    assert (run_dir / "memory/poses.jsonl").exists()
    assert (run_dir / "memory/observations.jsonl").exists()
    assert (run_dir / "memory/frames.jsonl").exists()

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
    flight_memory = json.loads((run_dir / "flight_memory.json").read_text(encoding="utf-8"))

    assert "control_connected" in event_names
    assert "preflight_takeoff_started" in event_names
    assert "preflight_takeoff_completed" in event_names
    assert "preflight_hover_completed" in event_names
    assert "initial_observation_updated" in event_names
    assert "short_horizon_planned" in event_names
    assert "step_completed" in event_names
    assert "automatic_observation_policy_applied" in event_names
    assert "automatic_observation_updated" in event_names
    assert "shutdown_started" in event_names
    assert "shutdown_hover_completed" in event_names
    assert "scene_stopped" in event_names
    assert "uav.connect" not in short_horizon_tools
    assert "uav.takeoff" not in short_horizon_tools
    assert mission_state["connected"] is True
    assert mission_state["airborne"] is True
    assert mission_state["recent_observations"]
    assert mission_state["recent_observations"][-1]["source"] == "automatic_phase_window"
    assert "pose" in mission_state["recent_observations"][-1]
    assert "frames" in mission_state["recent_observations"][-1]
    assert latest_observation["source"] == "automatic_phase_window"
    assert latest_context["active_instruction"]
    assert latest_context["current_phase"] is not None
    assert "context_schema" in latest_context
    assert "image" in latest_context["context_schema"]["module_content_fields"]
    fragments = {fragment["name"]: fragment for fragment in latest_context["fragments"]}
    assert fragments["image"]["content"]["recent_observation_sources"][-1] == "automatic_phase_window"
    assert "uav.takeoff" in latest_context["allowed_action_constraints"]["lifecycle_tools_excluded"]
    assert "recent_actions" not in flight_memory
    assert "recent_observations" not in flight_memory
    assert flight_memory["recent_action_refs"]
    assert flight_memory["recent_observation_refs"]


def test_task_runtime_can_land_on_shutdown(tmp_path):
    run_dir = tmp_path / "task_run_land"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
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
    runtime = TaskRuntime(LocalEnvClient(server), recorder, ExplodingParser(), planner=HoverOnlyPlanner())

    state = runtime.run_parsed_task("m1", parse_result)
    recorder.close()

    semantic_plan = json.loads((run_dir / "semantic_plan.json").read_text(encoding="utf-8"))
    assert state.completed
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
    assert "context_schema" in planner.contexts[0]
    assert "module_content_fields" in planner.contexts[0]["context_schema"]
    assert "image" in planner.contexts[0]["context_schema"]["module_content_fields"]
    assert "recent_rgb_frame_metadata" in planner.contexts[0]["context_schema"]["module_content_fields"]["image"]["content_fields"]
    action_constraints = planner.contexts[0]["allowed_action_constraints"]
    assert action_constraints["planner_tools"] == ["uav.hover", "uav.turn_relative", "uav.move_body"]
    assert "observe_step_schema" not in action_constraints
    assert planner.contexts[0]["observation_policy"]["mode"] == "automatic"
    assert set(action_constraints["action_schemas"]) == {"uav.hover", "uav.turn_relative", "uav.move_body"}
    assert action_constraints["action_schemas"]["uav.turn_relative"]["limits"] == {
        "yaw_deg_min": -30.0,
        "yaw_deg_max": 30.0,
    }
    assert action_constraints["action_schemas"]["uav.move_body"]["limits"] == {
        "abs_vx_vy_max": 6.0,
        "abs_vz_max": 3.0,
        "duration_max": 8.0,
    }
    assert "env.observe" in action_constraints["forbidden_tools"]
    assert "uav.set_pose" in action_constraints["forbidden_tools"]
    assert "flight_memory" not in planner.contexts[0]
    assert "instruction_history" not in planner.contexts[0]
    assert "recent_observations" not in planner.contexts[0]
    assert "recent_actions" not in planner.contexts[0]
    assert "model_action_planned" in event_names
    assert "safety_intervened" in event_names
    assert "automatic_observation_updated" in event_names
    assert "phase_transition_suppressed_by_safety" in event_names
    assert "phase_transition_window_limit_reached" in event_names
    assert planned_records[0]["payload"]["steps"][0]["args"]["yaw_deg"] == 30.0


def test_task_runtime_reuse_running_scene_does_not_repeat_preflight(tmp_path):
    run_dir = tmp_path / "continuous_task_run"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )

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
    latest_context = json.loads((run_dir / "contexts/latest_context.json").read_text(encoding="utf-8"))
    assert len(flight_memory["instruction_history"]) >= 2
    assert flight_memory["active_instruction"] == ENTRANCE_TASK
    assert flight_memory["recent_action_refs"]
    assert flight_memory["recent_observation_refs"]
    assert (run_dir / "memory/instructions.jsonl").read_text(encoding="utf-8").count("\n") >= 2
    assert (run_dir / "memory/actions.jsonl").exists()
    assert (run_dir / "memory/observations.jsonl").exists()
    assert (run_dir / "memory/poses.jsonl").exists()
    assert (run_dir / "memory/frames.jsonl").exists()
    assert latest_context["active_instruction"] == ENTRANCE_TASK
    assert latest_context["metadata"]["run_id"] == flight_memory["run_id"]
    assert "fragments" in latest_context
    assert "instruction_history" not in latest_context
    assert "recent_actions" not in latest_context
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
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )

    state = runtime.run_task("m1", LAMP_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    running_scene = server.get_or_start_scene("m1")

    assert state.completed
    assert "scene_stopped" not in event_names
    assert running_scene.status == "running"


def test_task_runtime_model_planner_keeps_memory_module_windows_across_many_instructions(tmp_path):
    class ContextEchoModelPlanner(ModelActionPlanner):
        def __init__(self) -> None:
            self.contexts = []
            super().__init__(
                ModelConfig(
                    task_parser_mode="model",
                    model_name="stub",
                    api_key="test-key",
                    base_url="https://example.test/v1",
                )
            )

        def _call_openai_compatible(self, context: dict):
            self.contexts.append(context)
            return {
                "phase": context["current_phase"]["name"],
                "steps": [
                    {"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold for context verification"},
                ],
                "phase_transition": {
                    "decision": ADVANCE_TO_NEXT_PHASE,
                    "reason": "context verification phase complete",
                },
            }

    instructions = [
        "右转寻找黑色路灯",
        "左转寻找建筑入口",
        "寻找红色车辆",
        "继续寻找黑色路灯",
        "确认建筑入口位置",
        "观察红色车辆附近区域",
        "面向黑色路灯后悬停",
        "停在建筑入口前",
        "靠近红色车辆并保持安全距离",
        "右转看有没有黑色路灯",
        "左转寻找建筑入口并停下",
        "寻找红色车辆然后悬停",
    ]
    run_dir = tmp_path / "many_instruction_model_run"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    planner = ContextEchoModelPlanner()
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    state = None
    for instruction in instructions:
        state = runtime.run_task("m1", instruction, scene_policy="reuse_running", stop_scene_on_finish=False)
    runtime.finish_reused_scene("m1", state)
    recorder.close()

    latest_context = json.loads((run_dir / "contexts/latest_context.json").read_text(encoding="utf-8"))
    flight_memory = json.loads((run_dir / "flight_memory.json").read_text(encoding="utf-8"))
    events = load_event_records(run_dir)
    fragments = {fragment["name"]: fragment for fragment in latest_context["fragments"]}

    assert state is not None and state.completed
    assert len(planner.contexts) >= len(instructions)
    assert latest_context["active_instruction"] == instructions[-1]
    assert "context_schema" in latest_context
    assert "action" in latest_context["context_schema"]["module_content_fields"]
    assert len(flight_memory["instruction_history"]) >= len(instructions)
    assert (run_dir / "memory/instructions.jsonl").read_text(encoding="utf-8").count("\n") >= len(instructions)
    assert fragments["instruction"]["content"]["active_instruction"] == instructions[-1]
    assert len(fragments["instruction"]["content"]["instruction_history"]) == 5
    assert len(fragments["action"]["content"]["recent_actions"]) == 5
    assert len(fragments["pose"]["content"]["recent_pose_history"]) <= 3
    assert len(fragments["image"]["content"]["recent_observation_sources"]) <= 3
    assert fragments["image"]["artifact_refs"]
    assert fragments["map"]["content"]["recalled_map_memories"] == []
    assert "flight_memory" not in latest_context
    assert "recent_actions" not in latest_context
    assert "instruction_history" not in latest_context
    assert "model_action_planned" in [record["event"] for record in events]


def test_task_runtime_continue_transition_keeps_same_phase_for_next_window(tmp_path):
    class ContinueThenCompletePlanner:
        planner_mode = "test"

        def __init__(self) -> None:
            self.seen_phases = []

        def plan_next(self, state, context=None):
            self.seen_phases.append(context["current_phase"]["name"])
            decision = CONTINUE_CURRENT_PHASE if len(self.seen_phases) == 1 else COMPLETE_TASK
            return ShortHorizonPlan(
                state.current_phase_name,
                [PlannedStep.tool_step("uav.hover", {}, "hold while checking current phase")],
                PhaseTransition(decision, "test transition"),
            )

    run_dir = tmp_path / "continue_transition"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    planner = ContinueThenCompletePlanner()
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    event_names = [record["event"] for record in load_event_records(run_dir)]
    assert state.completed
    assert planner.seen_phases[:2] == [planner.seen_phases[0], planner.seen_phases[0]]
    assert "phase_transition_continued" in event_names
    assert "phase_transition_applied" in event_names


def test_task_runtime_complete_transition_finishes_without_advancing_phase(tmp_path):
    class CompletePlanner:
        planner_mode = "test"

        def __init__(self) -> None:
            self.calls = 0

        def plan_next(self, state, context=None):
            self.calls += 1
            return ShortHorizonPlan(
                state.current_phase_name,
                [PlannedStep.tool_step("uav.hover", {}, "final state reached")],
                PhaseTransition(COMPLETE_TASK, "task final state reached"),
            )

    run_dir = tmp_path / "complete_transition"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    planner = CompletePlanner()
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    assert state.completed
    assert state.current_phase_index == 0
    assert planner.calls == 1


def test_task_runtime_continue_transition_window_limit_advances_phase(tmp_path):
    class AlwaysContinuePlanner:
        planner_mode = "test"

        def plan_next(self, state, context=None):
            return ShortHorizonPlan(
                state.current_phase_name,
                [PlannedStep.tool_step("uav.hover", {}, "keep checking current phase")],
                PhaseTransition(CONTINUE_CURRENT_PHASE, "not enough evidence"),
            )

    run_dir = tmp_path / "continue_loop_guard"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=AlwaysContinuePlanner(),
        shutdown_policy="none",
    )

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    assert state.completed
    assert "phase_transition_window_limit_reached" in event_names
    mission_state = json.loads((run_dir / "mission_state.json").read_text(encoding="utf-8"))
    assert mission_state["max_phase_window_count"] == 5
    assert mission_state["phase_window_counts"]
