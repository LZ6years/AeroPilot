"""Test current mission state transitions."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.semantic.schemas import SemanticPhase, SemanticPlan, TaskGoal
from src.agent.semantic.task_parser import TaskParser
from src.uav import pose_measurements


def make_state() -> MissionState:
    result = TaskParser(mode="rule_based").parse("前面右转会有一个黑色的路灯，右转面向路灯后停下")
    return MissionState("run_test", result.instruction, result.plan)


def test_mission_state_default_phase_window_limit_is_40():
    assert make_state().max_phase_window_count == 40


def test_mission_state_serializes_resolved_observation_modalities():
    state = make_state()
    state.observation_modalities = ("rgb", "pose")

    assert state.to_dict()["observation_modalities"] == ["rgb", "pose"]


def movement_phase(index: int, total: int, name: str, goal: str) -> SemanticPhase:
    return SemanticPhase(
        index=index,
        total=total,
        name=name,
        goal=goal,
        done_when=[f"{goal} is complete"],
        continue_when=[f"{goal} is incomplete"],
        metrics=[],
    )


def test_mission_state_advances_phase():
    state = make_state()

    assert state.current_phase_name == "orient_right"
    state.advance_phase()

    assert state.current_phase_name == "search_landmark"


def test_mission_state_tracks_semantic_plan_version():
    state = make_state()
    original_version = state.semantic_plan_version

    state.apply_semantic_plan_update(state.semantic_plan)

    assert state.semantic_plan_version == original_version + 1
    assert state.to_dict()["semantic_plan_version"] == 1


def test_mission_state_records_action_and_observation():
    state = make_state()

    state.record_action("uav.connect", {}, {"ok": True})
    state.record_action("uav.takeoff", {}, {"ok": True})
    state.record_observation({"modalities": ["rgb", "depth", "pose"], "pose": {"x": 1.0, "velocity": {"x": 2.0}}})

    assert state.connected
    assert state.airborne
    assert state.latest_observation["modalities"] == ["rgb", "depth", "pose"]
    assert "velocity" not in state.latest_observation["pose"]


def test_mission_state_error_requires_hover():
    state = make_state()

    state.record_error("RuntimeError", "boom")

    assert state.hover_required
    assert state.errors[-1]["error_type"] == "RuntimeError"


def test_mission_state_tracks_phase_execution_facts():
    state = make_state()

    state.record_phase_window_started()
    state.ensure_current_phase_progress({"source": "preflight", "pose": {"x": 0.0, "y": 0.0, "z": -2.0}})
    state.update_current_phase_progress(
        {"source": "automatic", "pose": {"x": 1.0, "y": 0.0, "z": -2.0}},
        last_window_measurements={"planar_displacement_m": 1.0},
        tool_facts_delta={
            "total_steps": 1,
            "successful_steps": 1,
            "move_duration_success_sum_sec": 0.5,
            "move_duration_by_direction_success": {"forward": 0.5},
            "tools_executed": {"uav.move_body": 1},
        },
    )

    data = state.to_dict()

    assert "phase_start_observations" not in data
    assert "phase_window_records" not in data
    assert "current_phase_execution_summary" not in data
    assert "latest_pose" not in data
    assert "yaw_scan_accum_deg" not in data
    assert data["safety"]["yaw_scan"] == {
        "offset_deg": 0.0,
        "extent_deg": 0.0,
    }
    assert "world_pose" not in data["current_phase_progress"]
    assert "start_pose" not in data["current_phase_progress"]
    assert "latest_pose" not in data["current_phase_progress"]
    assert "phase_origin_yaw_deg" not in data["current_phase_progress"]
    assert "phase_local_frame" not in data["current_phase_progress"]
    assert data["current_phase_progress"]["last_window"]["planar_m"] == 1.0
    assert "latest_window_measurements" not in data["current_phase_progress"]
    assert data["current_phase_progress"]["tool_facts"]["move_duration_success_sum_sec"] == 0.5
    assert data["current_phase_progress"]["measurements"]["forward_m"] == 1.0
    assert "coordinate_frames" not in data["current_phase_progress"]
    assert data["current_phase_key"] == "1:orient_right"


def test_mission_state_tracks_one_sided_yaw_scan_extent_and_resets_per_phase():
    state = make_state()

    state.record_action("uav.turn_relative", {"yaw_deg": 30.0}, {"ok": True})
    state.record_action("uav.turn_relative", {"yaw_deg": -20.0}, {"ok": True})

    assert state.yaw_scan_offset_deg == 10.0
    assert state.yaw_scan_extent_deg == 30.0

    state.record_action("uav.turn_relative", {"yaw_deg": -40.0}, {"ok": True})

    assert state.yaw_scan_offset_deg == -30.0
    assert state.yaw_scan_extent_deg == 30.0

    state.advance_phase()

    assert state.yaw_scan_offset_deg == 0.0
    assert state.yaw_scan_extent_deg == 0.0


def test_mission_state_uses_phase_index_to_separate_duplicate_phase_names():
    plan = SemanticPlan(
        intent="movement",
        task_goal=TaskGoal("move forward twice", "hover_after_movement"),
        phases=[
            movement_phase(1, 2, "move_forward", "first forward segment"),
            movement_phase(2, 2, "move_forward", "second forward segment"),
        ],
    )
    state = MissionState("run_test", "move twice", plan)

    state.record_phase_window_started()
    state.ensure_current_phase_progress({"pose": {"x": 0.0}})
    state.update_current_phase_progress({"pose": {"x": 1.0}})
    first_progress = state.current_phase_execution_summary
    state.advance_phase()
    state.record_phase_window_started()
    state.ensure_current_phase_progress({"pose": {"x": 10.0}})
    state.update_current_phase_progress({"pose": {"x": 11.0}})

    assert first_progress["phase_key"] == "1:move_forward"
    assert state.current_phase_execution_summary["phase_key"] == "2:move_forward"
    assert state.current_phase_execution_summary["measurements"]["forward_m"] == 1.0


def test_phase_local_projection_uses_phase_start_yaw():
    plan = SemanticPlan(
        intent="movement",
        task_goal=TaskGoal("move forward", "hover_after_movement"),
        phases=[movement_phase(1, 1, "move_forward", "move forward")],
    )
    state = MissionState("run_test", "move forward", plan)
    state.record_phase_window_started()
    state.ensure_current_phase_progress({"pose": {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 0.0}}})
    state.update_current_phase_progress({"pose": {"position": {"x": 5.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 0.0}}})

    measurements = state.current_phase_progress["measurements"]
    assert measurements["forward_m"] == 5.0
    assert measurements["right_m"] == 0.0


def test_phase_local_projection_handles_90_degree_yaw():
    plan = SemanticPlan(
        intent="movement",
        task_goal=TaskGoal("move forward", "hover_after_movement"),
        phases=[movement_phase(1, 1, "move_forward", "move forward")],
    )
    state = MissionState("run_test", "move forward", plan)
    state.record_phase_window_started()
    state.ensure_current_phase_progress(
        {"pose": {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 3.141592653589793 / 2}}}
    )
    state.update_current_phase_progress(
        {"pose": {"position": {"x": 0.0, "y": 5.0, "z": -2.0}, "euler_rad": {"yaw": 3.141592653589793 / 2}}}
    )

    measurements = state.current_phase_progress["measurements"]
    assert measurements["forward_m"] == 5.0
    assert measurements["right_m"] == 0.0


def test_shared_pose_measurements_cover_cardinal_yaws():
    assert pose_measurements(
        {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 0.0}},
        {"position": {"x": 5.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 0.0}},
    )["forward_displacement_m"] == 5.0
    assert pose_measurements(
        {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 3.141592653589793 / 2}},
        {"position": {"x": 0.0, "y": 5.0, "z": -2.0}, "euler_rad": {"yaw": 3.141592653589793 / 2}},
    )["forward_displacement_m"] == 5.0
    assert pose_measurements(
        {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 3.141592653589793}},
        {"position": {"x": -5.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 3.141592653589793}},
    )["forward_displacement_m"] == 5.0
    assert pose_measurements(
        {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": -3.141592653589793 / 2}},
        {"position": {"x": 0.0, "y": -5.0, "z": -2.0}, "euler_rad": {"yaw": -3.141592653589793 / 2}},
    )["forward_displacement_m"] == 5.0
