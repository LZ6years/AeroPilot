"""Test current mission state transitions."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.semantic.schemas import SemanticGoal, SemanticPhase, SemanticPlan, SemanticTarget
from src.agent.semantic.task_parser import TaskParser


def make_state() -> MissionState:
    result = TaskParser(mode="rule_based").parse("前面右转会有一个黑色的路灯，右转面向路灯后停下")
    return MissionState("run_test", result.instruction, result.plan)


def test_mission_state_advances_phase():
    state = make_state()

    assert state.current_phase_name == "orient_right"
    state.advance_phase()

    assert state.current_phase_name == "search_landmark"


def test_mission_state_records_action_and_observation():
    state = make_state()

    state.record_action("uav.connect", {}, {"ok": True})
    state.record_action("uav.takeoff", {}, {"ok": True})
    state.record_observation({"modalities": ["rgb", "depth", "pose"]})

    assert state.connected
    assert state.airborne
    assert state.latest_observation["modalities"] == ["rgb", "depth", "pose"]


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
        latest_window_measurements={"planar_displacement_m": 1.0},
        tool_facts_delta={
            "total_steps": 1,
            "successful_steps": 1,
            "move_body_duration_success_sum": 0.5,
            "move_body_duration_by_direction_success": {"forward": 0.5},
            "tools_executed": {"uav.move_body": 1},
        },
    )

    data = state.to_dict()

    assert "phase_start_observations" not in data
    assert "phase_window_records" not in data
    assert "current_phase_execution_summary" not in data
    assert data["current_phase_progress"]["start_pose"]["x"] == 0.0
    assert data["current_phase_progress"]["latest_pose"]["x"] == 1.0
    assert data["current_phase_progress"]["cumulative_tool_facts"]["move_body_duration_success_sum"] == 0.5
    assert data["current_phase_progress"]["phase_local_measurements"]["forward_displacement_m"] == 1.0
    assert "coordinate_frames" in data["current_phase_progress"]
    assert data["current_phase_key"] == "1:orient_right"


def test_mission_state_uses_phase_index_to_separate_duplicate_phase_names():
    plan = SemanticPlan(
        intent="movement",
        goal=SemanticGoal(SemanticTarget("move forward twice", {}), "hover_after_movement"),
        phases=[
            SemanticPhase(1, 2, "move_forward", "first forward segment"),
            SemanticPhase(2, 2, "move_forward", "second forward segment"),
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
    assert state.current_phase_execution_summary["start_pose"]["x"] == 10.0


def test_phase_local_projection_uses_phase_start_yaw():
    plan = SemanticPlan(
        intent="movement",
        goal=SemanticGoal(SemanticTarget("move forward", {}), "hover_after_movement"),
        phases=[SemanticPhase(1, 1, "move_forward", "move forward")],
    )
    state = MissionState("run_test", "move forward", plan)
    state.record_phase_window_started()
    state.ensure_current_phase_progress({"pose": {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 0.0}}})
    state.update_current_phase_progress({"pose": {"position": {"x": 5.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 0.0}}})

    measurements = state.current_phase_progress["phase_local_measurements"]
    assert measurements["forward_displacement_m"] == 5.0
    assert measurements["right_displacement_m"] == 0.0


def test_phase_local_projection_handles_90_degree_yaw():
    plan = SemanticPlan(
        intent="movement",
        goal=SemanticGoal(SemanticTarget("move forward", {}), "hover_after_movement"),
        phases=[SemanticPhase(1, 1, "move_forward", "move forward")],
    )
    state = MissionState("run_test", "move forward", plan)
    state.record_phase_window_started()
    state.ensure_current_phase_progress(
        {"pose": {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 3.141592653589793 / 2}}}
    )
    state.update_current_phase_progress(
        {"pose": {"position": {"x": 0.0, "y": 5.0, "z": -2.0}, "euler_rad": {"yaw": 3.141592653589793 / 2}}}
    )

    measurements = state.current_phase_progress["phase_local_measurements"]
    assert measurements["forward_displacement_m"] == 5.0
    assert measurements["right_displacement_m"] == 0.0
