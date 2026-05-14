"""Test mission state transitions and history."""

from __future__ import annotations

from src.agent.mission_state import MissionState
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
    assert state.recent_actions[-1]["tool"] == "uav.takeoff"
    assert state.recent_observations[-1]["modalities"] == ["rgb", "depth", "pose"]


def test_mission_state_error_requires_hover():
    state = make_state()

    state.record_error("RuntimeError", "boom")

    assert state.hover_required
    assert state.errors[-1]["error_type"] == "RuntimeError"
