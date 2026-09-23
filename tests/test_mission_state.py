"""Test current mission state transitions."""
import pytest
from src.agent.mission_state import MissionState
from src.uav import pose_measurements


def make_state():
    return MissionState("run", "Turn towards the lamp and stop")


def test_mission_state_serializes_resolved_observation_modalities():
    state = make_state()
    state.observation_modalities = ("rgb", "pose")

    assert state.to_dict()["observation_modalities"] == ["rgb", "pose"]



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



def test_mission_state_recharges_stationary_yaw_scan_segment():
    state = make_state()
    state.record_action("uav.turn_relative", {"yaw_deg": -30.0}, {"ok": True})
    state.record_action("uav.turn_relative", {"yaw_deg": -20.0}, {"ok": True})

    recharge = state.recharge_yaw_scan()

    assert recharge == {
        "previous_segment": {
            "offset_deg": -50.0,
            "extent_deg": 50.0,
            "turn_count": 2,
        },
        "recharge_count": 1,
    }
    assert state.yaw_scan_offset_deg == 0.0
    assert state.yaw_scan_extent_deg == 0.0
    assert state.yaw_scan_turn_count == 0
    assert state.yaw_scan_recharge_count == 1



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

@pytest.mark.parametrize("yaw,axis,delta", [(0, "x", 5), (90, "y", 5), (180, "x", -5), (-90, "y", -5)])
def test_instruction_origin_projection(yaw, axis, delta):
    state = make_state()
    before = {"x": 0, "y": 0, "z": 0, "yaw_deg": yaw}
    after = {**before, axis: delta}
    state.record_observation({"observation_id": "before", "pose": before})
    state.begin_physical_window()
    state.finish_physical_window({"observation_id": "after", "pose": after}, pose_measurements(before, after))
    assert state.motion_summary["measurements"]["forward_m"] == 5
    assert state.motion_summary["last_window"]["forward_m"] == 5
    state.record_observation({"observation_id": "refresh", "pose": after})
    assert state.motion_summary["last_window"]["forward_m"] == 5
    assert state.before_last_action_window_id == "before"


def test_guard_and_mutable_fields_are_instruction_scoped():
    one, two = make_state(), make_state()
    assert one.max_physical_windows == 300
    one.loaded_knowledge.append({"name": "guide"})
    one.record_error("test", "error")
    assert two.loaded_knowledge == [] and two.errors == []
    assert two.task_plan is None
