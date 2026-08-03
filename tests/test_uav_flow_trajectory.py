"""Tests for UAV-Flow trajectory conversion."""

from __future__ import annotations

import json

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.trajectory import export_trajectory, model_pose_to_uav_flow_state, world_pose_to_model_pose, world_to_local


def case() -> UavFlowCase:
    return UavFlowCase(
        case_id="c",
        file_name="c.json",
        source_path="c.json",
        class_name="Move",
        instruction="Move forward",
        instruction_unified="Move forward",
        selected_instruction="Move forward",
        initial_pos=[100.0, 200.0, 50.0, 0.0, 90.0, 0.0],
        reference_path_preprocessed=[[0, 0, 0, 0, 0, 0]],
    )


def test_world_to_local_matches_uav_flow_formula():
    local = world_to_local([100.0, 300.0, 60.0], [100.0, 200.0, 50.0, 0.0, 90.0, 0.0])

    assert round(local[0], 6) == 100.0
    assert round(local[1], 6) == 0.0
    assert round(local[2], 6) == 10.0


def test_world_pose_to_model_pose_uses_meters_and_local_yaw():
    pose = world_pose_to_model_pose([100.0, 300.0, 150.0, 0.0, 120.0, 0.0], [100.0, 200.0, 50.0, 0.0, 90.0, 0.0])

    assert pose["x"] == 1.0
    assert pose["z"] == 1.0
    assert pose["yaw_deg"] == 30.0
    assert pose["coordinate_frame"] == "uav_flow_initial_local_canonical_m"


def test_world_pose_to_model_pose_uses_local_pitch_and_roll():
    pose = world_pose_to_model_pose(
        [100.0, 200.0, 50.0, 15.0, 90.0, -5.0],
        [100.0, 200.0, 50.0, 10.0, 90.0, -10.0],
    )

    assert pose["pitch_deg"] == 5.0
    assert pose["roll_deg"] == 5.0


def test_world_pose_to_model_pose_preserves_canonical_down_as_negative_z():
    pose = world_pose_to_model_pose([100.0, 200.0, -350.0, 0.0, 90.0, 0.0], [100.0, 200.0, 50.0, 0.0, 90.0, 0.0])

    assert pose["z"] == -4.0


def test_export_trajectory_writes_uav_flow_state_shape(tmp_path):
    output = tmp_path / "trajectory.json"
    meta = export_trajectory(case(), [{"x": 1, "y": 2, "z": 0.5, "yaw_deg": 15}], output)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data == [{"state": [[100.0, 200.0, 50.0], [0.0, 15.0, 0.0]]}]
    assert meta["unit"] == "cm"
    assert meta["point_count"] == 1


def test_model_pose_to_uav_flow_state_defaults_missing_rotation():
    state = model_pose_to_uav_flow_state({"x": 0.5, "y": -0.25, "z": 0.0})

    assert state["state"] == [[50.0, -25.0, 0.0], [0.0, 0.0, 0.0]]
