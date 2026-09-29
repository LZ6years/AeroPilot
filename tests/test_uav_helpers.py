"""Test shared UAV coordinate and command helpers."""

from __future__ import annotations

from src.uav.config import UavActionConfig
from src.uav.commands import MOVE_BODY_ARGUMENT_SEMANTICS
from src.uav.frames import SIGNED_YAW_SEMANTICS
from src.uav import (
    AIRSIM_NED_VERTICAL_SIGN,
    CANONICAL_VERTICAL_SIGN,
    UNREALCV_VERTICAL_SIGN,
    move_body_velocity_from_args,
    normalize_yaw_delta_deg,
    pose_delta,
    pose_measurements,
    pose_value,
    velocity_norm_mps,
    yaw_deg,
)


def test_model_facing_argument_semantics_match_canonical_motion_conventions():
    config = UavActionConfig(linear_velocity_mps=2.0, vertical_velocity_mps=1.5)

    assert "positive moves forward" in MOVE_BODY_ARGUMENT_SEMANTICS["vx"].lower()
    assert "negative moves backward" in MOVE_BODY_ARGUMENT_SEMANTICS["vx"].lower()
    assert "positive moves right" in MOVE_BODY_ARGUMENT_SEMANTICS["vy"].lower()
    assert "negative moves left" in MOVE_BODY_ARGUMENT_SEMANTICS["vy"].lower()
    assert "positive moves up" in MOVE_BODY_ARGUMENT_SEMANTICS["vz"].lower()
    assert "negative moves down" in MOVE_BODY_ARGUMENT_SEMANTICS["vz"].lower()
    assert move_body_velocity_from_args({"direction": "forward"}, config)[0] > 0
    assert move_body_velocity_from_args({"direction": "backward"}, config)[0] < 0
    assert move_body_velocity_from_args({"direction": "right"}, config)[1] > 0
    assert move_body_velocity_from_args({"direction": "left"}, config)[1] < 0
    assert "negative turns left" in SIGNED_YAW_SEMANTICS.lower()
    assert "positive turns right" in SIGNED_YAW_SEMANTICS.lower()


def test_pose_helpers_support_flat_nested_orientation_and_euler_shapes():
    assert pose_value({"x": 1.5}, "x") == 1.5
    assert pose_value({"position": {"y": 2.5}}, "y") == 2.5
    assert yaw_deg({"orientation": {"yaw_deg": 45.0}}) == 45.0
    assert round(yaw_deg({"euler_rad": {"yaw": 3.141592653589793 / 2}}), 3) == 90.0


def test_pose_delta_normalizes_yaw_wraparound():
    delta = pose_delta(
        {"euler_rad": {"yaw": 159.0 * 3.141592653589793 / 180.0}},
        {"euler_rad": {"yaw": -172.0 * 3.141592653589793 / 180.0}},
    )

    assert normalize_yaw_delta_deg(-331.0) == 29.0
    assert delta["dyaw_deg"] == 29.0


def test_pose_measurements_cover_cardinal_yaws():
    assert pose_measurements(
        {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "euler_rad": {"yaw": 0.0}},
        {"position": {"x": 5.0, "y": 0.0, "z": 0.0}, "euler_rad": {"yaw": 0.0}},
    )["forward_displacement_m"] == 5.0
    assert pose_measurements(
        {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "euler_rad": {"yaw": 3.141592653589793 / 2}},
        {"position": {"x": 0.0, "y": 5.0, "z": 0.0}, "euler_rad": {"yaw": 3.141592653589793 / 2}},
    )["forward_displacement_m"] == 5.0
    assert pose_measurements(
        {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "euler_rad": {"yaw": 3.141592653589793}},
        {"position": {"x": -5.0, "y": 0.0, "z": 0.0}, "euler_rad": {"yaw": 3.141592653589793}},
    )["forward_displacement_m"] == 5.0
    assert pose_measurements(
        {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "euler_rad": {"yaw": -3.141592653589793 / 2}},
        {"position": {"x": 0.0, "y": -5.0, "z": 0.0}, "euler_rad": {"yaw": -3.141592653589793 / 2}},
    )["forward_displacement_m"] == 5.0


def test_pose_measurements_use_canonical_positive_up_vertical():
    up = pose_measurements({"z": 0.0}, {"z": 3.0})
    down = pose_measurements({"z": 0.0}, {"z": -2.0})

    assert up["up_displacement_m"] == 3.0
    assert up["vertical_delta_m"] == 3.0
    assert down["up_displacement_m"] == -2.0
    assert down["vertical_delta_m"] == -2.0


def test_move_body_direction_vertical_sign_conventions():
    config = UavActionConfig(linear_velocity_mps=2.0, vertical_velocity_mps=1.5, move_duration_sec=3.0)

    assert move_body_velocity_from_args({"direction": "forward"}, config) == (2.0, 0.0, 0.0, 3.0)
    assert move_body_velocity_from_args({"direction": "up"}, config, vertical_sign=CANONICAL_VERTICAL_SIGN) == (
        0.0,
        0.0,
        1.5,
        3.0,
    )
    assert move_body_velocity_from_args({"direction": "up"}, config, vertical_sign=AIRSIM_NED_VERTICAL_SIGN) == (
        0.0,
        0.0,
        -1.5,
        3.0,
    )
    assert move_body_velocity_from_args({"direction": "up"}, config, vertical_sign=UNREALCV_VERTICAL_SIGN) == (
        0.0,
        0.0,
        1.5,
        3.0,
    )
    assert move_body_velocity_from_args(
        {"vx": 0.0, "vy": 0.0, "vz": -0.5, "duration": 2.0},
        config,
    ) == (0.0, 0.0, -0.5, 2.0)
    assert move_body_velocity_from_args(
        {"vx": 0.0, "vy": 0.0, "vz": -0.5, "duration": 2.0},
        config,
        vertical_sign=AIRSIM_NED_VERTICAL_SIGN,
    ) == (0.0, 0.0, 0.5, 2.0)


def test_velocity_norm_mps_reads_airsim_velocity_shape():
    assert round(velocity_norm_mps({"velocity": {"x": 3.0, "y": 4.0, "z": 12.0}}), 3) == 13.0
    assert velocity_norm_mps({"position": {"x": 0.0}}) is None
