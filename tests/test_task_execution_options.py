"""Test task-level initialization and Agent observation configuration."""

from __future__ import annotations

import math

import pytest

from src.env_client import ObservationProfile
from src.runtime.configs import InitialPose, TaskExecutionOptions


def test_task_execution_options_keep_current_runtime_defaults():
    options = TaskExecutionOptions()

    assert options.observation_modalities == ("rgb", "depth", "pose")
    assert options.initial_pose is None
    assert options.to_dict() == {
        "observation_modalities": ["rgb", "depth", "pose"],
        "initial_pose": None,
    }


def test_task_execution_options_use_profile_modalities_and_canonical_pose():
    profile = ObservationProfile.from_dict(
        {
            "modalities": ["RGB", "pose"],
            "capture_sources": {"rgb": "scene"},
        }
    )
    pose = InitialPose(1, 2, 3, yaw_deg=90, ignore_collision=False)

    options = TaskExecutionOptions.from_observation_profile(
        profile,
        initial_pose=pose,
    )

    assert options.observation_modalities == ("rgb", "pose")
    assert options.initial_pose is pose
    assert pose.to_action_args() == {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "yaw_deg": 90.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "ignore_collision": False,
    }


@pytest.mark.parametrize(
    ("modalities", "message"),
    [
        ((), "non-empty"),
        (("rgb", ""), "non-empty"),
        (("rgb", "rgb", "pose"), "unique"),
        (("rgb", "depth"), "include pose"),
    ],
)
def test_task_execution_options_reject_invalid_modalities(modalities, message):
    with pytest.raises(ValueError, match=message):
        TaskExecutionOptions(observation_modalities=modalities)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, "not-a-number", True])
def test_initial_pose_rejects_non_finite_coordinates(value):
    with pytest.raises(ValueError, match="x must be a finite number"):
        InitialPose(value, 0.0, 0.0)


def test_initial_pose_rejects_non_boolean_collision_policy():
    with pytest.raises(ValueError, match="ignore_collision must be a boolean"):
        InitialPose(0.0, 0.0, 0.0, ignore_collision="yes")  # type: ignore[arg-type]
