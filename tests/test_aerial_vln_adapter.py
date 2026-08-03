"""Coordinate and episode-contract tests for AerialVLN."""

from __future__ import annotations

import math

import pytest

from evaluation.benchmarks.aerial_vln.adapter import (
    AERIAL_VLN_PROTOCOL,
    AerialVlnCaseAdapter,
    AerialVlnCoordinateAdapter,
)
from evaluation.benchmarks.aerial_vln.cases import AerialVlnCase
from evaluation.benchmarks.core import (
    EpisodeTrace,
    EpisodeTraceSample,
    ned_to_canonical_position,
    quaternion_wxyz_to_euler_deg,
)
from src.env_client import ObservationProfile


def test_wxyz_quaternion_to_euler_and_ned_roundtrip():
    half = math.sqrt(0.5)
    yaw, pitch, roll = quaternion_wxyz_to_euler_deg(
        (half, 0.0, 0.0, half)
    )

    assert yaw == pytest.approx(90.0)
    assert pitch == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)
    assert ned_to_canonical_position((1.0, 2.0, -3.0)) == (
        1.0,
        2.0,
        3.0,
    )


def test_wxyz_conversion_matches_airsim_sdk_when_available():
    airsim = pytest.importorskip("airsim")
    half = math.sqrt(0.5)

    yaw, pitch, roll = quaternion_wxyz_to_euler_deg(
        (half, 0.0, 0.0, half)
    )
    sdk_pitch, sdk_roll, sdk_yaw = airsim.to_eularian_angles(
        airsim.Quaternionr(0.0, 0.0, half, half)
    )

    assert yaw == pytest.approx(math.degrees(sdk_yaw))
    assert pitch == pytest.approx(math.degrees(sdk_pitch))
    assert roll == pytest.approx(math.degrees(sdk_roll))


def test_case_adapter_builds_canonical_initial_pose_without_truth_leak():
    case = make_case()
    episode = AerialVlnCaseAdapter(
        scene_map={"1": "airvln_env_1"},
        observation_profile=ObservationProfile(),
    ).prepare_case(case)

    initial = episode.execution_options.initial_pose
    assert initial is not None
    assert (initial.x, initial.y, initial.z) == (1.0, 2.0, 3.0)
    assert initial.yaw_deg == pytest.approx(90.0)
    assert episode.protocol_variant == AERIAL_VLN_PROTOCOL
    manifest = episode.planner_manifest()
    assert "goal_position_ned" not in str(manifest)
    assert "reference_path_ned" not in str(manifest)
    assert "official_actions" not in str(manifest)


def test_unknown_scene_fails_before_runtime():
    with pytest.raises(ValueError, match="no scene_map entry"):
        AerialVlnCaseAdapter(
            scene_map={"2": "other"},
            observation_profile=ObservationProfile(),
        ).prepare_case(make_case())


def test_runtime_canonical_pose_converts_back_to_ned():
    trace = EpisodeTrace(
        samples=(
            EpisodeTraceSample(
                pose={
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "coordinate_frame": "canonical_unspecified",
                },
                source="task_initial",
                tool=None,
                action_id=None,
                timestamp=None,
                coordinate_frame="canonical_unspecified",
            ),
        ),
        action_count=0,
        coordinate_frames=("canonical_unspecified",),
    )

    assert AerialVlnCoordinateAdapter().trace_to_ned(trace) == [
        (1.0, 2.0, -3.0)
    ]


def make_case() -> AerialVlnCase:
    half = math.sqrt(0.5)
    return AerialVlnCase(
        episode_id="ep",
        trajectory_id="traj",
        scene_id="1",
        instruction="Fly",
        start_position=(1.0, 2.0, -3.0),
        start_rotation_wxyz=(half, 0.0, 0.0, half),
        goal_position=(10.0, 20.0, -5.0),
        reference_path=((1.0, 2.0, -3.0), (10.0, 20.0, -5.0)),
        official_actions=(1, 0),
        raw={},
    )
