"""Golden semantic fixtures for AerialVLN metrics."""

from __future__ import annotations

import pytest

from evaluation.benchmarks.aerial_vln.adapter import (
    AerialVlnEvaluationPayload,
)
from evaluation.benchmarks.aerial_vln.metrics import (
    AerialVlnMetricSuite,
    normalized_dtw,
)
from evaluation.benchmarks.core import (
    BenchmarkExecutionOutcome,
    EpisodeTrace,
    EpisodeTraceSample,
)


def test_completed_near_goal_is_success_and_adds_virtual_stop():
    metrics = evaluate(
        [(0, 0, 0), (10, 0, 0)],
        outcome=BenchmarkExecutionOutcome(completed=True),
        action_count=1,
        goal=(10, 0, 0),
    )

    assert metrics["distance_to_goal"] == 0.0
    assert metrics["success"] is True
    assert metrics["oracle_success"] is True
    assert metrics["ndtw"] == pytest.approx(1.0)
    assert metrics["sdtw"] == pytest.approx(1.0)
    assert metrics["path_length"] == pytest.approx(10.0)
    assert metrics["uav_action_steps"] == 1
    assert metrics["steps_taken"] == 2


def test_near_goal_abnormal_stop_is_not_success_but_is_oracle_success():
    metrics = evaluate(
        [(10, 0, 0)],
        outcome=BenchmarkExecutionOutcome(
            stopped=True, stopped_reason="safety"
        ),
        action_count=0,
        goal=(10, 0, 0),
    )

    assert metrics["success"] is False
    assert metrics["oracle_success"] is True
    assert metrics["steps_taken"] == 0
    assert metrics["virtual_stop_step"] == 0


def test_oracle_only_success_uses_any_xy_sample():
    metrics = evaluate(
        [(10, 0, 100), (100, 0, 0)],
        outcome=BenchmarkExecutionOutcome(completed=True),
        action_count=1,
        goal=(0, 0, 0),
    )

    assert metrics["success"] is False
    assert metrics["oracle_success"] is True


def test_repeated_positions_do_not_change_ndtw_or_path_length():
    repeated = evaluate(
        [(0, 0, 0), (0, 0, 0), (10, 0, 0)],
        outcome=BenchmarkExecutionOutcome(completed=True),
        action_count=2,
        goal=(10, 0, 0),
    )
    unique = evaluate(
        [(0, 0, 0), (10, 0, 0)],
        outcome=BenchmarkExecutionOutcome(completed=True),
        action_count=1,
        goal=(10, 0, 0),
    )

    assert repeated["ndtw"] == pytest.approx(unique["ndtw"])
    assert repeated["path_length"] == pytest.approx(10.0)


def test_empty_action_completed_episode_has_only_virtual_stop():
    metrics = evaluate(
        [(0, 0, 0)],
        outcome=BenchmarkExecutionOutcome(completed=True),
        action_count=0,
        goal=(0, 0, 0),
    )

    assert metrics["uav_action_steps"] == 0
    assert metrics["steps_taken"] == 1


def test_ndtw_uses_xyz_not_xy_only():
    assert normalized_dtw(
        [(0, 0, 10)], [(0, 0, 0)], 20
    ) == pytest.approx(0.6065306597)


@pytest.mark.parametrize('distance,sr,sr25,margin', [
    (19.999, True, True, False), (20, True, True, False),
    (20.001, False, True, True), (24.999, False, True, True),
    (25, False, False, False), (25.001, False, False, False),
])
def test_supplemental_distance_boundaries(distance, sr, sr25, margin):
    metrics = evaluate([(distance, 0, 80)], outcome=BenchmarkExecutionOutcome(completed=True),
                       action_count=0, goal=(0, 0, 0))
    assert (metrics['success'], metrics['success_25m'], metrics['marginal_20_25m']) == (sr, sr25, margin)
    assert metrics['sdtw'] == float(sr) * metrics['ndtw']


@pytest.mark.parametrize('outcome', [BenchmarkExecutionOutcome(completed=False),
    BenchmarkExecutionOutcome(completed=True, stopped=True),
    BenchmarkExecutionOutcome(completed=True, error={'message': 'timeout'})])
def test_abnormal_end_never_counts_as_supplemental_success(outcome):
    metrics = evaluate([(22, 0, 0)], outcome=outcome, action_count=0, goal=(0, 0, 0))
    assert not metrics['success_25m'] and not metrics['marginal_20_25m']


def evaluate(points, *, outcome, action_count, goal):
    trace = EpisodeTrace(
        samples=tuple(
            EpisodeTraceSample(
                pose={
                    "x": x,
                    "y": y,
                    "z": -z,
                    "coordinate_frame": "canonical_positive_up_m",
                },
                source="task_initial" if index == 0 else "action_result",
                tool=None if index == 0 else "uav.move_body",
                action_id=None if index == 0 else f"act_{index}",
                timestamp=None,
                coordinate_frame="canonical_positive_up_m",
            )
            for index, (x, y, z) in enumerate(points)
        ),
        action_count=action_count,
        coordinate_frames=("canonical_positive_up_m",),
    )
    payload = AerialVlnEvaluationPayload(
        episode_id="ep",
        trajectory_id="traj",
        scene_id="1",
        goal_position_ned=goal,
        reference_path_ned=((0, 0, 0), (10, 0, 0)),
        official_actions=(1, 0),
    )
    return AerialVlnMetricSuite().evaluate(payload, trace, outcome)
