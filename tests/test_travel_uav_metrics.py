"""TravelUAV metric semantics over native-action traces."""

from __future__ import annotations

import pytest

from evaluation.benchmarks.core import (
    BenchmarkExecutionOutcome,
    EpisodeTrace,
    EpisodeTraceSample,
)
from evaluation.benchmarks.travel_uav.adapter import TravelUavEvaluationPayload
from evaluation.benchmarks.travel_uav.metrics import TravelUavMetricSuite


def test_success_requires_normal_completion_and_reports_official_spl():
    metrics = evaluate([(0, 0, 0), (80, 0, 0)], completed=True)
    assert metrics["navigation_error"] == 0
    assert metrics["success"] is True
    assert metrics["oracle_success"] is True
    assert metrics["effective_reference_path_length"] == 80
    assert metrics["spl"] == pytest.approx(1.0)


def test_abnormal_stop_near_goal_is_oracle_only():
    metrics = evaluate([(80, 0, 0)], completed=False, stopped=True)
    assert metrics["success"] is False
    assert metrics["oracle_success"] is True
    assert metrics["spl"] == 0


def test_xyz_navigation_error_and_repeated_positions():
    metrics = evaluate([(0, 0, 0), (0, 0, 0), (80, 0, 10)], completed=True)
    assert metrics["navigation_error"] == pytest.approx(10)
    assert metrics["path_length"] == pytest.approx((80**2 + 10**2) ** 0.5)


def evaluate(points, *, completed, stopped=False):
    trace = EpisodeTrace(
        samples=tuple(
            EpisodeTraceSample(
                pose={"x": x, "y": y, "z": -z},
                source="task_initial" if index == 0 else "action_result",
                tool=None if index == 0 else "uav.move_body",
                action_id=None if index == 0 else str(index),
                timestamp=None,
                coordinate_frame="canonical_positive_up_m",
            )
            for index, (x, y, z) in enumerate(points)
        ),
        action_count=max(0, len(points) - 1),
        coordinate_frames=("canonical_positive_up_m",),
    )
    payload = TravelUavEvaluationPayload(
        target_position_ned=(80, 0, 0),
        reference_path_ned=((0, 0, 0), (100, 0, 0)),
        reference_path_length_m=100,
    )
    return TravelUavMetricSuite().evaluate(
        payload,
        trace,
        BenchmarkExecutionOutcome(completed=completed, stopped=stopped),
    )
