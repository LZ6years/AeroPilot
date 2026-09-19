"""TravelUAV NE, SR, OSR, SPL, and path-length metrics."""

from __future__ import annotations

import math
from typing import Any, Sequence

from evaluation.benchmarks.core import BenchmarkExecutionOutcome, EpisodeTrace


SUCCESS_DISTANCE_M = 20.0


class TravelUavMetricSuite:
    """Evaluate a canonical Runtime trace against NED evaluation truth."""

    def evaluate(self, payload: Any, trace: EpisodeTrace, outcome: Any) -> dict[str, Any]:
        if payload is None:
            raise ValueError("TravelUAV evaluation payload is required")
        target = payload.target_position_ned
        trajectory = [_canonical_pose_to_ned(pose) for pose in trace.poses]
        final = trajectory[-1] if trajectory else None
        ne = _distance(final, target) if final is not None else None
        normal_completion = bool(
            outcome.completed and not outcome.stopped and outcome.error is None
        )
        success = bool(
            normal_completion and ne is not None and ne <= SUCCESS_DISTANCE_M
        )
        oracle_success = any(
            _distance(point, target) <= SUCCESS_DISTANCE_M
            for point in trajectory
        )
        actual_length = _path_length(trajectory)
        # Official TravelUAV metric code subtracts its 20 m success radius from
        # the reference path before applying the standard SPL denominator.
        optimal_length = max(
            0.0, float(payload.reference_path_length_m) - SUCCESS_DISTANCE_M
        )
        if not success:
            spl = 0.0
        elif optimal_length == 0.0:
            spl = 1.0 if actual_length == 0.0 else 0.0
        else:
            spl = optimal_length / max(optimal_length, actual_length)
        return {
            "navigation_error": ne,
            "success": success,
            "oracle_success": oracle_success,
            "spl": spl,
            "path_length": actual_length,
            "reference_path_length": float(payload.reference_path_length_m),
            "effective_reference_path_length": optimal_length,
            "success_distance": SUCCESS_DISTANCE_M,
            "uav_action_steps": trace.action_count,
            "trajectory_ned": [list(point) for point in trajectory],
            "reference_path_ned": [list(point) for point in payload.reference_path_ned],
        }


def _canonical_pose_to_ned(pose: dict[str, Any]) -> tuple[float, float, float]:
    position = pose.get("position")
    nested = position if isinstance(position, dict) else {}
    return (
        float(pose.get("x", nested.get("x", 0.0))),
        float(pose.get("y", nested.get("y", 0.0))),
        -float(pose.get("z", nested.get("z", 0.0))),
    )


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(first[i]) - float(second[i])) ** 2 for i in range(3)))


def _path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(_distance(points[index - 1], points[index]) for index in range(1, len(points)))
