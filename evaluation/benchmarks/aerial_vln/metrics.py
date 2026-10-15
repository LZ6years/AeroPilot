"""Official-semantics AerialVLN metrics for native Runtime actions."""

from __future__ import annotations

import math
from typing import Any, Sequence

from evaluation.benchmarks.aerial_vln.adapter import (
    AerialVlnCoordinateAdapter,
    AerialVlnEvaluationPayload,
)
from evaluation.benchmarks.core import (
    BenchmarkExecutionOutcome,
    EpisodeTrace,
)


SUCCESS_DISTANCE = 20.0


class AerialVlnMetricSuite:
    """Calculate official metric definitions over an action-aligned trace."""

    def __init__(self) -> None:
        self.coordinate_adapter = AerialVlnCoordinateAdapter()

    def evaluate(
        self,
        payload: AerialVlnEvaluationPayload | None,
        trace: EpisodeTrace,
        outcome: BenchmarkExecutionOutcome,
    ) -> dict[str, Any]:
        if payload is None:
            raise ValueError("AerialVLN evaluation payload is required")
        trajectory = self.coordinate_adapter.trace_to_ned(trace)
        reference = [
            tuple(float(value) for value in row[:3])
            for row in payload.reference_path_ned
        ]
        unique_trajectory = remove_consecutive_duplicates(trajectory)
        final = trajectory[-1] if trajectory else None
        distance = (
            xy_distance(final, payload.goal_position_ned)
            if final is not None
            else None
        )
        completed_normally = (
            outcome.completed
            and not outcome.stopped
            and outcome.error is None
        )
        success = bool(
            completed_normally
            and distance is not None
            and distance <= SUCCESS_DISTANCE
        )
        oracle_success = any(
            xy_distance(point, payload.goal_position_ned)
            <= SUCCESS_DISTANCE
            for point in trajectory
        )
        ndtw = normalized_dtw(
            unique_trajectory, reference, SUCCESS_DISTANCE
        )
        uav_steps = trace.action_count
        virtual_stop = 1 if completed_normally else 0
        return {
            "distance_to_goal": distance,
            "success": success,
            # Supplemental horizontal-distance metric; official SR/SDTW stay unchanged.
            "success_25m": bool(completed_normally and distance is not None and distance < 25.0),
            "marginal_20_25m": bool(completed_normally and distance is not None and 20.0 < distance < 25.0),
            "oracle_success": oracle_success,
            "ndtw": ndtw,
            "sdtw": float(success) * ndtw,
            "path_length": path_length(trajectory),
            "steps_taken": uav_steps + virtual_stop,
            "uav_action_steps": uav_steps,
            "virtual_stop_step": virtual_stop,
            "trajectory_ned": [list(point) for point in trajectory],
            "reference_path_ned": [list(point) for point in reference],
            "success_distance": SUCCESS_DISTANCE,
        }


def xy_distance(
    first: Sequence[float], second: Sequence[float]
) -> float:
    return math.hypot(
        float(first[0]) - float(second[0]),
        float(first[1]) - float(second[1]),
    )


def xyz_distance(
    first: Sequence[float], second: Sequence[float]
) -> float:
    return math.sqrt(
        sum(
            (float(first[index]) - float(second[index])) ** 2
            for index in range(3)
        )
    )


def remove_consecutive_duplicates(
    points: Sequence[Sequence[float]],
) -> list[tuple[float, float, float]]:
    unique: list[tuple[float, float, float]] = []
    for point in points:
        normalized = tuple(float(point[index]) for index in range(3))
        if not unique or normalized != unique[-1]:
            unique.append(normalized)
    return unique


def path_length(points: Sequence[Sequence[float]]) -> float:
    return sum(
        xyz_distance(points[index - 1], points[index])
        for index in range(1, len(points))
    )


def normalized_dtw(
    trajectory: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    success_distance: float,
) -> float:
    if not trajectory or not reference:
        return 0.0
    rows = len(trajectory)
    cols = len(reference)
    previous = [math.inf] * (cols + 1)
    previous[0] = 0.0
    for row in range(1, rows + 1):
        current = [math.inf] * (cols + 1)
        for col in range(1, cols + 1):
            cost = xyz_distance(
                trajectory[row - 1], reference[col - 1]
            )
            current[col] = cost + min(
                current[col - 1],
                previous[col],
                previous[col - 1],
            )
        previous = current
    dtw = previous[cols]
    return math.exp(
        -dtw / (len(reference) * float(success_distance))
    )
