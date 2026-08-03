"""Shared deterministic custom-map research probe helpers.

The former UAV-Flow custom EnvClient collector was removed with the benchmark
migration.  The reusable action script and aggregation helpers remain here
because custom-map research tests and future SimulationServer-based collectors
use the same contracts.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


ACTION_SCRIPT_CHOICES = ("short_debug_v1", "explore_20_v1")


def _action_script(
    name: str = "short_debug_v1",
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    if name == "short_debug_v1":
        return (
            (
                "turn_relative",
                "uav.turn_relative",
                {"yaw_deg": 15.0},
            ),
            (
                "move_body",
                "uav.move_body",
                {
                    "vx": 1.0,
                    "vy": 0.0,
                    "vz": 0.0,
                    "duration": 1.5,
                },
            ),
            (
                "turn_relative",
                "uav.turn_relative",
                {"yaw_deg": -20.0},
            ),
            (
                "move_body",
                "uav.move_body",
                {
                    "vx": 1.0,
                    "vy": 0.3,
                    "vz": 0.0,
                    "duration": 1.5,
                },
            ),
        )
    if name == "explore_20_v1":
        return (
            _turn(12.0),
            _move(0.9, 0.0, 0.0, 1.2),
            _move(0.0, 0.45, 0.0, 1.0),
            _move(0.75, -0.25, 0.0, 1.2),
            _turn(-18.0),
            _move(0.9, 0.0, 0.0, 1.2),
            _move(0.0, -0.45, 0.0, 1.0),
            _move(0.65, 0.25, 0.12, 1.1),
            _turn(20.0),
            _move(0.9, 0.0, 0.0, 1.2),
            _move(0.0, 0.5, 0.0, 1.0),
            _move(0.7, -0.3, -0.12, 1.1),
            _turn(-12.0),
            _move(1.0, 0.0, 0.0, 1.2),
            _move(0.0, -0.5, 0.0, 1.0),
            _move(0.65, 0.3, 0.0, 1.2),
            _turn(15.0),
            _move(0.85, 0.0, 0.0, 1.2),
            _move(0.0, 0.4, 0.0, 1.0),
            _move(0.55, -0.2, 0.0, 1.1),
        )
    raise ValueError(f"Unknown custom_map action script: {name}")


def _build_sweep_summary(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    point_counts: Counter[str] = Counter()
    pixel_counts: Counter[str] = Counter()
    matrix_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    totals = {
        "projection_point_count": 0,
        "sampled_point_count": 0,
        "action_count": 0,
        "observation_count": 0,
        "priority_override_count": 0,
    }
    for run in runs:
        by_class[str(run.get("class_name") or "unknown")] += 1
        for key in totals:
            totals[key] += int(run.get(key) or 0)
        point_counts.update(
            _counter_from_dict(
                run.get("semantic_point_counts_by_label")
            )
        )
        pixel_counts.update(
            _counter_from_dict(run.get("segmentation_pixel_counts"))
        )
        matrix_counts.update(
            _counter_from_dict(
                run.get("semantic_matrix_cell_counts_by_label")
            )
        )
        class_counts.update(
            str(label)
            for label in (
                run.get("semantic_point_counts_by_label") or {}
            )
        )
    return {
        "run_count": len(runs),
        "runs_by_class": dict(sorted(by_class.items())),
        "totals": totals,
        "semantic_point_counts_by_label": dict(sorted(point_counts.items())),
        "segmentation_pixel_counts": dict(sorted(pixel_counts.items())),
        "semantic_matrix_cell_counts_by_label": dict(
            sorted(matrix_counts.items())
        ),
        "semantic_class_run_presence": dict(sorted(class_counts.items())),
        "ratios": {
            "unknown_projection_ratio": _ratio(
                point_counts.get("unknown", 0),
                totals["projection_point_count"],
            ),
            "unknown_pixel_ratio": _ratio(
                pixel_counts.get("unknown", 0), sum(pixel_counts.values())
            ),
            "unknown_matrix_ratio": _ratio(
                matrix_counts.get("unknown", 0), sum(matrix_counts.values())
            ),
            "sampled_to_projected_ratio": _ratio(
                totals["sampled_point_count"],
                totals["projection_point_count"],
            ),
        },
        "runs": runs,
    }


def _turn(yaw_deg: float) -> tuple[str, str, dict[str, Any]]:
    return (
        "turn_relative",
        "uav.turn_relative",
        {"yaw_deg": yaw_deg},
    )


def _move(
    vx: float, vy: float, vz: float, duration: float
) -> tuple[str, str, dict[str, Any]]:
    return (
        "move_body",
        "uav.move_body",
        {"vx": vx, "vy": vy, "vz": vz, "duration": duration},
    )


def _counter_from_dict(value: Any) -> Counter[str]:
    if not isinstance(value, dict):
        return Counter()
    return Counter(
        {
            str(key): int(count or 0)
            for key, count in value.items()
        }
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)
