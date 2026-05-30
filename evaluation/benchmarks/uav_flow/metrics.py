"""nDTW metrics for UAV-Flow benchmark trajectories without scipy dependency."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

CM_TO_M = 0.01
FAST_SAMPLE_CLASSES = {"Turn", "Move"}
POSITION_ZEROED_CLASSES = {"Turn", "Rotate"}


def metric_interpretation_for_class(class_name: str | None, *, sample_step: int | None = None) -> dict[str, Any]:
    """Describe the UAV-Flow metric mode used for a class."""

    normalized = str(class_name or "")
    position_zeroed = normalized in POSITION_ZEROED_CLASSES
    mode = "rotation_sequence" if position_zeroed else "position_and_rotation_sequence"
    summary = (
        "Turn/Rotate nDTW follows the original UAV-Flow metric: xyz position is zeroed before DTW, "
        "so the score reflects sampled rotation-cosine sequence alignment rather than planar translation."
        if position_zeroed
        else "nDTW follows the original UAV-Flow metric: sampled xyz position and rotation-cosine values are compared together."
    )
    return {
        "class_name": normalized or None,
        "mode": mode,
        "position_zeroed": position_zeroed,
        "sample_step": sample_step,
        "summary": summary,
    }


def evaluate_case_ndtw(
    trajectory_path: str | Path,
    reference_path_preprocessed: list[list[float]],
    *,
    class_name: str | None = None,
    default_step: int = 5,
) -> dict[str, Any]:
    """Compute UAV-Flow-style nDTW for one case."""

    trajectory_path = Path(trajectory_path)
    if not trajectory_path.exists():
        return {
            "status": "skipped",
            "reason": "trajectory_missing",
            "ndtw": None,
            "metric_interpretation": metric_interpretation_for_class(class_name),
        }
    if not reference_path_preprocessed:
        return {
            "status": "skipped",
            "reason": "reference_path_missing",
            "ndtw": None,
            "metric_interpretation": metric_interpretation_for_class(class_name),
        }
    step = 2 if class_name in FAST_SAMPLE_CLASSES else int(default_step)
    zero_pos = class_name in POSITION_ZEROED_CLASSES
    interpretation = metric_interpretation_for_class(class_name, sample_step=step)
    model_vecs = sampled_state6d_from_model_log(trajectory_path, step=step, zero_pos=zero_pos)
    gt_vecs = sampled_state6d_from_reference(reference_path_preprocessed, step=step, max_points=20, zero_pos=zero_pos)
    dtw = dtw_distance(gt_vecs, model_vecs)
    gt_len = path_length(gt_vecs)
    score = ndtw(dtw, gt_len)
    if score is None:
        return {
            "status": "skipped",
            "reason": "empty_or_zero_length_path",
            "ndtw": None,
            "dtw": dtw,
            "gt_length": gt_len,
            "sample_step": step,
            "zero_pos": zero_pos,
            "metric_interpretation": interpretation,
        }
    return {
        "status": "ok",
        "ndtw": score,
        "dtw": dtw,
        "gt_length": gt_len,
        "sample_step": step,
        "zero_pos": zero_pos,
        "metric_interpretation": interpretation,
        "model_points": len(model_vecs),
        "reference_points": len(gt_vecs),
    }


def summarize_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate nDTW by class and overall."""

    by_class: dict[str, list[float]] = {}
    ok_scores = []
    for result in case_results:
        metrics = result.get("ndtw", {}) if isinstance(result, dict) else {}
        score = metrics.get("ndtw") if isinstance(metrics, dict) else None
        class_name = str(result.get("class_name") or "unclassified")
        if isinstance(score, (int, float)):
            by_class.setdefault(class_name, []).append(float(score))
            ok_scores.append(float(score))
    return {
        "case_count": len(case_results),
        "evaluated_count": len(ok_scores),
        "overall_mean_ndtw": _mean(ok_scores),
        "by_class": {
            class_name: {
                "evaluated_count": len(scores),
                "mean_ndtw": _mean(scores),
            }
            for class_name, scores in sorted(by_class.items())
        },
    }


def sampled_state6d_from_model_log(path: str | Path, *, step: int = 5, zero_pos: bool = False) -> list[list[float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vectors = []
    for index, item in enumerate(data):
        if index % step != 0 or not isinstance(item, dict):
            continue
        state = item.get("state")
        if not isinstance(state, list) or len(state) < 2:
            continue
        pos = [0.0, 0.0, 0.0] if zero_pos else [float(v) * CM_TO_M for v in state[0][:3]]
        rot = [float(v) for v in state[1][:3]]
        vectors.append([*pos, *[math.cos(math.radians(v)) for v in rot]])
    return vectors


def sampled_state6d_from_reference(
    states: list[list[float]],
    *,
    step: int = 5,
    max_points: int = 20,
    zero_pos: bool = False,
) -> list[list[float]]:
    vectors = []
    for index, item in enumerate(states):
        if index % step != 0 or len(item) < 6:
            continue
        pos = [0.0, 0.0, 0.0] if zero_pos else [float(v) * CM_TO_M for v in item[:3]]
        rot = [float(v) for v in item[3:6]]
        vectors.append([*pos, *[math.cos(math.radians(v)) for v in rot]])
    return vectors[:max_points]


def dtw_distance(vecs1: list[list[float]], vecs2: list[list[float]]) -> float | None:
    if not vecs1 or not vecs2:
        return None
    n = len(vecs1)
    m = len(vecs2)
    dp = [[math.inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _euclidean(vecs1[i - 1], vecs2[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_euclidean(points[i], points[i - 1]) for i in range(1, len(points)))


def ndtw(dtw_dist: float | None, gt_len: float, *, eta: float = 1.0) -> float | None:
    if dtw_dist is None or gt_len == 0:
        return None
    return math.exp(-float(dtw_dist) / (float(eta) * float(gt_len)))


def _euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None
