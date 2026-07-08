"""Dynamic target filtering for standalone STMR open-vocabulary masks.

GroundingDINO/SAM can repeatedly detect tiny far-away people or vehicles across
frames. Those detections are useful debug evidence, but they should not all be
accumulated into one semantic STMR layer. This module evaluates dynamic masks
with depth and image-quality signals, then selects one primary track cluster for
the final ``stmr_open_vocab_mask`` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from plugins.research.custom_map.context_modules.custom_map.projection import distance_confidence
from plugins.research.custom_map.context_modules.custom_map.projection_geometry import (
    DEPTH_MODE_OPTICAL_Z,
    camera_intrinsics,
    project_depth_pixel,
)
from plugins.research.custom_map.context_modules.custom_map.schemas import PosePoint
from plugins.research.custom_map.perception.semantic_palette import SEMANTIC_CLASSES, semantic_class_for_ade_label

from .sam_backend import SamMaskResult
from .tfidf_filter import TfidfMatch


DYNAMIC_SEMANTIC_LABELS = {"person", "vehicle", "two_wheeler_or_small_vehicle"}


@dataclass(frozen=True)
class DynamicTargetFilterConfig:
    """Thresholds for rejecting low-quality dynamic target detections."""

    enabled: bool = True
    min_detection_score: float = 0.45
    min_mask_pixels: int = 100
    max_depth_p95_m: float = 35.0
    small_edge_mask_pixels: int = 200
    edge_margin_px: int = 2
    primary_track_radius_m: float = 8.0


def build_dynamic_candidate(
    *,
    observation_id: str,
    detection_index: int,
    mask_result: SamMaskResult,
    match: TfidfMatch,
    image_size: tuple[int, int],
    depth: np.ndarray | None = None,
    pose: PosePoint | None = None,
    center_pose: PosePoint | None = None,
    fov_degrees: float = 90.0,
) -> dict[str, Any]:
    """Summarize one SAM mask before deciding whether it enters STMR."""

    semantic = semantic_class_for_ade_label(match.landmark)
    if semantic.semantic_id == SEMANTIC_CLASSES["unknown"].semantic_id:
        semantic = semantic_class_for_ade_label(match.phrase)
    width, height = image_size
    mask = _resize_mask(np.asarray(mask_result.mask, dtype=bool), width, height)
    ys, xs = np.where(mask)
    x0, y0, x1, y1 = [float(value) for value in mask_result.detection.box_xyxy]
    candidate: dict[str, Any] = {
        "candidate_id": f"{observation_id}:{detection_index}:{semantic.label}",
        "observation_id": observation_id,
        "detection_index": int(detection_index),
        "phrase": match.phrase,
        "landmark": match.landmark,
        "semantic_id": int(semantic.semantic_id),
        "label": semantic.label,
        "is_dynamic": semantic.label in DYNAMIC_SEMANTIC_LABELS,
        "score": round(float(match.score), 4),
        "similarity": round(float(match.similarity), 4),
        "iou_score": None if mask_result.iou_score is None else round(float(mask_result.iou_score), 4),
        "box_xyxy": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "bbox_area_ratio": round(max(0.0, (x1 - x0) * (y1 - y0)) / float(max(1, width * height)), 6),
        "edge_touching": _edge_touching((x0, y0, x1, y1), width, height, margin_px=2),
        "mask_pixel_count": int(len(xs)),
        "status": "pending",
        "reject_reasons": [],
    }
    if len(xs) == 0:
        candidate.update({"depth_valid_pixel_count": 0})
        return candidate

    if depth is not None:
        _add_depth_stats(candidate, mask, depth)
        if pose is not None and center_pose is not None:
            _add_local_centroid(candidate, mask, depth, pose, center_pose, fov_degrees=fov_degrees)
    return candidate


def apply_dynamic_target_policy(
    frame_items: list[dict[str, Any]],
    *,
    config: DynamicTargetFilterConfig | None = None,
) -> dict[str, Any]:
    """Mark candidates as static kept, rejected, primary track, or non-primary."""

    config = config or DynamicTargetFilterConfig()
    all_candidates = [candidate for item in frame_items for candidate in item.get("dynamic_target_candidates", [])]
    for candidate in all_candidates:
        _evaluate_candidate(candidate, config)
    dynamic_kept = [candidate for candidate in all_candidates if candidate.get("is_dynamic") and candidate.get("status") == "dynamic_kept"]
    primary_track = _select_primary_track(dynamic_kept, config.primary_track_radius_m)
    primary_ids = set(primary_track.get("candidate_ids", []))
    for candidate in dynamic_kept:
        if candidate["candidate_id"] in primary_ids:
            candidate["status"] = "primary_track"
        else:
            candidate["status"] = "kept_non_primary"
    for item in frame_items:
        candidates = item.get("dynamic_target_candidates", [])
        item["primary_dynamic_track"] = primary_track if primary_track.get("candidate_ids") else None
        item["rejected_masks"] = [candidate for candidate in candidates if str(candidate.get("status", "")).startswith("rejected")]
        item["kept_non_primary_masks"] = [candidate for candidate in candidates if candidate.get("status") == "kept_non_primary"]
    return {
        "enabled": bool(config.enabled),
        "candidate_count": len(all_candidates),
        "dynamic_candidate_count": sum(1 for item in all_candidates if item.get("is_dynamic")),
        "rejected_count": sum(1 for item in all_candidates if str(item.get("status", "")).startswith("rejected")),
        "primary_dynamic_track": primary_track,
        "status_counts": _status_counts(all_candidates),
    }


def should_write_candidate(candidate: dict[str, Any], *, dynamic_filter_enabled: bool = True) -> bool:
    """Return whether this candidate should be painted into the STMR mask."""

    if not dynamic_filter_enabled:
        return bool(candidate.get("tfidf_keep", True))
    if not candidate.get("is_dynamic"):
        return candidate.get("status") == "static_kept"
    return candidate.get("status") == "primary_track"


def _evaluate_candidate(candidate: dict[str, Any], config: DynamicTargetFilterConfig) -> None:
    if not config.enabled:
        candidate["status"] = "static_kept" if not candidate.get("is_dynamic") else "primary_track"
        return
    if not candidate.get("is_dynamic"):
        candidate["status"] = "static_kept"
        return
    reasons: list[str] = []
    if float(candidate.get("score", 0.0)) < config.min_detection_score:
        reasons.append("low_detection_score")
    if int(candidate.get("mask_pixel_count", 0)) < config.min_mask_pixels:
        reasons.append("tiny_mask")
    depth_p95 = candidate.get("depth_p95")
    if depth_p95 is not None and float(depth_p95) > config.max_depth_p95_m and int(candidate.get("mask_pixel_count", 0)) < config.small_edge_mask_pixels:
        reasons.append("far_tiny_dynamic_detection")
    if candidate.get("edge_touching") and int(candidate.get("mask_pixel_count", 0)) < config.small_edge_mask_pixels:
        reasons.append("small_edge_dynamic_detection")
    candidate["reject_reasons"] = reasons
    candidate["status"] = "rejected_" + "_".join(reasons) if reasons else "dynamic_kept"


def _select_primary_track(candidates: list[dict[str, Any]], radius_m: float) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    for candidate in candidates:
        point = _candidate_track_point(candidate)
        score = _candidate_track_score(candidate)
        candidate["track_score"] = round(score, 4)
        if point is None:
            clusters.append(_new_cluster(candidate, point, score))
            continue
        best_cluster = None
        best_distance = math.inf
        for cluster in clusters:
            if int(cluster.get("semantic_id", -9999)) != int(candidate.get("semantic_id", -8888)):
                continue
            center = cluster.get("center")
            if center is None:
                continue
            distance = math.dist(point, center)
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster
        if best_cluster is not None and best_distance <= radius_m:
            _add_to_cluster(best_cluster, candidate, point, score)
        else:
            clusters.append(_new_cluster(candidate, point, score))
    if not clusters:
        return {}
    clusters.sort(key=lambda item: (item["score_sum"], item["candidate_count"]), reverse=True)
    primary = clusters[0]
    return {
        "track_id": primary["track_id"],
        "label": primary["label"],
        "semantic_id": primary["semantic_id"],
        "candidate_ids": primary["candidate_ids"],
        "observation_ids": sorted(primary["observation_ids"]),
        "candidate_count": primary["candidate_count"],
        "score_sum": round(primary["score_sum"], 4),
        "center_local_forward_m": None if primary.get("center") is None else round(float(primary["center"][0]), 3),
        "center_local_right_m": None if primary.get("center") is None else round(float(primary["center"][1]), 3),
    }


def _new_cluster(candidate: dict[str, Any], point: tuple[float, float] | None, score: float) -> dict[str, Any]:
    return {
        "track_id": f"dynamic_track_{candidate['label']}_{_safe_id(candidate['candidate_id'])}",
        "label": candidate["label"],
        "semantic_id": candidate["semantic_id"],
        "candidate_ids": [candidate["candidate_id"]],
        "observation_ids": {candidate["observation_id"]},
        "candidate_count": 1,
        "score_sum": score,
        "center": point,
    }


def _add_to_cluster(cluster: dict[str, Any], candidate: dict[str, Any], point: tuple[float, float], score: float) -> None:
    previous_count = int(cluster["candidate_count"])
    old_center = cluster["center"]
    if old_center is not None:
        cluster["center"] = (
            (float(old_center[0]) * previous_count + point[0]) / (previous_count + 1),
            (float(old_center[1]) * previous_count + point[1]) / (previous_count + 1),
        )
    cluster["candidate_ids"].append(candidate["candidate_id"])
    cluster["observation_ids"].add(candidate["observation_id"])
    cluster["candidate_count"] = previous_count + 1
    cluster["score_sum"] = round(float(cluster["score_sum"]) + score, 4)


def _candidate_track_point(candidate: dict[str, Any]) -> tuple[float, float] | None:
    forward = candidate.get("local_centroid_forward_m")
    right = candidate.get("local_centroid_right_m")
    if forward is None or right is None:
        return None
    return (float(forward), float(right))


def _candidate_track_score(candidate: dict[str, Any]) -> float:
    score = float(candidate.get("score", 0.0))
    size_factor = min(1.0, math.sqrt(max(1.0, float(candidate.get("mask_pixel_count", 1))) / 1000.0))
    confidence = candidate.get("distance_confidence")
    if confidence is None:
        depth_p50 = candidate.get("depth_p50")
        confidence = distance_confidence(float(depth_p50)) if depth_p50 is not None else 0.65
    return score * (0.5 + 0.5 * size_factor) * float(confidence)


def _add_depth_stats(candidate: dict[str, Any], mask: np.ndarray, depth: np.ndarray) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != mask.shape:
        return
    values = depth[mask]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        candidate["depth_valid_pixel_count"] = 0
        return
    candidate.update(
        {
            "depth_valid_pixel_count": int(values.size),
            "depth_min": round(float(np.min(values)), 3),
            "depth_p50": round(float(np.percentile(values, 50)), 3),
            "depth_p95": round(float(np.percentile(values, 95)), 3),
            "distance_confidence": round(distance_confidence(float(np.percentile(values, 50))), 4),
        }
    )


def _add_local_centroid(
    candidate: dict[str, Any],
    mask: np.ndarray,
    depth: np.ndarray,
    pose: PosePoint,
    center_pose: PosePoint,
    *,
    fov_degrees: float,
) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != mask.shape:
        return
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    stride = max(1, int(math.sqrt(len(xs) / 250.0)))
    totals = {"local_forward_m": 0.0, "local_right_m": 0.0, "local_up_m": 0.0}
    count = 0
    for pixel_v, pixel_u in zip(ys[::stride], xs[::stride]):
        raw_depth = float(depth[int(pixel_v), int(pixel_u)])
        if not math.isfinite(raw_depth) or raw_depth <= 0:
            continue
        point = project_depth_pixel(
            pixel_u=int(pixel_u),
            pixel_v=int(pixel_v),
            raw_depth_m=raw_depth,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            pose=pose,
            center_pose=center_pose,
            depth_mode=DEPTH_MODE_OPTICAL_Z,
        )
        totals["local_forward_m"] += float(point["local_forward_m"])
        totals["local_right_m"] += float(point["local_right_m"])
        totals["local_up_m"] += float(point["local_up_m"])
        count += 1
    if count <= 0:
        return
    candidate.update(
        {
            "local_centroid_forward_m": round(totals["local_forward_m"] / count, 3),
            "local_centroid_right_m": round(totals["local_right_m"] / count, 3),
            "local_centroid_up_m": round(totals["local_up_m"] / count, 3),
            "local_centroid_sample_count": count,
        }
    )


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    from PIL import Image

    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8) > 0


def _edge_touching(box: tuple[float, float, float, float], width: int, height: int, *, margin_px: int) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= margin_px or y0 <= margin_px or x1 >= width - margin_px or y1 >= height - margin_px


def _status_counts(candidates: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        status = str(candidate.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)[-48:]
