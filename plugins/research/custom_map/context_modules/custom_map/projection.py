"""Project depth-aligned semantic masks into a UAV-centered top-down grid.

This module implements the STMR-style geometry bridge for the research
custom_map plugin: mask pixels provide semantic ids, depth provides metric
distance, and UAV pose places those semantic points into the run-local world.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .mask_semantics import UNKNOWN_ID, SemanticLookup, build_semantic_lookup
from .projection_geometry import (
    DEPTH_COMPARISON_SAMPLE_LIMIT,
    DEPTH_MODE_OPTICAL_Z,
    DEPTH_MODE_RAY_DISTANCE,
    camera_intrinsics,
    finish_range_accumulator,
    new_range_accumulator,
    project_depth_pixel,
    summarize_frame_projection_drift,
    update_range_accumulator,
    world_to_centered_cell,
)
from .schemas import CustomMapRecords, FrameRecord, PosePoint, SemanticLayer
from ...perception.semantic_palette import semantic_class_for_ade_label

DEFAULT_FOV_DEGREES = 90.0
DEFAULT_STRIDE = 2
DEFAULT_MAX_DEPTH_M = 80.0
CELL_POOLING_STRATEGY = "priority_confidence_over_background_v1"
BACKGROUND_SEMANTIC_IDS = {20}
CONFIDENCE_WEIGHTED_SEMANTIC_IDS = {10, 11, 12}
PRIORITY_SEMANTIC_GROUPS = (
    {10, 11, 12},
    {50, 60, 70},
    {30, 40, 61},
)
PRIORITY_MIN_POINTS = 3
PRIORITY_MIN_RATIO = 0.08
DISTANCE_CONFIDENCE_NEAR_M = 8.0
DISTANCE_CONFIDENCE_FAR_M = 35.0
DISTANCE_CONFIDENCE_MIN = 0.15
DYNAMIC_OVERRIDE_MIN_WEIGHTED_SCORE = 3.0
DYNAMIC_OVERRIDE_MIN_AVG_CONFIDENCE = 0.45
FAR_POINT_DISTANCE_M = 35.0


def build_projected_semantic_layer(
    records: CustomMapRecords,
    center_pose: PosePoint | None,
    *,
    grid_size: int,
    meters_per_cell: float,
    stride: int = DEFAULT_STRIDE,
    max_depth_m: float = DEFAULT_MAX_DEPTH_M,
    max_debug_points: int | None = None,
) -> SemanticLayer | None:
    if center_pose is None:
        return None
    cell_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    cell_evidence: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    lookup = build_semantic_lookup(records)
    legend = dict(lookup.legend)
    warnings: list[str] = []
    projected_points = 0
    sampled_points: list[dict[str, Any]] = []
    camera_models: list[dict[str, Any]] = []
    source_observation_ids: set[str] = set()
    semantic_point_counts: Counter[int] = Counter()
    semantic_source_observations: dict[int, set[str]] = defaultdict(set)
    semantic_confidence_stats: dict[int, dict[str, Any]] = defaultdict(_new_confidence_stats)
    depth_mode_ranges = {
        DEPTH_MODE_OPTICAL_Z: new_range_accumulator(),
        DEPTH_MODE_RAY_DISTANCE: new_range_accumulator(),
    }
    depth_mode_comparison_samples: list[dict[str, Any]] = []
    frame_point_ranges: list[dict[str, Any]] = []
    for depth_frame, mask_frame, pose in _depth_mask_pose_sets(records):
        result = _project_frame_pair(
            records,
            depth_frame,
            mask_frame,
            pose,
            center_pose,
            grid_size=grid_size,
            meters_per_cell=meters_per_cell,
            stride=stride,
            max_depth_m=max_depth_m,
            max_debug_points=None if max_debug_points is None else max(0, max_debug_points - len(sampled_points)),
            lookup=lookup,
        )
        if result is None:
            continue
        for cell, counts in result["cell_counts"].items():
            cell_counts[cell].update(counts)
        for cell, evidence_by_semantic_id in result["cell_evidence"].items():
            _merge_cell_evidence(cell_evidence[cell], evidence_by_semantic_id)
        legend.update(result["legend"])
        warnings.extend(result["warnings"])
        projected_points += int(result["projected_points"])
        sampled_points.extend(result["sampled_points"])
        camera_models.append(result["camera_model"])
        for mode, range_data in result["depth_mode_ranges"].items():
            _merge_range_accumulator(depth_mode_ranges[mode], range_data)
        remaining_comparison_slots = DEPTH_COMPARISON_SAMPLE_LIMIT - len(depth_mode_comparison_samples)
        if remaining_comparison_slots > 0:
            depth_mode_comparison_samples.extend(result["depth_mode_comparison_samples"][:remaining_comparison_slots])
        frame_point_ranges.append(result["frame_point_ranges"])
        semantic_point_counts.update(result["semantic_point_counts"])
        _merge_semantic_confidence_stats(semantic_confidence_stats, result["semantic_confidence_stats"])
        for semantic_id, observation_ids in result["semantic_source_observations"].items():
            semantic_source_observations[semantic_id].update(observation_ids)
        if mask_frame.observation_id:
            source_observation_ids.add(mask_frame.observation_id)

    if projected_points <= 0:
        return None
    raw_label_examples = _raw_label_examples_by_semantic_id(records)
    for semantic_id, examples in raw_label_examples.items():
        if semantic_id in legend:
            legend[semantic_id]["raw_label_top_examples"] = examples
    matrix = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    priority_overrides: list[dict[str, Any]] = []
    low_confidence_candidates: list[dict[str, Any]] = []
    for (row, col), counts in cell_counts.items():
        selected_semantic_id, decision = _select_cell_semantic(
            counts,
            row=row,
            col=col,
            evidence=cell_evidence.get((row, col)),
        )
        matrix[row][col] = selected_semantic_id
        if decision is not None:
            if decision.get("decision") == "low_confidence_suppressed":
                low_confidence_candidates.append(decision)
            else:
                priority_overrides.append(decision)
    debug_points = sampled_points if max_debug_points is None else sampled_points[:max_debug_points]
    return SemanticLayer(
        matrix=matrix,
        legend=legend,
        status="depth_mask_projected_to_topdown",
        warnings=sorted(set(warnings)),
        projection_debug={
            "projection_status": "depth_mask_projected_to_topdown",
            "point_count": projected_points,
            "sampled_points": debug_points,
            "debug_point_limit": max_debug_points,
            "camera_model": camera_models[-1] if camera_models else {},
            "camera_models": camera_models,
            "source_observation_count": len(source_observation_ids),
            "source_observation_ids": sorted(source_observation_ids),
            "semantic_point_counts": {str(key): value for key, value in sorted(semantic_point_counts.items())},
            "semantic_confidence_summary": {
                str(key): _summarize_confidence_stats(value)
                for key, value in sorted(semantic_confidence_stats.items())
            },
            "semantic_point_ratios": {
                str(key): round(float(value) / float(projected_points), 6)
                for key, value in sorted(semantic_point_counts.items())
            },
            "semantic_raw_label_examples": {
                str(key): value for key, value in sorted(raw_label_examples.items())
            },
            "semantic_source_observations": {
                str(key): sorted(value) for key, value in sorted(semantic_source_observations.items())
            },
            "semantic_classes": _semantic_classes(
                legend,
                semantic_point_counts,
                semantic_source_observations,
                raw_label_examples,
                semantic_confidence_stats,
            ),
            "source_frames_by_observation_id": _source_frames_by_observation_id(records),
            "cell_pooling_strategy": CELL_POOLING_STRATEGY,
            "priority_overrides": priority_overrides,
            "priority_override_count": len(priority_overrides),
            "low_confidence_candidates": low_confidence_candidates,
            "low_confidence_suppressed_counts": _suppressed_counts(low_confidence_candidates),
            "distance_confidence_config": {
                "near_m": DISTANCE_CONFIDENCE_NEAR_M,
                "far_m": DISTANCE_CONFIDENCE_FAR_M,
                "min_confidence": DISTANCE_CONFIDENCE_MIN,
                "function": "smoothstep",
            },
            "depth_interpretation_used": DEPTH_MODE_OPTICAL_Z,
            "depth_interpretation_comparison": {
                "primary_mode": DEPTH_MODE_OPTICAL_Z,
                "comparison_mode": DEPTH_MODE_RAY_DISTANCE,
                "ranges": {
                    mode: finish_range_accumulator(range_data)
                    for mode, range_data in depth_mode_ranges.items()
                },
                "sample_limit": DEPTH_COMPARISON_SAMPLE_LIMIT,
                "samples": depth_mode_comparison_samples,
            },
            "frame_point_ranges": frame_point_ranges,
            "multi_frame_projection_drift": summarize_frame_projection_drift(
                frame_point_ranges,
                depth_mode=DEPTH_MODE_OPTICAL_Z,
            ),
            "stride": stride,
            "max_depth_m": max_depth_m,
        },
    )


def _depth_mask_pose_sets(records: CustomMapRecords) -> list[tuple[FrameRecord, FrameRecord, PosePoint]]:
    by_observation: dict[str, dict[str, FrameRecord]] = defaultdict(dict)
    for frame in records.frames:
        if not frame.observation_id:
            continue
        modality = frame.modality.lower()
        if modality in {"depth", "depth_perspective"}:
            by_observation[frame.observation_id]["depth"] = frame
        elif modality == "segmentation":
            by_observation[frame.observation_id]["segmentation"] = frame
        elif modality in {"mask", "segmentation", "object_mask"}:
            by_observation[frame.observation_id]["mask"] = frame
    pairs = []
    for observation_id, frames in by_observation.items():
        depth = frames.get("depth")
        mask = frames.get("segmentation") or frames.get("mask")
        pose = records.poses_by_observation_id.get(observation_id)
        if depth is not None and mask is not None and pose is not None:
            pairs.append((depth, mask, pose))
    return pairs


def _project_frame_pair(
    records: CustomMapRecords,
    depth_frame: FrameRecord,
    mask_frame: FrameRecord,
    pose: PosePoint,
    center_pose: PosePoint,
    *,
    grid_size: int,
    meters_per_cell: float,
    stride: int,
    max_depth_m: float,
    max_debug_points: int | None,
    lookup: SemanticLookup,
) -> dict[str, Any] | None:
    depth = _read_depth(depth_frame.path)
    if depth is None:
        return None
    mask = _read_mask(mask_frame.path, target_shape=depth.shape)
    if mask is None:
        return None
    height, width = depth.shape
    fov_degrees = _fov_degrees(depth_frame, mask_frame)
    fx, fy, cx, cy = camera_intrinsics(width, height, fov_degrees)
    cell_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    cell_evidence: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    semantic_point_counts: Counter[int] = Counter()
    semantic_source_observations: dict[int, set[str]] = defaultdict(set)
    semantic_confidence_stats: dict[int, dict[str, Any]] = defaultdict(_new_confidence_stats)
    optical_range = new_range_accumulator()
    ray_range = new_range_accumulator()
    projected_points = 0
    sampled_points: list[dict[str, Any]] = []
    depth_mode_comparison_samples: list[dict[str, Any]] = []
    for v in range(0, height, max(1, int(stride))):
        for u in range(0, width, max(1, int(stride))):
            distance = float(depth[v, u])
            if not math.isfinite(distance) or distance <= 0.0 or distance > max_depth_m:
                continue
            color = tuple(int(channel) for channel in mask.getpixel((u, v)))
            semantic_id = lookup.semantic_id_for_color(color)
            if semantic_id == 0:
                continue
            optical_point = project_depth_pixel(
                pixel_u=u,
                pixel_v=v,
                raw_depth_m=distance,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                pose=pose,
                center_pose=center_pose,
                depth_mode=DEPTH_MODE_OPTICAL_Z,
            )
            ray_point = project_depth_pixel(
                pixel_u=u,
                pixel_v=v,
                raw_depth_m=distance,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                pose=pose,
                center_pose=center_pose,
                depth_mode=DEPTH_MODE_RAY_DISTANCE,
            )
            update_range_accumulator(optical_range, optical_point)
            update_range_accumulator(ray_range, ray_point)
            if len(depth_mode_comparison_samples) < DEPTH_COMPARISON_SAMPLE_LIMIT:
                depth_mode_comparison_samples.append(
                    {
                        "pixel_u": u,
                        "pixel_v": v,
                        "semantic_id": semantic_id,
                        "source_observation_id": mask_frame.observation_id,
                        DEPTH_MODE_OPTICAL_Z: _compact_projection_point(optical_point),
                        DEPTH_MODE_RAY_DISTANCE: _compact_projection_point(ray_point),
                    }
                )
            distance_m = float(optical_point["distance_m"])
            confidence = distance_confidence(distance_m)
            cell = world_to_centered_cell(
                float(optical_point["world_x"]),
                float(optical_point["world_y"]),
                center_pose,
                grid_size,
                meters_per_cell,
            )
            if cell is None:
                continue
            cell_counts[(cell["row"], cell["col"])][semantic_id] += 1
            _add_semantic_evidence(
                cell_evidence[(cell["row"], cell["col"])],
                semantic_id,
                confidence,
                distance_m,
                mask_frame.observation_id,
            )
            semantic_point_counts[semantic_id] += 1
            _add_confidence_stats(
                semantic_confidence_stats[semantic_id],
                confidence,
                distance_m,
                mask_frame.observation_id,
            )
            if mask_frame.observation_id:
                semantic_source_observations[semantic_id].add(mask_frame.observation_id)
            projected_points += 1
            if max_debug_points is None or len(sampled_points) < max_debug_points:
                debug_point = _sampled_projection_point(optical_point)
                debug_point.update(
                    {
                        "semantic_id": semantic_id,
                        "color": list(color),
                        "source_observation_id": mask_frame.observation_id,
                        "distance_confidence": round(confidence, 4),
                    }
                )
                sampled_points.append(debug_point)
    return {
        "cell_counts": cell_counts,
        "cell_evidence": cell_evidence,
        "legend": lookup.legend,
        "warnings": lookup.deduped_warnings(),
        "projected_points": projected_points,
        "sampled_points": sampled_points,
        "semantic_point_counts": semantic_point_counts,
        "semantic_confidence_stats": semantic_confidence_stats,
        "semantic_source_observations": semantic_source_observations,
        "depth_mode_ranges": {
            DEPTH_MODE_OPTICAL_Z: finish_range_accumulator(optical_range),
            DEPTH_MODE_RAY_DISTANCE: finish_range_accumulator(ray_range),
        },
        "depth_mode_comparison_samples": depth_mode_comparison_samples,
        "frame_point_ranges": {
            "observation_id": mask_frame.observation_id,
            "pose": pose.to_model_dict(),
            DEPTH_MODE_OPTICAL_Z: finish_range_accumulator(optical_range),
            DEPTH_MODE_RAY_DISTANCE: finish_range_accumulator(ray_range),
        },
        "camera_model": {
            "width": width,
            "height": height,
            "fov_degrees": fov_degrees,
            "fx": round(fx, 3),
            "fy": round(fy, 3),
            "cx": round(cx, 3),
            "cy": round(cy, 3),
        },
    }


def _select_cell_semantic(
    counts: Counter[int],
    *,
    row: int,
    col: int,
    evidence: dict[int, dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Pick a top-down cell label without letting broad background erase objects."""

    if not counts:
        return 0, None
    positive_counts = Counter({semantic_id: count for semantic_id, count in counts.items() if semantic_id > 0})
    if not positive_counts:
        if counts.get(UNKNOWN_ID, 0) > 0:
            return UNKNOWN_ID, None
        return counts.most_common(1)[0][0], None

    majority_semantic_id, majority_count = positive_counts.most_common(1)[0]
    total_points = sum(positive_counts.values())
    override_threshold = max(PRIORITY_MIN_POINTS, math.ceil(total_points * PRIORITY_MIN_RATIO))

    for group in PRIORITY_SEMANTIC_GROUPS:
        candidates = [(semantic_id, positive_counts[semantic_id]) for semantic_id in group if semantic_id in positive_counts]
        if not candidates:
            continue
        selected_semantic_id, selected_count = max(candidates, key=lambda item: (item[1], -item[0]))
        if selected_semantic_id == majority_semantic_id:
            return majority_semantic_id, None
        if majority_semantic_id not in BACKGROUND_SEMANTIC_IDS:
            continue
        if selected_count < override_threshold:
            continue
        confidence_summary = _summarize_cell_semantic_evidence(evidence, selected_semantic_id)
        if evidence is not None and selected_semantic_id in CONFIDENCE_WEIGHTED_SEMANTIC_IDS:
            weighted_score = float(confidence_summary.get("weighted_score", 0.0))
            avg_confidence = float(confidence_summary.get("avg_confidence", 0.0))
            if (
                weighted_score < DYNAMIC_OVERRIDE_MIN_WEIGHTED_SCORE
                or avg_confidence < DYNAMIC_OVERRIDE_MIN_AVG_CONFIDENCE
            ):
                return majority_semantic_id, {
                    "decision": "low_confidence_suppressed",
                    "row": row,
                    "col": col,
                    "suppressed_semantic_id": selected_semantic_id,
                    "suppressed_count": int(selected_count),
                    "majority_semantic_id": majority_semantic_id,
                    "majority_count": int(majority_count),
                    "total_points": int(total_points),
                    "required_weighted_score": DYNAMIC_OVERRIDE_MIN_WEIGHTED_SCORE,
                    "required_avg_confidence": DYNAMIC_OVERRIDE_MIN_AVG_CONFIDENCE,
                    "evidence": confidence_summary,
                    "counts": {str(key): int(value) for key, value in sorted(counts.items())},
                }
        return selected_semantic_id, {
            "decision": "priority_override",
            "row": row,
            "col": col,
            "selected_semantic_id": selected_semantic_id,
            "selected_count": int(selected_count),
            "selected_evidence": confidence_summary,
            "majority_semantic_id": majority_semantic_id,
            "majority_count": int(majority_count),
            "total_points": int(total_points),
            "threshold": int(override_threshold),
            "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        }

    return majority_semantic_id, None


def _sampled_projection_point(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth_mode": point["depth_mode"],
        "pixel_u": point["pixel_u"],
        "pixel_v": point["pixel_v"],
        "raw_depth_m": round(float(point["raw_depth_m"]), 3),
        "camera_x": round(float(point["camera_x"]), 3),
        "camera_y": round(float(point["camera_y"]), 3),
        "camera_z": round(float(point["camera_z"]), 3),
        "camera_forward_m": round(float(point["camera_forward_m"]), 3),
        "camera_right_m": round(float(point["camera_right_m"]), 3),
        "camera_up_m": round(float(point["camera_up_m"]), 3),
        "world_x": round(float(point["world_x"]), 3),
        "world_y": round(float(point["world_y"]), 3),
        "world_z": round(float(point["world_z"]), 3),
        "local_forward_m": round(float(point["local_forward_m"]), 3),
        "local_right_m": round(float(point["local_right_m"]), 3),
        "local_up_m": round(float(point["local_up_m"]), 3),
        "distance_m": round(float(point["distance_m"]), 3),
    }


def _compact_projection_point(point: dict[str, Any]) -> dict[str, float]:
    return {
        "camera_x": round(float(point["camera_x"]), 3),
        "camera_y": round(float(point["camera_y"]), 3),
        "camera_z": round(float(point["camera_z"]), 3),
        "world_x": round(float(point["world_x"]), 3),
        "world_y": round(float(point["world_y"]), 3),
        "world_z": round(float(point["world_z"]), 3),
        "local_forward_m": round(float(point["local_forward_m"]), 3),
        "local_right_m": round(float(point["local_right_m"]), 3),
        "local_up_m": round(float(point["local_up_m"]), 3),
        "distance_m": round(float(point["distance_m"]), 3),
    }


def _merge_range_accumulator(destination: dict[str, Any], source_summary: dict[str, Any]) -> None:
    destination["count"] = int(destination.get("count", 0)) + int(source_summary.get("count", 0))
    for key, source_range in source_summary.items():
        if key == "count" or not isinstance(source_range, list) or len(source_range) < 2:
            continue
        if key not in destination:
            destination[key] = [math.inf, -math.inf]
        destination[key][0] = min(float(destination[key][0]), float(source_range[0]))
        destination[key][1] = max(float(destination[key][1]), float(source_range[1]))


def distance_confidence(
    distance_m: float,
    *,
    near_m: float = DISTANCE_CONFIDENCE_NEAR_M,
    far_m: float = DISTANCE_CONFIDENCE_FAR_M,
    min_confidence: float = DISTANCE_CONFIDENCE_MIN,
) -> float:
    """Smoothly down-weight semantic evidence as projected distance increases."""

    if distance_m <= near_m:
        return 1.0
    if distance_m >= far_m:
        return min_confidence
    t = (distance_m - near_m) / (far_m - near_m)
    smooth = 3.0 * t * t - 2.0 * t * t * t
    return 1.0 - (1.0 - min_confidence) * smooth


def _new_semantic_evidence() -> dict[str, Any]:
    return {
        "count": 0,
        "weighted_score": 0.0,
        "confidence_sum": 0.0,
        "distance_sum": 0.0,
        "max_confidence": 0.0,
        "far_count": 0,
        "source_observation_ids": set(),
    }


def _new_confidence_stats() -> dict[str, Any]:
    return _new_semantic_evidence()


def _add_semantic_evidence(
    evidence_by_semantic_id: dict[int, dict[str, Any]],
    semantic_id: int,
    confidence: float,
    distance_m: float,
    observation_id: str | None,
) -> None:
    evidence = evidence_by_semantic_id.setdefault(semantic_id, _new_semantic_evidence())
    _add_confidence_stats(evidence, confidence, distance_m, observation_id)


def _add_confidence_stats(
    stats: dict[str, Any],
    confidence: float,
    distance_m: float,
    observation_id: str | None,
) -> None:
    stats["count"] = int(stats.get("count", 0)) + 1
    stats["weighted_score"] = float(stats.get("weighted_score", 0.0)) + float(confidence)
    stats["confidence_sum"] = float(stats.get("confidence_sum", 0.0)) + float(confidence)
    stats["distance_sum"] = float(stats.get("distance_sum", 0.0)) + float(distance_m)
    stats["max_confidence"] = max(float(stats.get("max_confidence", 0.0)), float(confidence))
    if distance_m >= FAR_POINT_DISTANCE_M:
        stats["far_count"] = int(stats.get("far_count", 0)) + 1
    if observation_id:
        source_ids = stats.setdefault("source_observation_ids", set())
        if isinstance(source_ids, set):
            source_ids.add(observation_id)


def _merge_cell_evidence(
    destination: dict[int, dict[str, Any]],
    source: dict[int, dict[str, Any]],
) -> None:
    for semantic_id, evidence in source.items():
        target = destination.setdefault(semantic_id, _new_semantic_evidence())
        _merge_confidence_stat(target, evidence)


def _merge_semantic_confidence_stats(
    destination: dict[int, dict[str, Any]],
    source: dict[int, dict[str, Any]],
) -> None:
    for semantic_id, stats in source.items():
        _merge_confidence_stat(destination[semantic_id], stats)


def _merge_confidence_stat(destination: dict[str, Any], source: dict[str, Any]) -> None:
    destination["count"] = int(destination.get("count", 0)) + int(source.get("count", 0))
    destination["weighted_score"] = float(destination.get("weighted_score", 0.0)) + float(source.get("weighted_score", 0.0))
    destination["confidence_sum"] = float(destination.get("confidence_sum", 0.0)) + float(source.get("confidence_sum", 0.0))
    destination["distance_sum"] = float(destination.get("distance_sum", 0.0)) + float(source.get("distance_sum", 0.0))
    destination["max_confidence"] = max(float(destination.get("max_confidence", 0.0)), float(source.get("max_confidence", 0.0)))
    destination["far_count"] = int(destination.get("far_count", 0)) + int(source.get("far_count", 0))
    target_ids = destination.setdefault("source_observation_ids", set())
    source_ids = source.get("source_observation_ids", set())
    if isinstance(target_ids, set):
        if isinstance(source_ids, set):
            target_ids.update(source_ids)
        elif isinstance(source_ids, list):
            target_ids.update(str(item) for item in source_ids)


def _summarize_cell_semantic_evidence(
    evidence_by_semantic_id: dict[int, dict[str, Any]] | None,
    semantic_id: int,
) -> dict[str, Any]:
    if evidence_by_semantic_id is None:
        return {}
    return _summarize_confidence_stats(evidence_by_semantic_id.get(semantic_id, {}))


def _summarize_confidence_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats.get("count", 0))
    if count <= 0:
        return {
            "count": 0,
            "weighted_score": 0.0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "avg_distance_m": 0.0,
            "far_point_ratio": 0.0,
            "source_observation_ids": [],
        }
    source_ids = stats.get("source_observation_ids", set())
    if isinstance(source_ids, set):
        source_list = sorted(source_ids)
    elif isinstance(source_ids, list):
        source_list = sorted(str(item) for item in source_ids)
    else:
        source_list = []
    return {
        "count": count,
        "weighted_score": round(float(stats.get("weighted_score", 0.0)), 4),
        "avg_confidence": round(float(stats.get("confidence_sum", 0.0)) / float(count), 4),
        "max_confidence": round(float(stats.get("max_confidence", 0.0)), 4),
        "avg_distance_m": round(float(stats.get("distance_sum", 0.0)) / float(count), 3),
        "far_point_ratio": round(float(stats.get("far_count", 0)) / float(count), 6),
        "source_observation_ids": source_list,
    }


def _suppressed_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for candidate in candidates:
        semantic_id = candidate.get("suppressed_semantic_id")
        if semantic_id is not None:
            counts[int(semantic_id)] += 1
    return {str(key): value for key, value in sorted(counts.items())}


def _semantic_classes(
    legend: dict[int, dict[str, Any]],
    semantic_point_counts: Counter[int],
    semantic_source_observations: dict[int, set[str]],
    raw_label_examples: dict[int, list[dict[str, Any]]] | None = None,
    semantic_confidence_stats: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    classes = []
    raw_label_examples = raw_label_examples or {}
    semantic_confidence_stats = semantic_confidence_stats or {}
    for semantic_id, point_count in semantic_point_counts.most_common():
        item = dict(legend.get(semantic_id, {}))
        confidence_summary = _summarize_confidence_stats(semantic_confidence_stats.get(semantic_id, {}))
        classes.append(
            {
                "semantic_id": semantic_id,
                "label": item.get("label", f"unknown_region_{semantic_id}"),
                "category": item.get("category"),
                "color": item.get("color"),
                "source": item.get("source"),
                "object_name": item.get("object_name"),
                "confidence": item.get("confidence"),
                "point_count": int(point_count),
                "avg_distance_m": confidence_summary["avg_distance_m"],
                "avg_distance_confidence": confidence_summary["avg_confidence"],
                "max_distance_confidence": confidence_summary["max_confidence"],
                "far_point_ratio": confidence_summary["far_point_ratio"],
                "source_observation_ids": sorted(semantic_source_observations.get(semantic_id, set())),
                "raw_label_top_examples": raw_label_examples.get(semantic_id, []),
            }
        )
    return classes


def _raw_label_examples_by_semantic_id(records: CustomMapRecords, max_examples: int = 8) -> dict[int, list[dict[str, Any]]]:
    """Summarize closed-set raw labels behind each compressed custom_map class."""

    by_semantic_id: dict[int, Counter[str]] = defaultdict(Counter)
    for frame in records.frames:
        if frame.modality.lower() != "segmentation":
            continue
        stats = frame.metadata.get("segmentation_stats")
        if not isinstance(stats, dict):
            continue
        raw_counts = stats.get("ade_label_pixel_counts")
        if not isinstance(raw_counts, dict):
            continue
        for raw_label, count in raw_counts.items():
            try:
                label_count = int(count)
            except (TypeError, ValueError):
                continue
            semantic = semantic_class_for_ade_label(str(raw_label))
            by_semantic_id[semantic.semantic_id][str(raw_label)] += label_count
    return {
        semantic_id: [
            {"label": label, "pixel_count": int(count)}
            for label, count in counter.most_common(max_examples)
        ]
        for semantic_id, counter in sorted(by_semantic_id.items())
        if counter
    }


def _source_frames_by_observation_id(records: CustomMapRecords) -> dict[str, dict[str, dict[str, Any]]]:
    source_frames: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for frame in records.frames:
        if not frame.observation_id or frame.path is None:
            continue
        modality = frame.modality.lower()
        if modality not in {"rgb", "depth", "depth_perspective", "mask", "segmentation", "object_mask"}:
            continue
        slot = "depth" if modality in {"depth", "depth_perspective"} else "mask" if modality in {"mask", "segmentation", "object_mask"} else "rgb"
        if slot == "mask" and "mask" in source_frames[frame.observation_id] and modality != "segmentation":
            continue
        source_frames[frame.observation_id][slot] = {
            "modality": frame.modality,
            "artifact_path": str(frame.path),
            "mock": bool(frame.metadata.get("mock", False)),
        }
    return {key: dict(value) for key, value in sorted(source_frames.items())}


def _read_depth(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists() or path.suffix.lower() != ".npy":
        return None
    try:
        data = np.load(path)
    except (OSError, ValueError):
        return None
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        side = int(math.sqrt(array.size))
        if side * side != array.size:
            return None
        array = array.reshape((side, side))
    if array.ndim != 2:
        return None
    return array


def _read_mask(path: Path | None, target_shape: tuple[int, int]) -> Image.Image | None:
    if path is None or not path.exists():
        return None
    try:
        image = Image.open(path).convert("RGB")
    except OSError:
        return None
    target_size = (int(target_shape[1]), int(target_shape[0]))
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.NEAREST)
    return image


def _intrinsics(width: int, height: int, fov_degrees: float) -> tuple[float, float, float, float]:
    fov_rad = math.radians(float(fov_degrees))
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def _fov_degrees(*frames: FrameRecord) -> float:
    for frame in frames:
        for key in ("fov_degrees", "FOV_Degrees", "fov"):
            value = frame.metadata.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return DEFAULT_FOV_DEGREES


def _world_to_centered_cell(
    world_x: float,
    world_y: float,
    center_pose: PosePoint,
    grid_size: int,
    meters_per_cell: float,
) -> dict[str, int] | None:
    dx = world_x - center_pose.x
    dy = world_y - center_pose.y
    yaw_rad = math.radians(center_pose.yaw_deg or 0.0)
    forward_m = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
    right_m = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    center = grid_size // 2
    row = int(round(center - forward_m / meters_per_cell))
    col = int(round(center + right_m / meters_per_cell))
    if row < 0 or col < 0 or row >= grid_size or col >= grid_size:
        return None
    return {"row": row, "col": col}
