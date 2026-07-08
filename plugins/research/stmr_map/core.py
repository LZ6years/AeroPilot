"""Paper-style Semantic-Topo-Metric Representation construction.

The current custom_map module projects semantic points directly into a local
grid with engineering priors. This module keeps a separate STMR path that more
closely follows the paper: semantic point cloud -> voxel map -> top-height
column projection -> 20x20 metric matrix.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from plugins.research.custom_map.context_modules.custom_map.mask_semantics import (
    UNKNOWN_ID,
    VISITED_PATH_ID,
    SemanticLookup,
    build_semantic_lookup,
)
from plugins.research.custom_map.context_modules.custom_map.projection_geometry import (
    DEPTH_COMPARISON_SAMPLE_LIMIT,
    DEPTH_MODE_OPTICAL_Z,
    DEPTH_MODE_RAY_DISTANCE,
    camera_intrinsics,
    finish_range_accumulator,
    local_to_centered_cell,
    new_range_accumulator,
    project_depth_pixel,
    summarize_frame_projection_drift,
    update_range_accumulator,
    world_to_centered_cell,
    world_to_local,
)
from plugins.research.custom_map.context_modules.custom_map.projection import DEFAULT_FOV_DEGREES
from plugins.research.custom_map.context_modules.custom_map.schemas import CustomMapRecords, FrameRecord, PosePoint
from plugins.research.custom_map.context_modules.custom_map.trajectory import build_trajectory

STMR_GRID_SIZE = 20
STMR_METERS_PER_CELL = 5.0
STMR_LOCAL_EXTENT_M = STMR_GRID_SIZE * STMR_METERS_PER_CELL
DEFAULT_VOXEL_SIZE_M = 1.0
DEFAULT_STRIDE = 2
DEFAULT_MAX_DEPTH_M = 80.0


@dataclass(frozen=True)
class StmrMap:
    """A compact STMR artifact plus debug evidence for human inspection."""

    semantic_matrix: list[list[int]]
    semantic_legend: dict[int, dict[str, Any]]
    path_cells: list[dict[str, int]]
    current_cell: dict[str, int] | None
    center_pose: dict[str, Any]
    map_config: dict[str, Any]
    coordinate_frame: dict[str, Any]
    projection_debug: dict[str, Any]
    source_frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _VoxelEvidence:
    counts: Counter[int] = field(default_factory=Counter)
    max_z: float = -math.inf
    point_count: int = 0
    observation_ids: set[str] = field(default_factory=set)


def build_stmr_map(
    records: CustomMapRecords,
    *,
    subgoal_semantic_ids: set[int] | None = None,
    grid_size: int = STMR_GRID_SIZE,
    meters_per_cell: float = STMR_METERS_PER_CELL,
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
    stride: int = DEFAULT_STRIDE,
    max_depth_m: float = DEFAULT_MAX_DEPTH_M,
) -> StmrMap:
    """Build a paper-style STMR matrix from open-vocabulary mask frames."""

    trajectory = build_trajectory(records)
    center_pose = trajectory.path[-1] if trajectory.path else None
    legend = _base_legend()
    warnings = list(records.warnings)
    if center_pose is None:
        return StmrMap(
            semantic_matrix=_empty_matrix(grid_size),
            semantic_legend=legend,
            path_cells=[],
            current_cell=None,
            center_pose={},
            map_config=_map_config(grid_size, meters_per_cell, voxel_size_m, "pose_missing"),
            coordinate_frame=_coordinate_frame(None, grid_size, meters_per_cell, voxel_size_m),
            projection_debug=_empty_debug("pose_missing"),
            warnings=[*warnings, "stmr_pose_missing"],
        )

    voxels: dict[tuple[int, int, int], _VoxelEvidence] = defaultdict(_VoxelEvidence)
    source_frames: dict[str, dict[str, Any]] = {}
    point_count = 0
    source_observation_ids: set[str] = set()
    sampled_points: list[dict[str, Any]] = []
    depth_mode_ranges = {
        DEPTH_MODE_OPTICAL_Z: new_range_accumulator(),
        DEPTH_MODE_RAY_DISTANCE: new_range_accumulator(),
    }
    depth_mode_comparison_samples: list[dict[str, Any]] = []
    frame_point_ranges: list[dict[str, Any]] = []
    for depth_frame, mask_frame, pose in _depth_mask_pose_sets(records):
        lookup = build_semantic_lookup(records, mask_frame)
        frame_result = _project_frame_to_voxels(
            depth_frame,
            mask_frame,
            pose,
            center_pose,
            lookup,
            voxels,
            voxel_size_m=voxel_size_m,
            stride=stride,
            max_depth_m=max_depth_m,
        )
        point_count += int(frame_result["projected_points"])
        sampled_points.extend(frame_result["sampled_points"])
        for mode, range_data in frame_result["depth_mode_ranges"].items():
            _merge_range_accumulator(depth_mode_ranges[mode], range_data)
        remaining_comparison_slots = DEPTH_COMPARISON_SAMPLE_LIMIT - len(depth_mode_comparison_samples)
        if remaining_comparison_slots > 0:
            depth_mode_comparison_samples.extend(frame_result["depth_mode_comparison_samples"][:remaining_comparison_slots])
        frame_point_ranges.append(frame_result["frame_point_ranges"])
        if mask_frame.observation_id:
            source_observation_ids.add(mask_frame.observation_id)
            source_frames[mask_frame.observation_id] = _source_frame_bundle(records, mask_frame.observation_id)
        legend.update(lookup.legend)
        warnings.extend(lookup.deduped_warnings())

    voxel_labels = _voxel_labels(voxels)
    voxel_world_centers = _voxel_world_centers(
        voxel_labels,
        center_pose,
        grid_size=grid_size,
        meters_per_cell=meters_per_cell,
        voxel_size_m=voxel_size_m,
    )
    matrix, matrix_debug = _project_voxels_to_matrix(
        voxel_labels,
        center_pose,
        subgoal_semantic_ids=subgoal_semantic_ids or set(),
        grid_size=grid_size,
        meters_per_cell=meters_per_cell,
        voxel_size_m=voxel_size_m,
    )
    path_cells = _pose_cells(trajectory.path, center_pose, grid_size, meters_per_cell)
    _stamp_path(matrix, path_cells)
    return StmrMap(
        semantic_matrix=matrix,
        semantic_legend=legend,
        path_cells=path_cells,
        current_cell={"row": grid_size // 2, "col": grid_size // 2},
        center_pose=center_pose.to_model_dict(),
        map_config=_map_config(grid_size, meters_per_cell, voxel_size_m, "stmr_voxel_top_height"),
        coordinate_frame=_coordinate_frame(center_pose, grid_size, meters_per_cell, voxel_size_m),
        projection_debug={
            "projection_status": "stmr_voxel_top_height",
            "point_count": point_count,
            "sampled_points": sampled_points,
            "voxel_count": len(voxel_labels),
            "source_observation_count": len(source_observation_ids),
            "source_observation_ids": sorted(source_observation_ids),
            "voxel_stats": _voxel_stats(voxel_labels),
            "voxel_world_centers": voxel_world_centers,
            "top_height_projection": matrix_debug["top_height_projection"],
            "subgoal_priority_hits": matrix_debug["subgoal_priority_hits"],
            "alignment_warnings": matrix_debug["alignment_warnings"],
            "semantic_cell_counts": {
                str(key): value for key, value in sorted(_matrix_counts(matrix).items())
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
            "voxel_size_m": voxel_size_m,
        },
        source_frames=source_frames,
        warnings=sorted(set(warnings)),
    )


def _depth_mask_pose_sets(records: CustomMapRecords) -> list[tuple[FrameRecord, FrameRecord, PosePoint]]:
    by_observation: dict[str, dict[str, FrameRecord]] = defaultdict(dict)
    for frame in records.frames:
        if not frame.observation_id:
            continue
        modality = frame.modality.lower()
        if modality in {"depth", "depth_perspective"}:
            by_observation[frame.observation_id]["depth"] = frame
        elif modality == "stmr_open_vocab_mask":
            by_observation[frame.observation_id]["mask"] = frame
    pairs = []
    for observation_id, frames in by_observation.items():
        depth = frames.get("depth")
        mask = frames.get("mask")
        pose = records.poses_by_observation_id.get(observation_id)
        if depth is not None and mask is not None and pose is not None:
            pairs.append((depth, mask, pose))
    return pairs


def _project_frame_to_voxels(
    depth_frame: FrameRecord,
    mask_frame: FrameRecord,
    pose: PosePoint,
    center_pose: PosePoint,
    lookup: SemanticLookup,
    voxels: dict[tuple[int, int, int], _VoxelEvidence],
    *,
    voxel_size_m: float,
    stride: int,
    max_depth_m: float,
) -> dict[str, Any]:
    depth = _read_depth(depth_frame.path)
    mask = _read_mask(mask_frame.path, depth.shape if depth is not None else None)
    if depth is None or mask is None:
        return _empty_frame_projection_debug(mask_frame.observation_id, pose)
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics(width, height, _fov_degrees(depth_frame, mask_frame))
    projected = 0
    sampled_points: list[dict[str, Any]] = []
    optical_range = new_range_accumulator()
    ray_range = new_range_accumulator()
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
            world_x = float(optical_point["world_x"])
            world_y = float(optical_point["world_y"])
            world_z = float(optical_point["world_z"])
            key = (
                math.floor(world_x / voxel_size_m),
                math.floor(world_y / voxel_size_m),
                math.floor(world_z / voxel_size_m),
            )
            voxel = voxels[key]
            voxel.counts[semantic_id] += 1
            voxel.point_count += 1
            voxel.max_z = max(voxel.max_z, world_z)
            if mask_frame.observation_id:
                voxel.observation_ids.add(mask_frame.observation_id)
            projected += 1
            debug_point = _sampled_projection_point(optical_point)
            debug_point.update(
                {
                    "semantic_id": semantic_id,
                    "color": list(color),
                    "source_observation_id": mask_frame.observation_id,
                }
            )
            sampled_points.append(debug_point)
    return {
        "projected_points": projected,
        "sampled_points": sampled_points,
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
    }


def _empty_frame_projection_debug(observation_id: str | None, pose: PosePoint) -> dict[str, Any]:
    return {
        "projected_points": 0,
        "sampled_points": [],
        "depth_mode_ranges": {
            DEPTH_MODE_OPTICAL_Z: {"count": 0},
            DEPTH_MODE_RAY_DISTANCE: {"count": 0},
        },
        "depth_mode_comparison_samples": [],
        "frame_point_ranges": {
            "observation_id": observation_id,
            "pose": pose.to_model_dict(),
            DEPTH_MODE_OPTICAL_Z: {"count": 0},
            DEPTH_MODE_RAY_DISTANCE: {"count": 0},
        },
    }


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


def _voxel_labels(voxels: dict[tuple[int, int, int], _VoxelEvidence]) -> list[dict[str, Any]]:
    labels = []
    for key, evidence in voxels.items():
        semantic_id = evidence.counts.most_common(1)[0][0] if evidence.counts else UNKNOWN_ID
        labels.append(
            {
                "key": key,
                "semantic_id": semantic_id,
                "point_count": evidence.point_count,
                "max_z": evidence.max_z,
                "counts": dict(evidence.counts),
                "observation_ids": sorted(evidence.observation_ids),
            }
        )
    return labels


def _project_voxels_to_matrix(
    voxel_labels: list[dict[str, Any]],
    center_pose: PosePoint,
    *,
    subgoal_semantic_ids: set[int],
    grid_size: int,
    meters_per_cell: float,
    voxel_size_m: float,
) -> tuple[list[list[int]], dict[str, Any]]:
    columns: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for voxel in voxel_labels:
        vx, vy, _vz = voxel["key"]
        columns[(vx, vy)].append(voxel)

    cell_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    top_height_projection: list[dict[str, Any]] = []
    subgoal_priority_hits: list[dict[str, Any]] = []
    alignment_warnings: list[str] = []
    for (vx, vy), column in columns.items():
        selected = _select_column_voxel(column, subgoal_semantic_ids)
        center = _voxel_center(selected["key"], voxel_size_m)
        cell = _world_to_centered_cell(center["world_x"], center["world_y"], center_pose, grid_size, meters_per_cell)
        if cell is None:
            continue
        local = _world_to_local(center["world_x"], center["world_y"], center["world_z"], center_pose)
        check_cell = _local_to_centered_cell(local, grid_size, meters_per_cell)
        if check_cell != cell:
            alignment_warnings.append(
                f"voxel_cell_mismatch:{selected['key']} projected={cell} local_reprojected={check_cell}"
            )
        semantic_id = int(selected["semantic_id"])
        cell_counts[(cell["row"], cell["col"])][semantic_id] += 1
        record = {
            "row": cell["row"],
            "col": cell["col"],
            "semantic_id": semantic_id,
            "max_z": round(float(selected["max_z"]), 3),
            "voxel_key": list(selected["key"]),
            "voxel_center_world": {
                "world_x": round(center["world_x"], 3),
                "world_y": round(center["world_y"], 3),
                "world_z": round(center["world_z"], 3),
            },
            "voxel_center_local": {
                "local_forward_m": round(local["local_forward_m"], 3),
                "local_right_m": round(local["local_right_m"], 3),
                "local_up_m": round(local["local_up_m"], 3),
            },
        }
        top_height_projection.append(record)
        if selected.get("subgoal_priority"):
            subgoal_priority_hits.append(record)

    matrix = _empty_matrix(grid_size)
    for (row, col), counts in cell_counts.items():
        positive_counts = Counter({key: value for key, value in counts.items() if key > 0})
        matrix[row][col] = positive_counts.most_common(1)[0][0] if positive_counts else counts.most_common(1)[0][0]
    return matrix, {
        "top_height_projection": top_height_projection[:500],
        "subgoal_priority_hits": subgoal_priority_hits,
        "alignment_warnings": sorted(set(alignment_warnings))[:100],
    }


def _select_column_voxel(column: list[dict[str, Any]], subgoal_semantic_ids: set[int]) -> dict[str, Any]:
    matching = [item for item in column if int(item["semantic_id"]) in subgoal_semantic_ids]
    candidates = matching or column
    selected = max(candidates, key=lambda item: (float(item["max_z"]), int(item["point_count"])))
    selected = dict(selected)
    selected["subgoal_priority"] = bool(matching)
    return selected


def _pose_cells(
    poses: list[PosePoint],
    center_pose: PosePoint,
    grid_size: int,
    meters_per_cell: float,
) -> list[dict[str, int]]:
    cells = []
    seen = set()
    for pose in poses:
        cell = _world_to_centered_cell(pose.x, pose.y, center_pose, grid_size, meters_per_cell)
        if cell is None:
            continue
        key = (cell["row"], cell["col"])
        if key not in seen:
            seen.add(key)
            cells.append(cell)
    return cells


def _stamp_path(matrix: list[list[int]], path_cells: list[dict[str, int]]) -> None:
    for cell in path_cells:
        row, col = cell["row"], cell["col"]
        if 0 <= row < len(matrix) and matrix and 0 <= col < len(matrix[row]):
            matrix[row][col] = VISITED_PATH_ID


def _world_to_centered_cell(
    world_x: float,
    world_y: float,
    center_pose: PosePoint,
    grid_size: int,
    meters_per_cell: float,
) -> dict[str, int] | None:
    local = _world_to_local(world_x, world_y, None, center_pose)
    return _local_to_centered_cell(local, grid_size, meters_per_cell)


def _world_to_local(
    world_x: float,
    world_y: float,
    world_z: float | None,
    center_pose: PosePoint,
) -> dict[str, float]:
    """Convert UAV-Flow world coordinates into the current UAV-centered frame.

    Local +forward is the current UAV heading, +right is body-right in the
    horizontal plane, and +up is relative world z. This is the frame used by the
    20x20 heading-up matrix and by the debug 3D compare viewer.
    """

    dx = world_x - center_pose.x
    dy = world_y - center_pose.y
    yaw_rad = math.radians(center_pose.yaw_deg or 0.0)
    forward_m = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
    right_m = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    up_m = 0.0 if world_z is None else world_z - (center_pose.z or 0.0)
    return {
        "local_forward_m": forward_m,
        "local_right_m": right_m,
        "local_up_m": up_m,
    }


def _local_to_centered_cell(
    local: dict[str, float],
    grid_size: int,
    meters_per_cell: float,
) -> dict[str, int] | None:
    center = grid_size // 2
    row = int(round(center - local["local_forward_m"] / meters_per_cell))
    col = int(round(center + local["local_right_m"] / meters_per_cell))
    if row < 0 or col < 0 or row >= grid_size or col >= grid_size:
        return None
    return {"row": row, "col": col}


def _voxel_center(key: tuple[int, int, int] | list[int], voxel_size_m: float) -> dict[str, float]:
    vx, vy, vz = key
    return {
        "world_x": (float(vx) + 0.5) * voxel_size_m,
        "world_y": (float(vy) + 0.5) * voxel_size_m,
        "world_z": (float(vz) + 0.5) * voxel_size_m,
    }


def _voxel_world_centers(
    voxel_labels: list[dict[str, Any]],
    center_pose: PosePoint,
    *,
    grid_size: int,
    meters_per_cell: float,
    voxel_size_m: float,
) -> list[dict[str, Any]]:
    centers = []
    for voxel in voxel_labels:
        world = _voxel_center(voxel["key"], voxel_size_m)
        local = _world_to_local(world["world_x"], world["world_y"], world["world_z"], center_pose)
        cell = _local_to_centered_cell(local, grid_size, meters_per_cell)
        centers.append(
            {
                "voxel_key": list(voxel["key"]),
                "semantic_id": int(voxel["semantic_id"]),
                "point_count": int(voxel["point_count"]),
                "max_z": round(float(voxel["max_z"]), 3),
                "world_x": round(world["world_x"], 3),
                "world_y": round(world["world_y"], 3),
                "world_z": round(world["world_z"], 3),
                "local_forward_m": round(local["local_forward_m"], 3),
                "local_right_m": round(local["local_right_m"], 3),
                "local_up_m": round(local["local_up_m"], 3),
                "grid_row": None if cell is None else cell["row"],
                "grid_col": None if cell is None else cell["col"],
                "counts": {str(key): value for key, value in sorted(voxel.get("counts", {}).items())},
                "observation_ids": list(voxel.get("observation_ids", [])),
            }
        )
    return centers


def _source_frame_bundle(records: CustomMapRecords, observation_id: str) -> dict[str, Any]:
    bundle = {}
    for frame in records.frames:
        if frame.observation_id != observation_id or frame.path is None:
            continue
        modality = frame.modality.lower()
        if modality in {"rgb", "depth", "depth_perspective", "stmr_open_vocab_mask"}:
            bundle[modality] = {
                "artifact_path": str(frame.path),
                "metadata": dict(frame.metadata),
            }
    return bundle


def _read_depth(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists() or path.suffix.lower() != ".npy":
        return None
    try:
        array = np.asarray(np.load(path), dtype=np.float32)
    except (OSError, ValueError):
        return None
    return array if array.ndim == 2 else None


def _read_mask(path: Path | None, target_shape: tuple[int, int] | None) -> Image.Image | None:
    if path is None or not path.exists() or target_shape is None:
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


def _matrix_counts(matrix: list[list[int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in matrix:
        counts.update(row)
    return counts


def _voxel_stats(voxel_labels: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_counts: Counter[int] = Counter()
    for item in voxel_labels:
        semantic_counts[int(item["semantic_id"])] += 1
    return {
        "voxel_count": len(voxel_labels),
        "semantic_voxel_counts": {str(key): value for key, value in sorted(semantic_counts.items())},
    }


def _map_config(grid_size: int, meters_per_cell: float, voxel_size_m: float, status: str) -> dict[str, Any]:
    return {
        "grid_size": grid_size,
        "meters_per_cell": meters_per_cell,
        "local_extent_m": grid_size * meters_per_cell,
        "voxel_size_m": voxel_size_m,
        "orientation": "uav_centered_current_heading_up",
        "semantic_projection": status,
        "representation": "paper_style_stmr",
    }


def _coordinate_frame(
    center_pose: PosePoint | None,
    grid_size: int,
    meters_per_cell: float,
    voxel_size_m: float,
) -> dict[str, Any]:
    return {
        "center_pose": {} if center_pose is None else center_pose.to_model_dict(),
        "world_frame": "uav_flow_pose_xy_yaw",
        "local_frame": "uav_centered_current_heading_up",
        "camera_to_body": "forward=z_cam, right=x_cam, up=-y_cam",
        "world_projection": {
            "world_x": "pose.x + forward*cos(yaw) - right*sin(yaw)",
            "world_y": "pose.y + forward*sin(yaw) + right*cos(yaw)",
            "world_z": "pose.z + up",
        },
        "world_to_local": {
            "local_forward_m": "cos(center_yaw)*dx + sin(center_yaw)*dy",
            "local_right_m": "-sin(center_yaw)*dx + cos(center_yaw)*dy",
            "local_up_m": "world_z - center_pose.z",
            "dx": "world_x - center_pose.x",
            "dy": "world_y - center_pose.y",
        },
        "local_to_grid": {
            "row": "round(grid_center - local_forward_m / meters_per_cell)",
            "col": "round(grid_center + local_right_m / meters_per_cell)",
        },
        "grid_size": grid_size,
        "meters_per_cell": meters_per_cell,
        "voxel_size_m": voxel_size_m,
    }


def _empty_debug(status: str) -> dict[str, Any]:
    return {
        "projection_status": status,
        "point_count": 0,
        "voxel_count": 0,
        "source_observation_count": 0,
        "voxel_stats": {},
        "voxel_world_centers": [],
        "top_height_projection": [],
        "subgoal_priority_hits": [],
        "alignment_warnings": [],
    }


def _empty_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def _base_legend() -> dict[int, dict[str, Any]]:
    return {
        UNKNOWN_ID: {"label": "unknown", "source": "default", "color": [148, 163, 184]},
        VISITED_PATH_ID: {"label": "visited_path", "source": "trajectory", "color": [37, 99, 235]},
        0: {"label": "free", "source": "default", "color": [255, 255, 255]},
    }
