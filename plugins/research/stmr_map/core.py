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
from typing import Any, Iterable

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
    summarize_frame_projection_drift,
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
TRAJECTORY_POSITION_EPSILON_M = 0.05


@dataclass(frozen=True)
class StmrMap:
    """A compact STMR artifact plus debug evidence for human inspection."""

    semantic_matrix: list[list[int]]
    semantic_legend: dict[int, dict[str, Any]]
    path_cells: list[dict[str, int]]
    trajectory_trace: list[dict[str, Any]]
    trajectory_summary: dict[str, Any]
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


@dataclass(frozen=True)
class IncrementalStmrUpdate:
    """Serializable statistics for one idempotent incremental update."""

    processed_observation_ids: tuple[str, ...]
    duplicate_observation_ids: tuple[str, ...]
    pending_observation_ids: tuple[str, ...]
    new_projected_point_count: int
    new_voxel_count: int
    total_projected_point_count: int
    total_voxel_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_observation_ids": list(self.processed_observation_ids),
            "duplicate_observation_ids": list(self.duplicate_observation_ids),
            "pending_observation_ids": list(self.pending_observation_ids),
            "new_projected_point_count": self.new_projected_point_count,
            "new_voxel_count": self.new_voxel_count,
            "total_projected_point_count": self.total_projected_point_count,
            "total_voxel_count": self.total_voxel_count,
        }


class IncrementalStmrMapState:
    """Accumulate world-space STMR evidence without reprojecting old frames.

    The state is intentionally process-local. Raw RGB/depth/mask/pose files and
    JSONL memory remain the durable source of truth, so a new state can recover
    by importing the full record set once after a process restart.
    """

    def __init__(
        self,
        *,
        grid_size: int = STMR_GRID_SIZE,
        meters_per_cell: float = STMR_METERS_PER_CELL,
        voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
        stride: int = DEFAULT_STRIDE,
        max_depth_m: float = DEFAULT_MAX_DEPTH_M,
    ) -> None:
        self.grid_size = int(grid_size)
        self.meters_per_cell = float(meters_per_cell)
        self.voxel_size_m = float(voxel_size_m)
        self.stride = int(stride)
        self.max_depth_m = float(max_depth_m)
        self._voxels: dict[tuple[int, int, int], _VoxelEvidence] = defaultdict(_VoxelEvidence)
        self._processed_observation_ids: set[str] = set()
        self._source_frames: dict[str, dict[str, Any]] = {}
        self._legend = _base_legend()
        self._warnings: set[str] = set()
        self._point_count = 0
        self._sampled_points: list[dict[str, Any]] = []
        self._depth_mode_ranges = {
            DEPTH_MODE_OPTICAL_Z: new_range_accumulator(),
            DEPTH_MODE_RAY_DISTANCE: new_range_accumulator(),
        }
        self._depth_mode_comparison_samples: list[dict[str, Any]] = []
        self._frame_point_ranges: list[dict[str, Any]] = []
        self._trajectory_path: list[PosePoint] = []
        self._center_pose: PosePoint | None = None

    @property
    def processed_observation_ids(self) -> tuple[str, ...]:
        """Observation IDs already committed to the accumulated evidence."""

        return tuple(sorted(self._processed_observation_ids))

    @property
    def voxel_count(self) -> int:
        return len(self._voxels)

    @property
    def point_count(self) -> int:
        return self._point_count

    def update(
        self,
        records: CustomMapRecords,
        observation_ids: Iterable[str] | None = None,
        *,
        profiler: Any | None = None,
    ) -> IncrementalStmrUpdate:
        """Project and commit only complete observations not seen before.

        ``observation_ids`` is mainly useful for online callers and profiling.
        Omitting it imports every complete, unprocessed observation in record
        order. Incomplete observations stay pending and may be retried later.
        """

        trajectory = build_trajectory(records)
        self._trajectory_path = list(trajectory.path)
        self._center_pose = trajectory.path[-1] if trajectory.path else None
        self._warnings.update(records.warnings)

        requested = None if observation_ids is None else {str(item) for item in observation_ids}
        complete_pairs = _depth_mask_pose_sets(records)
        complete_ids = {
            str(mask_frame.observation_id)
            for _depth_frame, mask_frame, _pose in complete_pairs
            if mask_frame.observation_id
        }
        duplicate_ids = sorted(
            observation_id
            for observation_id in (requested or complete_ids)
            if observation_id in self._processed_observation_ids
        )
        pending_ids = sorted((requested or set()) - complete_ids - self._processed_observation_ids)
        processed_ids: list[str] = []
        new_point_count = 0
        new_voxel_count = 0

        for depth_frame, mask_frame, pose in complete_pairs:
            observation_id = str(mask_frame.observation_id or "")
            if not observation_id or observation_id in self._processed_observation_ids:
                continue
            if requested is not None and observation_id not in requested:
                continue

            lookup = build_semantic_lookup(records, mask_frame)
            frame_voxels: dict[tuple[int, int, int], _VoxelEvidence] = defaultdict(_VoxelEvidence)
            center_pose = self._center_pose or pose
            if profiler is None:
                frame_result = _project_frame_to_voxels(
                    depth_frame,
                    mask_frame,
                    pose,
                    center_pose,
                    lookup,
                    frame_voxels,
                    voxel_size_m=self.voxel_size_m,
                    stride=self.stride,
                    max_depth_m=self.max_depth_m,
                )
            else:
                with profiler.stage(
                    "stmr_incremental_projection",
                    metadata={"observation_id": observation_id},
                ) as stage:
                    frame_result = _project_frame_to_voxels(
                        depth_frame,
                        mask_frame,
                        pose,
                        center_pose,
                        lookup,
                        frame_voxels,
                        voxel_size_m=self.voxel_size_m,
                        stride=self.stride,
                        max_depth_m=self.max_depth_m,
                    )
                    stage.update(
                        projected_point_count=frame_result["projected_points"],
                        frame_voxel_count=len(frame_voxels),
                        frame_status=frame_result["frame_status"],
                    )

            if frame_result["frame_status"] != "ok":
                self._warnings.add(f"stmr_observation_unreadable:{observation_id}")
                pending_ids.append(observation_id)
                continue

            if profiler is None:
                added_voxels = self._commit_frame(
                    records,
                    mask_frame,
                    lookup,
                    frame_voxels,
                    frame_result,
                )
            else:
                with profiler.stage(
                    "stmr_incremental_merge",
                    metadata={"observation_id": observation_id},
                ) as stage:
                    added_voxels = self._commit_frame(
                        records,
                        mask_frame,
                        lookup,
                        frame_voxels,
                        frame_result,
                    )
                    stage.update(
                        new_voxel_count=added_voxels,
                        total_voxel_count=len(self._voxels),
                        total_projected_point_count=self._point_count,
                    )

            processed_ids.append(observation_id)
            new_point_count += int(frame_result["projected_points"])
            new_voxel_count += added_voxels

        return IncrementalStmrUpdate(
            processed_observation_ids=tuple(processed_ids),
            duplicate_observation_ids=tuple(duplicate_ids),
            pending_observation_ids=tuple(sorted(set(pending_ids))),
            new_projected_point_count=new_point_count,
            new_voxel_count=new_voxel_count,
            total_projected_point_count=self._point_count,
            total_voxel_count=len(self._voxels),
        )

    def snapshot(
        self,
        *,
        subgoal_semantic_ids: set[int] | None = None,
        debug_point_limit: int | None = None,
    ) -> StmrMap:
        """Build the current UAV-centered matrix without reading old frames."""

        center_pose = self._center_pose
        legend = dict(self._legend)
        warnings = sorted(self._warnings)
        if center_pose is None:
            return StmrMap(
                semantic_matrix=_empty_matrix(self.grid_size),
                semantic_legend=legend,
                path_cells=[],
                trajectory_trace=[],
                trajectory_summary=_empty_trajectory_summary(),
                current_cell=None,
                center_pose={},
                map_config=_map_config(
                    self.grid_size,
                    self.meters_per_cell,
                    self.voxel_size_m,
                    "pose_missing",
                ),
                coordinate_frame=_coordinate_frame(
                    None,
                    self.grid_size,
                    self.meters_per_cell,
                    self.voxel_size_m,
                ),
                projection_debug=_empty_debug("pose_missing"),
                warnings=[*warnings, "stmr_pose_missing"],
            )

        voxel_labels = _voxel_labels(self._voxels)
        voxel_world_centers = _voxel_world_centers(
            voxel_labels,
            center_pose,
            grid_size=self.grid_size,
            meters_per_cell=self.meters_per_cell,
            voxel_size_m=self.voxel_size_m,
        )
        matrix, matrix_debug = _project_voxels_to_matrix(
            voxel_labels,
            center_pose,
            subgoal_semantic_ids=subgoal_semantic_ids or set(),
            grid_size=self.grid_size,
            meters_per_cell=self.meters_per_cell,
            voxel_size_m=self.voxel_size_m,
        )
        path_cells = _pose_cells(
            self._trajectory_path,
            center_pose,
            self.grid_size,
            self.meters_per_cell,
        )
        trajectory_trace, trajectory_summary = _pose_trace(
            self._trajectory_path,
            center_pose,
            self.grid_size,
            self.meters_per_cell,
        )
        _stamp_path(matrix, path_cells)
        selected_points = _limit_debug_points(self._sampled_points, debug_point_limit)
        sampled_points = [_recenter_projection_point(item, center_pose) for item in selected_points]
        comparison_samples = [
            _recenter_comparison_sample(item, center_pose)
            for item in self._depth_mode_comparison_samples
        ]
        return StmrMap(
            semantic_matrix=matrix,
            semantic_legend=legend,
            path_cells=path_cells,
            trajectory_trace=trajectory_trace,
            trajectory_summary=trajectory_summary,
            current_cell={"row": self.grid_size // 2, "col": self.grid_size // 2},
            center_pose=center_pose.to_model_dict(),
            map_config=_map_config(
                self.grid_size,
                self.meters_per_cell,
                self.voxel_size_m,
                "stmr_voxel_top_height",
            ),
            coordinate_frame=_coordinate_frame(
                center_pose,
                self.grid_size,
                self.meters_per_cell,
                self.voxel_size_m,
            ),
            projection_debug={
                "projection_status": "stmr_voxel_top_height",
                "build_mode": "incremental",
                "point_count": self._point_count,
                "sampled_points": sampled_points,
                "sampled_points_total": len(self._sampled_points),
                "debug_points_truncated": len(sampled_points) < len(self._sampled_points),
                "voxel_count": len(voxel_labels),
                "source_observation_count": len(self._processed_observation_ids),
                "source_observation_ids": sorted(self._processed_observation_ids),
                "processed_observation_count": len(self._processed_observation_ids),
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
                        for mode, range_data in self._depth_mode_ranges.items()
                    },
                    "sample_limit": DEPTH_COMPARISON_SAMPLE_LIMIT,
                    "samples": comparison_samples,
                },
                "frame_point_ranges": list(self._frame_point_ranges),
                "multi_frame_projection_drift": summarize_frame_projection_drift(
                    self._frame_point_ranges,
                    depth_mode=DEPTH_MODE_OPTICAL_Z,
                ),
                "stride": self.stride,
                "max_depth_m": self.max_depth_m,
                "voxel_size_m": self.voxel_size_m,
            },
            source_frames=dict(self._source_frames),
            warnings=warnings,
        )

    def _commit_frame(
        self,
        records: CustomMapRecords,
        mask_frame: FrameRecord,
        lookup: SemanticLookup,
        frame_voxels: dict[tuple[int, int, int], _VoxelEvidence],
        frame_result: dict[str, Any],
    ) -> int:
        """Atomically merge one successfully projected observation."""

        added_voxels = 0
        for key, evidence in frame_voxels.items():
            if key not in self._voxels:
                added_voxels += 1
            _merge_voxel_evidence(self._voxels[key], evidence)
        self._point_count += int(frame_result["projected_points"])
        self._sampled_points.extend(frame_result["sampled_points"])
        for mode, range_data in frame_result["depth_mode_ranges"].items():
            _merge_range_accumulator(self._depth_mode_ranges[mode], range_data)
        remaining_slots = DEPTH_COMPARISON_SAMPLE_LIMIT - len(self._depth_mode_comparison_samples)
        if remaining_slots > 0:
            self._depth_mode_comparison_samples.extend(
                frame_result["depth_mode_comparison_samples"][:remaining_slots]
            )
        self._frame_point_ranges.append(frame_result["frame_point_ranges"])
        observation_id = str(mask_frame.observation_id or "")
        self._processed_observation_ids.add(observation_id)
        self._source_frames[observation_id] = _source_frame_bundle(records, observation_id)
        self._legend.update(lookup.legend)
        self._warnings.update(lookup.deduped_warnings())
        return added_voxels


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
    """Build a full map through the incremental engine for API compatibility."""

    state = IncrementalStmrMapState(
        grid_size=grid_size,
        meters_per_cell=meters_per_cell,
        voxel_size_m=voxel_size_m,
        stride=stride,
        max_depth_m=max_depth_m,
    )
    state.update(records)
    result = state.snapshot(subgoal_semantic_ids=subgoal_semantic_ids, debug_point_limit=None)
    result.projection_debug["build_mode"] = "batch_compatibility"
    return result


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
        return _empty_frame_projection_debug(mask_frame.observation_id, pose, status="unreadable")
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics(width, height, _fov_degrees(depth_frame, mask_frame))
    step = max(1, int(stride))
    v_axis = np.arange(0, height, step, dtype=np.int64)
    u_axis = np.arange(0, width, step, dtype=np.int64)
    pixel_u, pixel_v = np.meshgrid(u_axis, v_axis)
    distances = np.asarray(depth[np.ix_(v_axis, u_axis)], dtype=np.float64).reshape(-1)
    mask_array = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    colors = mask_array[v_axis[:, None], u_axis[None, :], :3].reshape(-1, 3)
    packed_colors = (
        colors[:, 0].astype(np.int64) << 16
        | colors[:, 1].astype(np.int64) << 8
        | colors[:, 2].astype(np.int64)
    )
    unique_colors, inverse_colors = np.unique(packed_colors, return_inverse=True)
    unique_semantics = np.asarray(
        [
            lookup.semantic_id_for_color(
                (
                    int((color >> 16) & 255),
                    int((color >> 8) & 255),
                    int(color & 255),
                )
            )
            for color in unique_colors
        ],
        dtype=np.int64,
    )
    semantic_ids = unique_semantics[inverse_colors]
    valid = np.isfinite(distances) & (distances > 0.0) & (distances <= max_depth_m) & (semantic_ids != 0)
    if not np.any(valid):
        return _empty_frame_projection_debug(mask_frame.observation_id, pose, status="ok")

    pixel_u = pixel_u.reshape(-1)[valid]
    pixel_v = pixel_v.reshape(-1)[valid]
    distances = distances[valid]
    colors = colors[valid]
    semantic_ids = semantic_ids[valid]
    camera_x = np.round((pixel_u - cx) * distances / fx, 4)
    camera_y = np.round((pixel_v - cy) * distances / fy, 4)
    camera_z = np.round(distances, 4)
    forward = camera_z
    right = camera_x
    up = -camera_y
    yaw_rad = math.radians(pose.yaw_deg or 0.0)
    world_x = np.round(pose.x + forward * math.cos(yaw_rad) - right * math.sin(yaw_rad), 4)
    world_y = np.round(pose.y + forward * math.sin(yaw_rad) + right * math.cos(yaw_rad), 4)
    world_z = np.round((pose.z or 0.0) + up, 4)
    local_forward, local_right, local_up = _vectorized_world_to_local(
        world_x,
        world_y,
        world_z,
        center_pose,
    )
    distance_m = np.round(np.sqrt(forward * forward + right * right + up * up), 4)
    optical_arrays = {
        "world_x": world_x,
        "world_y": world_y,
        "world_z": world_z,
        "local_forward_m": local_forward,
        "local_right_m": local_right,
        "local_up_m": local_up,
        "camera_x": camera_x,
        "camera_y": camera_y,
        "camera_z": camera_z,
        "raw_depth_m": distances,
    }

    ray_x = (pixel_u - cx) / fx
    ray_y = (pixel_v - cy) / fy
    ray_norm = np.sqrt(ray_x * ray_x + ray_y * ray_y + 1.0)
    ray_camera_x = np.round(distances * ray_x / ray_norm, 4)
    ray_camera_y = np.round(distances * ray_y / ray_norm, 4)
    ray_camera_z = np.round(distances / ray_norm, 4)
    ray_forward = ray_camera_z
    ray_right = ray_camera_x
    ray_up = -ray_camera_y
    ray_world_x = np.round(
        pose.x + ray_forward * math.cos(yaw_rad) - ray_right * math.sin(yaw_rad),
        4,
    )
    ray_world_y = np.round(
        pose.y + ray_forward * math.sin(yaw_rad) + ray_right * math.cos(yaw_rad),
        4,
    )
    ray_world_z = np.round((pose.z or 0.0) + ray_up, 4)
    ray_local_forward, ray_local_right, ray_local_up = _vectorized_world_to_local(
        ray_world_x,
        ray_world_y,
        ray_world_z,
        center_pose,
    )
    ray_arrays = {
        "world_x": ray_world_x,
        "world_y": ray_world_y,
        "world_z": ray_world_z,
        "local_forward_m": ray_local_forward,
        "local_right_m": ray_local_right,
        "local_up_m": ray_local_up,
        "camera_x": ray_camera_x,
        "camera_y": ray_camera_y,
        "camera_z": ray_camera_z,
        "raw_depth_m": distances,
    }

    voxel_keys = np.floor(
        np.column_stack((world_x, world_y, world_z)) / float(voxel_size_m)
    ).astype(np.int64)
    unique_voxel_keys, voxel_inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    voxel_point_counts = np.bincount(voxel_inverse, minlength=len(unique_voxel_keys))
    voxel_max_z = np.full(len(unique_voxel_keys), -math.inf, dtype=np.float64)
    np.maximum.at(voxel_max_z, voxel_inverse, world_z)
    for index, raw_key in enumerate(unique_voxel_keys):
        key = tuple(int(value) for value in raw_key)
        voxel = voxels[key]
        voxel.point_count += int(voxel_point_counts[index])
        voxel.max_z = max(voxel.max_z, float(voxel_max_z[index]))
        if mask_frame.observation_id:
            voxel.observation_ids.add(mask_frame.observation_id)
    semantic_groups, semantic_counts = np.unique(
        np.column_stack((voxel_inverse, semantic_ids)),
        axis=0,
        return_counts=True,
    )
    for (voxel_index, semantic_id), count in zip(semantic_groups, semantic_counts):
        key = tuple(int(value) for value in unique_voxel_keys[int(voxel_index)])
        voxels[key].counts[int(semantic_id)] += int(count)

    sampled_points = [
        {
            "depth_mode": DEPTH_MODE_OPTICAL_Z,
            "pixel_u": int(u),
            "pixel_v": int(v),
            "raw_depth_m": round(float(raw_depth), 3),
            "camera_x": round(float(cam_x), 3),
            "camera_y": round(float(cam_y), 3),
            "camera_z": round(float(cam_z), 3),
            "camera_forward_m": round(float(cam_z), 3),
            "camera_right_m": round(float(cam_x), 3),
            "camera_up_m": round(float(-cam_y), 3),
            "world_x": round(float(wx), 3),
            "world_y": round(float(wy), 3),
            "world_z": round(float(wz), 3),
            "local_forward_m": round(float(lf), 3),
            "local_right_m": round(float(lr), 3),
            "local_up_m": round(float(lu), 3),
            "distance_m": round(float(metric_distance), 3),
            "semantic_id": int(semantic_id),
            "color": [int(channel) for channel in color],
            "source_observation_id": mask_frame.observation_id,
        }
        for u, v, raw_depth, cam_x, cam_y, cam_z, wx, wy, wz, lf, lr, lu, metric_distance, semantic_id, color in zip(
            pixel_u,
            pixel_v,
            distances,
            camera_x,
            camera_y,
            camera_z,
            world_x,
            world_y,
            world_z,
            local_forward,
            local_right,
            local_up,
            distance_m,
            semantic_ids,
            colors,
        )
    ]
    comparison_count = min(DEPTH_COMPARISON_SAMPLE_LIMIT, len(sampled_points))
    depth_mode_comparison_samples = [
        {
            "pixel_u": int(pixel_u[index]),
            "pixel_v": int(pixel_v[index]),
            "semantic_id": int(semantic_ids[index]),
            "source_observation_id": mask_frame.observation_id,
            DEPTH_MODE_OPTICAL_Z: _compact_vectorized_point(
                optical_arrays,
                distance_m,
                index,
            ),
            DEPTH_MODE_RAY_DISTANCE: _compact_vectorized_point(
                ray_arrays,
                np.round(np.sqrt(ray_forward * ray_forward + ray_right * ray_right + ray_up * ray_up), 4),
                index,
            ),
        }
        for index in range(comparison_count)
    ]
    optical_range = _vectorized_range_summary(optical_arrays)
    ray_range = _vectorized_range_summary(ray_arrays)
    projected = len(sampled_points)
    return {
        "frame_status": "ok",
        "projected_points": projected,
        "sampled_points": sampled_points,
        "depth_mode_ranges": {
            DEPTH_MODE_OPTICAL_Z: optical_range,
            DEPTH_MODE_RAY_DISTANCE: ray_range,
        },
        "depth_mode_comparison_samples": depth_mode_comparison_samples,
        "frame_point_ranges": {
            "observation_id": mask_frame.observation_id,
            "pose": pose.to_model_dict(),
            DEPTH_MODE_OPTICAL_Z: optical_range,
            DEPTH_MODE_RAY_DISTANCE: ray_range,
        },
    }


def _vectorized_world_to_local(
    world_x: np.ndarray,
    world_y: np.ndarray,
    world_z: np.ndarray,
    center_pose: PosePoint,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = world_x - center_pose.x
    dy = world_y - center_pose.y
    yaw_rad = math.radians(center_pose.yaw_deg or 0.0)
    local_forward = np.round(math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy, 4)
    local_right = np.round(-math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy, 4)
    local_up = np.round(world_z - (center_pose.z or 0.0), 4)
    return local_forward, local_right, local_up


def _vectorized_range_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    count = len(arrays.get("world_x", []))
    if count <= 0:
        return {"count": 0}
    return {
        "count": count,
        **{
            key: [round(float(np.min(values)), 3), round(float(np.max(values)), 3)]
            for key, values in arrays.items()
        },
    }


def _compact_vectorized_point(
    arrays: dict[str, np.ndarray],
    distance_m: np.ndarray,
    index: int,
) -> dict[str, float]:
    return {
        "camera_x": round(float(arrays["camera_x"][index]), 3),
        "camera_y": round(float(arrays["camera_y"][index]), 3),
        "camera_z": round(float(arrays["camera_z"][index]), 3),
        "world_x": round(float(arrays["world_x"][index]), 3),
        "world_y": round(float(arrays["world_y"][index]), 3),
        "world_z": round(float(arrays["world_z"][index]), 3),
        "local_forward_m": round(float(arrays["local_forward_m"][index]), 3),
        "local_right_m": round(float(arrays["local_right_m"][index]), 3),
        "local_up_m": round(float(arrays["local_up_m"][index]), 3),
        "distance_m": round(float(distance_m[index]), 3),
    }


def _empty_frame_projection_debug(
    observation_id: str | None,
    pose: PosePoint,
    *,
    status: str = "empty",
) -> dict[str, Any]:
    return {
        "frame_status": status,
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


def _merge_voxel_evidence(destination: _VoxelEvidence, source: _VoxelEvidence) -> None:
    """Merge one committed frame's evidence into the persistent world voxel."""

    destination.counts.update(source.counts)
    destination.point_count += int(source.point_count)
    destination.max_z = max(float(destination.max_z), float(source.max_z))
    destination.observation_ids.update(source.observation_ids)


def _limit_debug_points(
    points: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Return deterministic, history-spanning points for a lightweight snapshot."""

    if limit is None or int(limit) >= len(points):
        return list(points)
    limit = max(0, int(limit))
    if limit == 0 or not points:
        return []
    if limit == 1:
        return [points[0]]
    step = (len(points) - 1) / float(limit - 1)
    return [points[int(round(index * step))] for index in range(limit)]


def _recenter_projection_point(point: dict[str, Any], center_pose: PosePoint) -> dict[str, Any]:
    """Refresh local coordinates from canonical world coordinates for a snapshot."""

    result = dict(point)
    world_x = point.get("world_x")
    world_y = point.get("world_y")
    world_z = point.get("world_z")
    if world_x is None or world_y is None:
        return result
    local = _world_to_local(float(world_x), float(world_y), _float_or_none(world_z), center_pose)
    result.update(
        {
            "local_forward_m": round(float(local["local_forward_m"]), 3),
            "local_right_m": round(float(local["local_right_m"]), 3),
            "local_up_m": round(float(local["local_up_m"]), 3),
        }
    )
    return result


def _recenter_comparison_sample(sample: dict[str, Any], center_pose: PosePoint) -> dict[str, Any]:
    result = dict(sample)
    for mode in (DEPTH_MODE_OPTICAL_Z, DEPTH_MODE_RAY_DISTANCE):
        point = sample.get(mode)
        if isinstance(point, dict):
            result[mode] = _recenter_projection_point(point, center_pose)
    return result


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _pose_trace(
    poses: list[PosePoint],
    center_pose: PosePoint,
    grid_size: int,
    meters_per_cell: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project ordered poses without collapsing sub-cell motion.

    ``path_cells`` remains the compatibility representation stamped into the
    semantic matrix. This trace is a separate visual/model aid: fractional
    coordinates preserve direction even when several movements stay inside one
    coarse 5 m cell, and segment IDs prevent lines across off-map gaps.
    """

    trace: list[dict[str, Any]] = []
    segment_index = -1
    previous_visible = False
    previous_distinct_pose: PosePoint | None = None
    task_start_visible = False
    total_translation_m = 0.0
    recent_motion: dict[str, float] | None = None

    for sequence_index, pose in enumerate(poses):
        if previous_distinct_pose is not None:
            dx = pose.x - previous_distinct_pose.x
            dy = pose.y - previous_distinct_pose.y
            distance_m = math.hypot(dx, dy)
            if distance_m < TRAJECTORY_POSITION_EPSILON_M:
                continue
            total_translation_m += distance_m
            delta_local = _world_delta_to_local(dx, dy, center_pose)
            recent_motion = {
                "forward_delta_m": round(delta_local["local_forward_m"], 3),
                "right_delta_m": round(delta_local["local_right_m"], 3),
                "distance_m": round(distance_m, 3),
            }

        previous_distinct_pose = pose
        point = _fractional_grid_point(pose, center_pose, grid_size, meters_per_cell)
        visible = point is not None
        if sequence_index == 0:
            task_start_visible = visible
        if not visible:
            previous_visible = False
            continue
        if not previous_visible:
            segment_index += 1
        trace.append(
            {
                "sequence_index": sequence_index,
                "segment_index": segment_index,
                "row": point["row"],
                "col": point["col"],
                "local_forward_m": point["local_forward_m"],
                "local_right_m": point["local_right_m"],
                "observation_id": pose.observation_id,
            }
        )
        previous_visible = True

    recent = recent_motion or {"forward_delta_m": 0.0, "right_delta_m": 0.0, "distance_m": 0.0}
    summary = {
        "pose_count": len(poses),
        "visible_trace_point_count": len(trace),
        "visible_segment_count": len({int(item["segment_index"]) for item in trace}),
        "visible_start_marker": "START" if trace and task_start_visible else ("ENTRY" if trace else None),
        "total_translation_m": round(total_translation_m, 3),
        "recent_motion_local_m": recent,
        "translation_status": "actual_translation" if recent_motion is not None else "stationary_or_yaw_only",
        "direction_convention": "ordered_trace_old_to_new; arrowheads_toward_newer_poses; image_top_is_current_uav_forward",
    }
    return trace, summary


def _fractional_grid_point(
    pose: PosePoint,
    center_pose: PosePoint,
    grid_size: int,
    meters_per_cell: float,
) -> dict[str, float] | None:
    local = _world_to_local(pose.x, pose.y, pose.z, center_pose)
    center = grid_size // 2
    row = center - local["local_forward_m"] / meters_per_cell
    col = center + local["local_right_m"] / meters_per_cell
    # Integer row/col values are cell centers, hence the half-cell boundary.
    if row < -0.5 or col < -0.5 or row >= grid_size - 0.5 or col >= grid_size - 0.5:
        return None
    return {
        "row": round(row, 4),
        "col": round(col, 4),
        "local_forward_m": round(local["local_forward_m"], 3),
        "local_right_m": round(local["local_right_m"], 3),
    }


def _world_delta_to_local(dx: float, dy: float, center_pose: PosePoint) -> dict[str, float]:
    yaw_rad = math.radians(center_pose.yaw_deg or 0.0)
    return {
        "local_forward_m": math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy,
        "local_right_m": -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy,
    }


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


def _empty_trajectory_summary() -> dict[str, Any]:
    return {
        "pose_count": 0,
        "visible_trace_point_count": 0,
        "visible_segment_count": 0,
        "visible_start_marker": None,
        "total_translation_m": 0.0,
        "recent_motion_local_m": {
            "forward_delta_m": 0.0,
            "right_delta_m": 0.0,
            "distance_m": 0.0,
        },
        "translation_status": "stationary_or_yaw_only",
        "direction_convention": "ordered_trace_old_to_new; arrowheads_toward_newer_poses; image_top_is_current_uav_forward",
    }


def _empty_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def _base_legend() -> dict[int, dict[str, Any]]:
    return {
        UNKNOWN_ID: {"label": "unknown", "source": "default", "color": [148, 163, 184]},
        VISITED_PATH_ID: {"label": "visited_path", "source": "trajectory", "color": [37, 99, 235]},
        0: {"label": "free", "source": "default", "color": [255, 255, 255]},
    }
