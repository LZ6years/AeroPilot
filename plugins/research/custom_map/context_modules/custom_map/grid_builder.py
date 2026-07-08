"""Build a UAV-centered semantic-topo-metric grid from compressed evidence."""

from __future__ import annotations

import math
from typing import Any

from .mask_semantics import VISITED_PATH_ID
from .schemas import GridMap, PosePoint, SemanticLayer, TrajectorySummary

DEFAULT_GRID_SIZE = 20
DEFAULT_METERS_PER_CELL = 5.0


def build_grid_map(
    trajectory: TrajectorySummary,
    semantic_layer: SemanticLayer,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    meters_per_cell: float = DEFAULT_METERS_PER_CELL,
) -> GridMap:
    center_pose = trajectory.path[-1] if trajectory.path else None
    path_cells = _pose_cells(trajectory.path, center_pose, grid_size, meters_per_cell)
    failed_cells = _pose_cells(trajectory.failed_markers, center_pose, grid_size, meters_per_cell)
    current_cell = {"row": grid_size // 2, "col": grid_size // 2} if center_pose else None
    matrix = _resize_matrix(semantic_layer.matrix, grid_size)
    _stamp_path_cells(matrix, path_cells)
    return GridMap(
        semantic_matrix=matrix,
        semantic_legend=dict(semantic_layer.legend),
        path_cells=path_cells,
        failed_cells=failed_cells,
        current_cell=current_cell,
        center_pose=center_pose.to_model_dict() if center_pose else {},
        path_summary=trajectory.summary,
        map_config={
            "grid_size": grid_size,
            "meters_per_cell": meters_per_cell,
            "orientation": "uav_centered_current_heading_up",
            "semantic_projection": semantic_layer.status,
        },
        warnings=list(semantic_layer.warnings),
        projection_debug=dict(semantic_layer.projection_debug),
    )


def _pose_cells(
    poses: list[PosePoint],
    center_pose: PosePoint | None,
    grid_size: int,
    meters_per_cell: float,
) -> list[dict[str, int]]:
    if center_pose is None:
        return []
    cells: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for pose in poses:
        cell = _project_pose_to_cell(pose, center_pose, grid_size, meters_per_cell)
        if cell is None:
            continue
        key = (cell["row"], cell["col"])
        if key in seen:
            continue
        seen.add(key)
        cells.append(cell)
    return cells


def _project_pose_to_cell(
    pose: PosePoint,
    center_pose: PosePoint,
    grid_size: int,
    meters_per_cell: float,
) -> dict[str, int] | None:
    dx = pose.x - center_pose.x
    dy = pose.y - center_pose.y
    yaw_rad = math.radians(center_pose.yaw_deg or 0.0)
    forward_m = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
    right_m = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    center = grid_size // 2
    row = int(round(center - forward_m / meters_per_cell))
    col = int(round(center + right_m / meters_per_cell))
    if row < 0 or col < 0 or row >= grid_size or col >= grid_size:
        return None
    return {"row": row, "col": col}


def _resize_matrix(matrix: list[list[int]], grid_size: int) -> list[list[int]]:
    if len(matrix) == grid_size and all(len(row) == grid_size for row in matrix):
        return [list(row) for row in matrix]
    return [[0 for _ in range(grid_size)] for _ in range(grid_size)]


def _stamp_path_cells(matrix: list[list[int]], path_cells: list[dict[str, int]]) -> None:
    for cell in path_cells:
        row = cell.get("row")
        col = cell.get("col")
        if row is None or col is None:
            continue
        if 0 <= row < len(matrix) and matrix and 0 <= col < len(matrix[row]):
            matrix[row][col] = VISITED_PATH_ID
