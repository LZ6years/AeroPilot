"""Shared data structures for the custom_map context module.

The plugin keeps raw run memory separate from model-facing map summaries. These
dataclasses describe the compressed data passed between the reader, trajectory,
semantic-mask, grid, and renderer stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PosePoint:
    x: float
    y: float
    z: float | None = None
    yaw_deg: float | None = None
    timestamp: str | None = None
    source: str | None = None
    action_id: str | None = None
    observation_id: str | None = None

    def to_model_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"x": round(self.x, 3), "y": round(self.y, 3)}
        if self.z is not None:
            data["z"] = round(self.z, 3)
        if self.yaw_deg is not None:
            data["yaw_deg"] = round(self.yaw_deg, 3)
        return data


@dataclass(frozen=True)
class ActionRecord:
    id: str | None
    tool: str
    args: dict[str, Any]
    ok: bool | None
    timestamp: str | None = None


@dataclass(frozen=True)
class FrameRecord:
    id: str | None
    observation_id: str | None
    modality: str
    path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None


@dataclass(frozen=True)
class CustomMapRecords:
    run_dir: Path | None
    actions: list[ActionRecord]
    poses: list[PosePoint]
    frames: list[FrameRecord]
    poses_by_observation_id: dict[str, PosePoint] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrajectorySummary:
    path: list[PosePoint]
    failed_markers: list[PosePoint]
    segments: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class SemanticLayer:
    matrix: list[list[int]]
    legend: dict[int, dict[str, Any]]
    status: str
    warnings: list[str] = field(default_factory=list)
    projection_debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GridMap:
    semantic_matrix: list[list[int]]
    semantic_legend: dict[int, dict[str, Any]]
    path_cells: list[dict[str, int]]
    failed_cells: list[dict[str, int]]
    current_cell: dict[str, int] | None
    center_pose: dict[str, Any]
    path_summary: dict[str, Any]
    map_config: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    projection_debug: dict[str, Any] = field(default_factory=dict)
