"""Resolve mask colors or ids into explicit semantic labels.

Simulator masks and segmentation images identify regions, not meanings. This
module only assigns semantic names when a label source is present; otherwise it
uses unknown_region labels so model-facing context does not hallucinate classes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .schemas import CustomMapRecords, FrameRecord, SemanticLayer

VISITED_PATH_ID = -1
UNKNOWN_ID = -2
DEFAULT_GRID_SIZE = 20


@dataclass
class SemanticLookup:
    """Map mask colors to semantic ids while preserving unknown regions."""

    label_map: dict[str, dict[str, Any]]
    legend: dict[int, dict[str, Any]] = field(default_factory=lambda: _base_legend())
    unknown_ids: dict[tuple[int, int, int], int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def semantic_id_for_color(self, color: tuple[int, int, int]) -> int:
        if color == (0, 0, 0):
            return 0
        label = self.label_map.get(_color_key(color))
        if label is None:
            self.legend.setdefault(
                UNKNOWN_ID,
                {
                    "label": "unknown",
                    "color": list(color),
                    "source": "mask_color_without_label",
                },
            )
            self.warnings.append(f"custom_map_mask_label_missing:{_color_key(color)}")
            return UNKNOWN_ID
        semantic_id = int(label.get("id", 1))
        if semantic_id == 0:
            return 0
        legend_item = {
            "label": str(label.get("label") or f"semantic_{semantic_id}"),
            "category": label.get("category"),
            "color": list(color),
            "source": str(label.get("source") or "mask_label_map"),
        }
        for key in ("object_name", "confidence"):
            if label.get(key) is not None:
                legend_item[key] = label.get(key)
        self.legend.setdefault(semantic_id, legend_item)
        return semantic_id

    def deduped_warnings(self) -> list[str]:
        return sorted(set(self.warnings))


def build_semantic_lookup(records: CustomMapRecords, frame: FrameRecord | None = None) -> SemanticLookup:
    mask_frame = frame or _latest_mask_frame(records.frames)
    label_map = _load_label_map(records.run_dir, mask_frame) if mask_frame is not None else _file_label_map(records.run_dir)
    return SemanticLookup(label_map=label_map)


def build_semantic_layer(records: CustomMapRecords, grid_size: int = DEFAULT_GRID_SIZE) -> SemanticLayer:
    mask_frame = _latest_mask_frame(records.frames)
    if mask_frame is None or mask_frame.path is None or not mask_frame.path.exists():
        return SemanticLayer(
            matrix=_empty_matrix(grid_size),
            legend=_base_legend(),
            status="mask_unavailable",
            projection_debug={
                "projection_status": "mask_unavailable",
                "point_count": 0,
                "sampled_points": [],
                "source_observation_count": 0,
            },
        )

    lookup = build_semantic_lookup(records, mask_frame)
    matrix = _empty_matrix(grid_size)

    try:
        image = Image.open(mask_frame.path).convert("RGB")
    except OSError as exc:
        return SemanticLayer(
            matrix=matrix,
            legend=lookup.legend,
            status="mask_unreadable",
            warnings=[f"custom_map_mask_unreadable:{exc}"],
            projection_debug={
                "projection_status": "mask_unreadable",
                "point_count": 0,
                "sampled_points": [],
                "source_observation_count": 0,
            },
        )

    width, height = image.size
    pixels = image.load()
    for row in range(grid_size):
        y0 = int(row * height / grid_size)
        y1 = max(y0 + 1, int((row + 1) * height / grid_size))
        for col in range(grid_size):
            x0 = int(col * width / grid_size)
            x1 = max(x0 + 1, int((col + 1) * width / grid_size))
            color = _dominant_color(pixels, x0, x1, y0, y1)
            if color == (0, 0, 0):
                continue
            semantic_id = lookup.semantic_id_for_color(color)
            matrix[row][col] = semantic_id

    return SemanticLayer(
        matrix=matrix,
        legend=lookup.legend,
        status="mask_downsampled_without_depth_projection",
        warnings=lookup.deduped_warnings(),
        projection_debug={
            "projection_status": "mask_downsampled_without_depth_projection",
            "point_count": 0,
            "sampled_points": [],
            "source_observation_count": 0,
        },
    )


def _latest_mask_frame(frames: list[FrameRecord]) -> FrameRecord | None:
    for frame in reversed(frames):
        if frame.modality.lower() == "segmentation":
            return frame
    for frame in reversed(frames):
        if frame.modality.lower() in {"mask", "object_mask"}:
            return frame
    return None


def _load_label_map(run_dir: Path | None, frame: FrameRecord) -> dict[str, dict[str, Any]]:
    label_map: dict[str, dict[str, Any]] = {}
    for source in (_frame_label_map(frame), _file_label_map(run_dir)):
        for key, value in source.items():
            if isinstance(value, dict):
                label_map[str(key)] = dict(value)
    return label_map


def _frame_label_map(frame: FrameRecord) -> dict[str, dict[str, Any]]:
    for key in ("mask_label_map", "label_map", "color_label_map"):
        value = frame.metadata.get(key)
        if isinstance(value, dict):
            return {str(item_key): dict(item_value) for item_key, item_value in value.items() if isinstance(item_value, dict)}
    return {}


def _file_label_map(run_dir: Path | None) -> dict[str, dict[str, Any]]:
    if run_dir is None:
        return {}
    path = run_dir / "memory" / "map" / "mask_label_map.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    colors = data.get("colors") if isinstance(data, dict) else None
    if not isinstance(colors, dict):
        return {}
    return {str(key): dict(value) for key, value in colors.items() if isinstance(value, dict)}


def _dominant_color(pixels: Any, x0: int, x1: int, y0: int, y1: int) -> tuple[int, int, int]:
    counts: Counter[tuple[int, int, int]] = Counter()
    for y in range(y0, y1):
        for x in range(x0, x1):
            counts[tuple(int(channel) for channel in pixels[x, y])] += 1
    return counts.most_common(1)[0][0] if counts else (0, 0, 0)


def _color_key(color: tuple[int, int, int]) -> str:
    return f"{color[0]},{color[1]},{color[2]}"


def _empty_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def _base_legend() -> dict[int, dict[str, Any]]:
    return {
        UNKNOWN_ID: {"label": "unknown", "source": "default", "color": [148, 163, 184]},
        VISITED_PATH_ID: {"label": "visited_path", "source": "trajectory", "color": [37, 99, 235]},
        0: {"label": "free", "source": "default", "color": [255, 255, 255]},
    }
