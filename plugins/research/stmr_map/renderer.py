"""Render paper-style STMR matrices for standalone inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .core import StmrMap

CELL_PX = 24
GRID_ORIGIN = (36, 58)
IMAGE_SIZE = (920, 600)
BACKGROUND = (248, 250, 252)
GRID_LINE = (203, 213, 225)
TEXT = (15, 23, 42)
PATH_COLOR = (37, 99, 235)
TRACE_COLOR = (234, 88, 12)
TRACE_OUTLINE = (255, 255, 255)
UAV_COLOR = (22, 163, 74)
UNKNOWN_COLOR = (148, 163, 184)
FREE_COLOR = (255, 255, 255)
ARROW_SPACING_PX = CELL_PX * 1.5
MAX_ARROWHEADS = 6


def render_stmr_map(stmr_map: StmrMap, output_dir: Path) -> Path:
    """Render ``latest_stmr_map.png``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "latest_stmr_map.png"
    image = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 24), "STMR Map: voxel top-height semantic-topo-metric matrix", fill=TEXT, font=font)
    _draw_matrix(draw, stmr_map, font)
    _draw_legend(draw, stmr_map, font)
    _draw_debug(draw, stmr_map, font)
    image.save(path)
    return path


def _draw_matrix(draw: ImageDraw.ImageDraw, stmr_map: StmrMap, font: Any) -> None:
    x0, y0 = GRID_ORIGIN
    matrix = stmr_map.semantic_matrix
    draw.text((x0, y0 - 18), "20x20, 5m/cell, top = CURRENT UAV HEADING", fill=TEXT, font=font)
    for row, values in enumerate(matrix):
        for col, semantic_id in enumerate(values):
            x = x0 + col * CELL_PX
            y = y0 + row * CELL_PX
            draw.rectangle((x, y, x + CELL_PX, y + CELL_PX), fill=_semantic_color(semantic_id, stmr_map.semantic_legend), outline=GRID_LINE)
    _draw_trajectory_trace(draw, stmr_map, font)
    if stmr_map.current_cell:
        x = x0 + stmr_map.current_cell["col"] * CELL_PX
        y = y0 + stmr_map.current_cell["row"] * CELL_PX
        draw.polygon([(x + CELL_PX // 2, y + 4), (x + CELL_PX - 4, y + CELL_PX - 4), (x + 4, y + CELL_PX - 4)], fill=UAV_COLOR)
        draw.text((x + CELL_PX - 2, y + 2), "NOW", fill=UAV_COLOR, font=font)


def _draw_trajectory_trace(draw: ImageDraw.ImageDraw, stmr_map: StmrMap, font: Any) -> None:
    """Overlay actual old-to-new travel without changing semantic cells."""

    segments = _trace_pixel_segments(stmr_map.trajectory_trace)
    moving_segments = [segment for segment in segments if _polyline_length(segment) >= 2.0]
    if not moving_segments:
        return
    for segment in moving_segments:
        draw.line(segment, fill=TRACE_OUTLINE, width=7, joint="curve")
        draw.line(segment, fill=TRACE_COLOR, width=4, joint="curve")
    for x, y, ux, uy in _arrow_placements(moving_segments):
        _draw_arrowhead(draw, x, y, ux, uy)

    first = moving_segments[0][0]
    marker = str(stmr_map.trajectory_summary.get("visible_start_marker") or "ENTRY")
    radius = 6
    draw.ellipse(
        (first[0] - radius, first[1] - radius, first[0] + radius, first[1] + radius),
        fill=TRACE_OUTLINE,
        outline=TRACE_COLOR,
        width=3,
    )
    draw.text((first[0] + 8, first[1] - 13), marker, fill=TRACE_COLOR, font=font)


def _trace_pixel_segments(trace: list[dict[str, Any]]) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current_segment: int | None = None
    for item in trace:
        try:
            segment_index = int(item["segment_index"])
            row = float(item["row"])
            col = float(item["col"])
        except (KeyError, TypeError, ValueError):
            continue
        if current_segment != segment_index:
            segments.append([])
            current_segment = segment_index
        segments[-1].append(
            (
                GRID_ORIGIN[0] + (col + 0.5) * CELL_PX,
                GRID_ORIGIN[1] + (row + 0.5) * CELL_PX,
            )
        )
    return [segment for segment in segments if segment]


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(
        ((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5
        for left, right in zip(points, points[1:])
    )


def _arrow_placements(
    segments: list[list[tuple[float, float]]],
    *,
    spacing_px: float = ARROW_SPACING_PX,
    max_arrows: int = MAX_ARROWHEADS,
) -> list[tuple[float, float, float, float]]:
    """Return arrow tips and unit travel vectors at regular path distance."""

    placements: list[tuple[float, float, float, float]] = []
    longest: list[tuple[float, float]] | None = None
    longest_length = 0.0
    for points in segments:
        length = _polyline_length(points)
        if length > longest_length:
            longest, longest_length = points, length
        target_distance = spacing_px
        traversed = 0.0
        for left, right in zip(points, points[1:]):
            dx, dy = right[0] - left[0], right[1] - left[1]
            edge_length = (dx * dx + dy * dy) ** 0.5
            if edge_length <= 0.0:
                continue
            while traversed + edge_length >= target_distance and len(placements) < max_arrows:
                ratio = (target_distance - traversed) / edge_length
                placements.append((left[0] + ratio * dx, left[1] + ratio * dy, dx / edge_length, dy / edge_length))
                target_distance += spacing_px
            traversed += edge_length
            if len(placements) >= max_arrows:
                return placements
    if not placements and longest is not None and longest_length >= 2.0:
        return _placement_at_distance(longest, longest_length / 2.0)
    return placements


def _placement_at_distance(
    points: list[tuple[float, float]],
    target_distance: float,
) -> list[tuple[float, float, float, float]]:
    traversed = 0.0
    for left, right in zip(points, points[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        edge_length = (dx * dx + dy * dy) ** 0.5
        if edge_length <= 0.0:
            continue
        if traversed + edge_length >= target_distance:
            ratio = (target_distance - traversed) / edge_length
            return [(left[0] + ratio * dx, left[1] + ratio * dy, dx / edge_length, dy / edge_length)]
        traversed += edge_length
    return []


def _draw_arrowhead(draw: ImageDraw.ImageDraw, x: float, y: float, ux: float, uy: float) -> None:
    def polygon(length: float, half_width: float) -> list[tuple[float, float]]:
        base_x, base_y = x - ux * length, y - uy * length
        perpendicular_x, perpendicular_y = -uy, ux
        return [
            (x, y),
            (base_x + perpendicular_x * half_width, base_y + perpendicular_y * half_width),
            (base_x - perpendicular_x * half_width, base_y - perpendicular_y * half_width),
        ]

    draw.polygon(polygon(13.0, 8.0), fill=TRACE_OUTLINE)
    draw.polygon(polygon(10.0, 5.0), fill=TRACE_COLOR)


def _draw_legend(draw: ImageDraw.ImageDraw, stmr_map: StmrMap, font: Any) -> None:
    x0, y0 = 560, 64
    draw.text((x0, y0), "Legend", fill=TEXT, font=font)
    y = y0 + 24
    for semantic_id, item in sorted(stmr_map.semantic_legend.items(), key=lambda pair: pair[0])[:12]:
        draw.rectangle((x0, y, x0 + 16, y + 16), fill=_semantic_color(semantic_id, stmr_map.semantic_legend), outline=GRID_LINE)
        draw.text((x0 + 24, y + 2), f"{semantic_id}: {item.get('label')}", fill=TEXT, font=font)
        y += 23
    draw.line((x0, y + 8, x0 + 18, y + 8), fill=TRACE_OUTLINE, width=7)
    draw.line((x0, y + 8, x0 + 18, y + 8), fill=TRACE_COLOR, width=4)
    _draw_arrowhead(draw, x0 + 18, y + 8, 1.0, 0.0)
    draw.text((x0 + 26, y + 2), "actual travel: old -> new", fill=TEXT, font=font)
    y += 23
    draw.polygon([(x0 + 8, y), (x0 + 16, y + 16), (x0, y + 16)], fill=UAV_COLOR)
    draw.text((x0 + 26, y + 2), "NOW: current pose + heading", fill=TEXT, font=font)


def _draw_debug(draw: ImageDraw.ImageDraw, stmr_map: StmrMap, font: Any) -> None:
    x0, y0 = 560, 450
    debug = stmr_map.projection_debug
    trajectory = stmr_map.trajectory_summary
    lines = [
        f"points: {debug.get('point_count', 0)}",
        f"voxels: {debug.get('voxel_count', 0)}",
        f"observations: {debug.get('source_observation_count', 0)}",
        f"subgoal hits: {len(debug.get('subgoal_priority_hits', []))}",
        f"trajectory poses: {trajectory.get('pose_count', 0)}",
        f"motion: {trajectory.get('translation_status', 'stationary_or_yaw_only')}",
    ]
    for index, line in enumerate(lines):
        draw.text((x0, y0 + index * 20), line, fill=TEXT, font=font)


def _semantic_color(semantic_id: int, legend: dict[int, dict[str, Any]]) -> tuple[int, int, int]:
    if semantic_id == -1:
        return PATH_COLOR
    if semantic_id == -2:
        return UNKNOWN_COLOR
    if semantic_id == 0:
        return FREE_COLOR
    raw = legend.get(semantic_id, {}).get("color")
    if isinstance(raw, list) and len(raw) >= 3:
        return tuple(max(55, int(channel)) for channel in raw[:3])  # type: ignore[return-value]
    return (219, 234, 254)
