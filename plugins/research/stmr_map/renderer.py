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
UAV_COLOR = (22, 163, 74)
UNKNOWN_COLOR = (148, 163, 184)
FREE_COLOR = (255, 255, 255)


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
    draw.text((x0, y0 - 18), "20x20, 5m/cell, top = UAV forward", fill=TEXT, font=font)
    for row, values in enumerate(matrix):
        for col, semantic_id in enumerate(values):
            x = x0 + col * CELL_PX
            y = y0 + row * CELL_PX
            draw.rectangle((x, y, x + CELL_PX, y + CELL_PX), fill=_semantic_color(semantic_id, stmr_map.semantic_legend), outline=GRID_LINE)
    if stmr_map.current_cell:
        x = x0 + stmr_map.current_cell["col"] * CELL_PX
        y = y0 + stmr_map.current_cell["row"] * CELL_PX
        draw.polygon([(x + CELL_PX // 2, y + 4), (x + CELL_PX - 4, y + CELL_PX - 4), (x + 4, y + CELL_PX - 4)], fill=UAV_COLOR)


def _draw_legend(draw: ImageDraw.ImageDraw, stmr_map: StmrMap, font: Any) -> None:
    x0, y0 = 560, 64
    draw.text((x0, y0), "Legend", fill=TEXT, font=font)
    y = y0 + 24
    for semantic_id, item in sorted(stmr_map.semantic_legend.items(), key=lambda pair: pair[0])[:16]:
        draw.rectangle((x0, y, x0 + 16, y + 16), fill=_semantic_color(semantic_id, stmr_map.semantic_legend), outline=GRID_LINE)
        draw.text((x0 + 24, y + 2), f"{semantic_id}: {item.get('label')}", fill=TEXT, font=font)
        y += 23


def _draw_debug(draw: ImageDraw.ImageDraw, stmr_map: StmrMap, font: Any) -> None:
    x0, y0 = 560, 460
    debug = stmr_map.projection_debug
    lines = [
        f"points: {debug.get('point_count', 0)}",
        f"voxels: {debug.get('voxel_count', 0)}",
        f"observations: {debug.get('source_observation_count', 0)}",
        f"subgoal hits: {len(debug.get('subgoal_priority_hits', []))}",
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

