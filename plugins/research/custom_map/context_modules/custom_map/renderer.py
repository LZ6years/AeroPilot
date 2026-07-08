"""Render custom_map spatial evidence as a planner-facing PNG artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .schemas import GridMap

CELL_PX = 22
GRID_ORIGIN = (40, 60)
IMAGE_SIZE = (900, 560)
BACKGROUND = (248, 250, 252)
GRID_LINE = (203, 213, 225)
TEXT = (30, 41, 59)
PATH_COLOR = (37, 99, 235)
FAILED_COLOR = (220, 38, 38)
UAV_COLOR = (22, 163, 74)
FREE_COLOR = (255, 255, 255)
UNKNOWN_COLOR = (148, 163, 184)


def render_custom_map(grid_map: GridMap, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "latest_custom_map.png"
    image = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((30, 24), "Custom Map: UAV-centered semantic-topo-metric evidence", fill=TEXT, font=font)
    _draw_local_grid(draw, grid_map, font)
    _draw_overview(draw, grid_map, font)
    _draw_legend(draw, grid_map, font)

    image.save(path)
    return path


def _draw_local_grid(draw: ImageDraw.ImageDraw, grid_map: GridMap, font: Any) -> None:
    x0, y0 = GRID_ORIGIN
    matrix = grid_map.semantic_matrix
    size = len(matrix)
    draw.text((x0, y0 - 18), "Local map 20x20, top = UAV forward", fill=TEXT, font=font)
    for row in range(size):
        for col in range(size):
            x = x0 + col * CELL_PX
            y = y0 + row * CELL_PX
            color = _semantic_color(matrix[row][col], grid_map.semantic_legend)
            draw.rectangle((x, y, x + CELL_PX, y + CELL_PX), fill=color, outline=GRID_LINE)
    for cell in grid_map.path_cells:
        _fill_cell(draw, cell, PATH_COLOR)
    for cell in grid_map.failed_cells:
        _draw_failed_cell(draw, cell)
    if grid_map.current_cell:
        _draw_uav_cell(draw, grid_map.current_cell)


def _draw_overview(draw: ImageDraw.ImageDraw, grid_map: GridMap, font: Any) -> None:
    x0, y0 = 520, 60
    w, h = 320, 160
    draw.text((x0, y0 - 18), "Global traversed overview", fill=TEXT, font=font)
    draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(255, 255, 255), outline=GRID_LINE)
    if not grid_map.path_cells:
        draw.text((x0 + 14, y0 + 70), "No trajectory yet", fill=(100, 116, 139), font=font)
        return
    points = []
    for cell in grid_map.path_cells:
        points.append((x0 + int(cell["col"] / 19 * w), y0 + int(cell["row"] / 19 * h)))
    if len(points) > 1:
        draw.line(points, fill=PATH_COLOR, width=3)
    for point in points:
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=PATH_COLOR)
    draw.polygon([(x0 + w - 24, y0 + h - 24), (x0 + w - 12, y0 + h - 16), (x0 + w - 24, y0 + h - 8)], fill=UAV_COLOR)


def _draw_legend(draw: ImageDraw.ImageDraw, grid_map: GridMap, font: Any) -> None:
    x0, y0 = 520, 250
    draw.text((x0, y0), "Legend", fill=TEXT, font=font)
    y = y0 + 24
    items = sorted(grid_map.semantic_legend.items(), key=lambda item: item[0])[:8]
    for semantic_id, label in items:
        color = _semantic_color(semantic_id, grid_map.semantic_legend)
        draw.rectangle((x0, y, x0 + 16, y + 16), fill=color, outline=GRID_LINE)
        draw.text((x0 + 24, y + 2), f"{semantic_id}: {label.get('label')}", fill=TEXT, font=font)
        y += 24
    draw.polygon([(x0, y + 16), (x0 + 8, y), (x0 + 16, y + 16)], fill=UAV_COLOR)
    draw.text((x0 + 24, y + 2), "current UAV", fill=TEXT, font=font)


def _fill_cell(draw: ImageDraw.ImageDraw, cell: dict[str, int], color: tuple[int, int, int]) -> None:
    x0, y0 = GRID_ORIGIN
    x = x0 + cell["col"] * CELL_PX
    y = y0 + cell["row"] * CELL_PX
    inset = 6
    draw.ellipse((x + inset, y + inset, x + CELL_PX - inset, y + CELL_PX - inset), fill=color)


def _draw_failed_cell(draw: ImageDraw.ImageDraw, cell: dict[str, int]) -> None:
    x0, y0 = GRID_ORIGIN
    x = x0 + cell["col"] * CELL_PX
    y = y0 + cell["row"] * CELL_PX
    draw.line((x + 5, y + 5, x + CELL_PX - 5, y + CELL_PX - 5), fill=FAILED_COLOR, width=3)
    draw.line((x + CELL_PX - 5, y + 5, x + 5, y + CELL_PX - 5), fill=FAILED_COLOR, width=3)


def _draw_uav_cell(draw: ImageDraw.ImageDraw, cell: dict[str, int]) -> None:
    x0, y0 = GRID_ORIGIN
    x = x0 + cell["col"] * CELL_PX
    y = y0 + cell["row"] * CELL_PX
    draw.polygon(
        [
            (x + CELL_PX // 2, y + 4),
            (x + CELL_PX - 4, y + CELL_PX - 4),
            (x + 4, y + CELL_PX - 4),
        ],
        fill=UAV_COLOR,
    )


def _semantic_color(semantic_id: int, legend: dict[int, dict[str, Any]]) -> tuple[int, int, int]:
    if semantic_id == -1:
        return PATH_COLOR
    if semantic_id == -2:
        return UNKNOWN_COLOR
    if semantic_id == 0:
        return FREE_COLOR
    label = legend.get(semantic_id, {})
    raw_color = label.get("color")
    if isinstance(raw_color, list) and len(raw_color) >= 3:
        return tuple(max(80, int(channel)) for channel in raw_color[:3])  # type: ignore[return-value]
    palette = [
        (254, 226, 226),
        (219, 234, 254),
        (220, 252, 231),
        (254, 249, 195),
        (237, 233, 254),
        (255, 237, 213),
    ]
    return palette[semantic_id % len(palette)]
