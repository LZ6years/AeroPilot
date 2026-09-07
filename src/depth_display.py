"""Shared geometry for rendered metric-depth images and their scene ROI."""

from __future__ import annotations

import math


def depth_legend_height(scene_height: int) -> int:
    """Return the deterministic color-bar height appended below a scene."""

    return max(42, min(68, round(int(scene_height) * 0.18)))


def depth_display_height(scene_height: int) -> int:
    """Return scene plus appended color-bar height in displayed pixels."""

    return int(scene_height) + depth_legend_height(scene_height)


def display_normalized_roi_to_scene_pixels(
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    scene_width: int,
    scene_height: int,
) -> tuple[int, int, int, int]:
    """Map displayed-image normalized coordinates into aligned scene pixels.

    The fused provider image keeps scene width unchanged and appends only a
    color bar below it. A box that touches that bar is invalid rather than
    silently clipped into the scene.
    """

    width = int(scene_width)
    height = int(scene_height)
    display_height = depth_display_height(height)
    x0 = max(0, min(width, math.floor(float(x_min) * width)))
    y0 = max(0, min(display_height, math.floor(float(y_min) * display_height)))
    x1 = max(0, min(width, math.ceil(float(x_max) * width)))
    y1 = max(0, min(display_height, math.ceil(float(y_max) * display_height)))
    if y0 >= height or y1 > height:
        raise ValueError("requested ROI must stay inside the scene above the color bar")
    return x0, y0, x1, y1
