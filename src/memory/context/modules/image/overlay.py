"""Pixel-aligned RGB-depth overlay rendering and persistence."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from src.depth_display import depth_legend_height
from src.memory.context.contracts import DerivedMediaWriter, MediaRef
from src.perception.frame_payload import load_depth_frame, load_rgb_frame

CURRENT_RGB_DEPTH_OVERLAY_USAGE = (
    "RGB + METRIC DEPTH OVERLAY. False colors provide qualitative range only; "
    "the bottom color bar uses meters. Display colors do not provide precise target-distance measurements."
)
DEFAULT_DEPTH_DISPLAY_MAX_M = 10.0
_DEPTH_OVERLAY_ALPHA = 0.42
_DEPTH_COLOR_STOPS = (
    (180, 0, 0),
    (255, 110, 0),
    (255, 230, 0),
    (50, 190, 80),
    (0, 170, 220),
    (50, 60, 180),
)

def current_rgb_depth_overlay_media_input(
    current_depth: dict[str, Any] | None,
    *,
    current_rgb: dict[str, Any] | None,
    depth_display_max_m: float,
    media_writer: DerivedMediaWriter | None,
) -> dict[str, Any] | None:
    """Persist one derived overlay and return a binary-free MediaRef."""

    if (
        not current_depth
        or current_depth.get("mock") is True
        or not current_rgb
        or current_rgb.get("mock") is True
    ):
        return None
    if media_writer is None:
        raise ValueError(
            "include_current_depth requires a Runtime-owned derived media writer"
        )
    rendered = _rgb_depth_overlay_png(
        current_rgb,
        current_depth,
        metadata=current_depth,
        depth_display_max_m=depth_display_max_m,
    )
    if rendered is None:
        return None
    payload, scene_size, display_size, legend_height = rendered
    observation_id = str(current_rgb.get("observation_id") or "")
    path, digest = media_writer.write_png(observation_id, payload)
    media = MediaRef(
        ref=path,
        sha256=digest,
        mime_type="image/png",
        observation_id=observation_id,
        frame_id=str(current_rgb.get("id") or ""),
        role="current_rgb_depth_overlay",
        usage=CURRENT_RGB_DEPTH_OVERLAY_USAGE,
        width=scene_size[0],
        height=scene_size[1],
        display_width=display_size[0],
        display_height=display_size[1],
        color_bar_height=legend_height,
        source_frame_ids=tuple(
            str(value)
            for value in (current_rgb.get("id"), current_depth.get("id"))
            if value
        ),
    )
    return media.to_model_input()



def _rgb_depth_overlay_png(
    rgb_path: Path | dict[str, Any],
    depth_path: Path | dict[str, Any],
    *,
    metadata: dict[str, Any],
    depth_display_max_m: float,
) -> tuple[bytes, tuple[int, int], tuple[int, int], int] | None:
    """Render aligned RGB/depth bytes without creating a transient data URL."""

    try:
        import numpy as np

        rgb_frame = rgb_path if isinstance(rgb_path, dict) else {"path": str(rgb_path)}
        depth_frame = depth_path if isinstance(depth_path, dict) else {**metadata, "path": str(depth_path)}
        rgb = load_rgb_frame(rgb_frame)
        if depth_frame.get("capture_source") == "depth_vis" or depth_frame.get("depth_unit") == "normalized":
            raise ValueError("DepthVis cannot be used as metric depth")
        depth = load_depth_frame(depth_frame)
        if rgb.size != (int(depth.shape[1]), int(depth.shape[0])):
            return None
        valid = np.isfinite(depth) & (depth > 0.0)
        if not bool(valid.any()):
            return None
        colored = _colorize_metric_depth(depth, valid, depth_display_max_m)
        rgb_array = np.asarray(rgb, dtype=np.float32)
        overlay = (
            (1.0 - _DEPTH_OVERLAY_ALPHA) * rgb_array
            + _DEPTH_OVERLAY_ALPHA * colored.astype(np.float32)
        )
        overlay[~valid] = rgb_array[~valid]
        scene = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
        annotated = _annotated_rgb_depth_overlay_image(scene, depth_display_max_m)
        output = BytesIO()
        annotated.save(output, format="PNG")
    except (OSError, EOFError, UnidentifiedImageError, ValueError, TypeError):
        return None
    return output.getvalue(), scene.size, annotated.size, annotated.height - scene.height


def _colorize_metric_depth(depth: Any, valid: Any, depth_display_max_m: float) -> Any:
    import numpy as np

    normalized = np.clip(depth / float(depth_display_max_m), 0.0, 1.0)
    stop_positions = np.linspace(0.0, 1.0, len(_DEPTH_COLOR_STOPS), dtype=np.float32)
    colors = np.asarray(_DEPTH_COLOR_STOPS, dtype=np.float32)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = np.interp(
            normalized,
            stop_positions,
            colors[:, channel],
        ).astype(np.uint8)
    output[~valid] = 0
    return output


def _annotated_rgb_depth_overlay_image(
    scene: Image.Image,
    depth_display_max_m: float,
) -> Image.Image:
    legend_height = depth_legend_height(scene.height)
    annotated = Image.new(
        "RGB",
        (scene.width, scene.height + legend_height),
        (20, 24, 30),
    )
    annotated.paste(scene, (0, 0))
    _draw_depth_legend(
        annotated,
        top=scene.height,
        height=legend_height,
        depth_display_max_m=depth_display_max_m,
    )
    return annotated


def _draw_depth_legend(
    image: Image.Image,
    *,
    top: int,
    height: int,
    depth_display_max_m: float,
) -> None:
    import numpy as np

    draw = ImageDraw.Draw(image)
    left = max(1, min(round(image.width * 0.04), max(1, image.width // 4)))
    right = max(left + 1, image.width - left)
    bar_top = top + 7
    bar_height = max(9, min(16, height // 3))
    gradient_depth = np.linspace(
        0.0,
        depth_display_max_m,
        max(1, right - left),
        dtype=np.float32,
    )[None, :]
    gradient = _colorize_metric_depth(
        gradient_depth,
        np.ones_like(gradient_depth, dtype=bool),
        depth_display_max_m,
    )
    gradient_image = Image.fromarray(gradient, mode="RGB").resize(
        (right - left, bar_height),
        resample=Image.Resampling.NEAREST,
    )
    image.paste(gradient_image, (left, bar_top))
    draw.rectangle(
        (left, bar_top, right - 1, bar_top + bar_height - 1),
        outline=(235, 235, 235),
        width=1,
    )
    label_top = bar_top + bar_height + 3
    font = _default_font(max(10, min(16, height // 4)))
    labels = []
    for index in range(6):
        fraction = index / 5
        x = round(left + fraction * (right - left))
        value = depth_display_max_m * fraction
        suffix = "+ m" if index == 5 else " m"
        labels.append((x, f"{value:g}{suffix}"))
    for x, label in labels:
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        if x == left:
            text_x = x
        elif x == right:
            text_x = x - text_width
        else:
            text_x = x - text_width // 2
        draw.text((text_x, label_top), label, fill=(255, 255, 255), font=font)


def _default_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
