"""Robust metric-depth sampling for a tight normalized image region."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from src.depth_display import display_normalized_roi_to_scene_pixels
from src.plugins.contracts.skill import (
    SkillCapabilityFailure,
    SkillCapabilityResult,
)


RESULT_MESSAGE = (
    "Approximate camera-to-visible-surface distance, computed as the robust "
    "median metric depth inside the requested small ROI and rounded to 0.1 m."
)
_MAX_COHERENT_MAD_M = 0.25


def handle(payload: dict) -> SkillCapabilityResult:
    args = dict(payload.get("args", {}))
    runtime_context = dict(payload.get("runtime_context", {}))
    latest_observation = dict(runtime_context.get("latest_observation", {}))
    latest_id = str(latest_observation.get("observation_id") or "")
    requested_id = str(args.get("observation_id") or "")
    if requested_id != latest_id:
        raise ValueError(
            "metric depth sampling only accepts the latest observation_id; "
            f"requested={requested_id!r}, latest={latest_id!r}"
        )

    frames = latest_observation.get("frames")
    depth_frame = frames.get("depth") if isinstance(frames, dict) else None
    if not isinstance(depth_frame, dict):
        raise ValueError("latest observation has no metric-depth frame")
    if depth_frame.get("depth_unit") != "meter":
        raise ValueError("latest depth frame is not declared in meters")
    depth_path = _frame_path(depth_frame)
    if depth_path is None:
        raise ValueError("metric-depth artifact is unavailable")

    depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
    depth = np.squeeze(depth)
    if depth.ndim == 1:
        width = _positive_int(depth_frame.get("width"))
        height = _positive_int(depth_frame.get("height"))
        if width is None or height is None or depth.size != width * height:
            raise ValueError("flat metric-depth frame has invalid dimensions")
        depth = depth.reshape((height, width))
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("metric-depth frame must be a non-empty 2D array")

    height, width = int(depth.shape[0]), int(depth.shape[1])
    x0, y0, x1, y1 = display_normalized_roi_to_scene_pixels(
        x_min=float(args["x_min"]),
        y_min=float(args["y_min"]),
        x_max=float(args["x_max"]),
        y_max=float(args["y_max"]),
        scene_width=width,
        scene_height=height,
    )
    if x1 - x0 < 3 or y1 - y0 < 3:
        raise ValueError("requested ROI must cover at least 3x3 scene pixels")

    artifact_refs, model_inputs = _write_roi_confirmation(
        rgb_frame=frames.get("rgb") if isinstance(frames, dict) else None,
        run_dir=runtime_context.get("run_dir"),
        observation_id=latest_id,
        tool_call_id=runtime_context.get("tool_call_id"),
        roi_pixels=(x0, y0, x1, y1),
        scene_size=(width, height),
    )

    region = depth[y0:y1, x0:x1]
    valid_mask = np.isfinite(region) & (region > 0.0)
    valid = region[valid_mask].astype(np.float64)
    valid_count = int(valid.size)
    coverage = valid_count / int(region.size)
    if valid_count < 9 or coverage < 0.5:
        raise SkillCapabilityFailure(
            "requested ROI has insufficient valid metric depth "
            f"(valid_pixels={valid_count}, coverage={coverage:.3f})",
            data={
                "observation_id": latest_id,
                "audit_details": {
                    "frame_id": depth_frame.get("id"),
                    "roi_pixels": [x0, y0, x1, y1],
                    "valid_pixel_count": valid_count,
                },
            },
            artifact_refs=artifact_refs,
            model_inputs=model_inputs,
        )

    raw_median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - raw_median)))
    if mad > _MAX_COHERENT_MAD_M:
        raise SkillCapabilityFailure(
            "requested ROI crosses inconsistent depth surfaces; shrink and "
            f"recenter it inside one visible surface (mad_m={mad:.3f})",
            data={
                "observation_id": latest_id,
                "audit_details": {
                    "frame_id": depth_frame.get("id"),
                    "roi_pixels": [x0, y0, x1, y1],
                    "valid_pixel_count": valid_count,
                    "raw_median_m": raw_median,
                    "mad_m": mad,
                },
            },
            artifact_refs=artifact_refs,
            model_inputs=model_inputs,
        )
    threshold = max(0.25, 3.0 * mad)
    inliers = valid[np.abs(valid - raw_median) <= threshold]
    inlier_count = int(inliers.size)
    if inlier_count < 9 or inlier_count < int(np.ceil(valid_count * 0.5)):
        raise SkillCapabilityFailure(
            "requested ROI has insufficient robust depth inliers "
            f"(inlier_pixels={inlier_count}, valid_pixels={valid_count})",
            data={
                "observation_id": latest_id,
                "audit_details": {
                    "frame_id": depth_frame.get("id"),
                    "roi_pixels": [x0, y0, x1, y1],
                    "valid_pixel_count": valid_count,
                    "inlier_pixel_count": inlier_count,
                    "raw_median_m": raw_median,
                    "mad_m": mad,
                },
            },
            artifact_refs=artifact_refs,
            model_inputs=model_inputs,
        )

    distance_m = round(float(np.median(inliers)) + 1e-12, 1)
    return SkillCapabilityResult(
        message=RESULT_MESSAGE,
        data={
            "observation_id": latest_id,
            "distance_m": distance_m,
            "audit_details": {
                "frame_id": depth_frame.get("id"),
                "roi_pixels": [x0, y0, x1, y1],
                "valid_pixel_count": valid_count,
                "inlier_pixel_count": inlier_count,
                "raw_median_m": raw_median,
                "mad_m": mad,
            },
        },
        artifact_refs=artifact_refs,
        model_inputs=model_inputs,
    )


def _write_roi_confirmation(
    *,
    rgb_frame: object,
    run_dir: object,
    observation_id: str,
    tool_call_id: object,
    roi_pixels: tuple[int, int, int, int],
    scene_size: tuple[int, int],
) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    """Save a target-agnostic three-panel ROI confirmation image.

    No semantic label is drawn. The red rectangle represents only the exact
    region submitted by the model, allowing the next VLM turn to decide
    whether it actually lies inside the intended visible surface.
    """

    if not isinstance(rgb_frame, dict) or not isinstance(run_dir, str) or not run_dir:
        return (), ()
    rgb_path = _frame_path(rgb_frame)
    if rgb_path is None:
        return (), ()
    width, height = scene_size
    x0, y0, x1, y1 = roi_pixels
    try:
        with Image.open(rgb_path) as source:
            scene = source.convert("RGB")
        if scene.size != (width, height):
            return (), ()
        line_width = max(2, round(min(width, height) / 96))
        full = scene.copy()
        ImageDraw.Draw(full).rectangle(
            (x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)),
            outline=(255, 30, 70),
            width=line_width,
        )

        roi_width = max(1, x1 - x0)
        roi_height = max(1, y1 - y0)
        side = min(
            min(width, height),
            max(48, 6 * max(roi_width, roi_height)),
        )
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        crop_x0 = max(0, min(width - side, center_x - side // 2))
        crop_y0 = max(0, min(height - side, center_y - side // 2))
        crop = scene.crop((crop_x0, crop_y0, crop_x0 + side, crop_y0 + side))
        crop_draw = ImageDraw.Draw(crop)
        crop_draw.rectangle(
            (
                x0 - crop_x0,
                y0 - crop_y0,
                max(x0 - crop_x0, x1 - crop_x0 - 1),
                max(y0 - crop_y0, y1 - crop_y0 - 1),
            ),
            outline=(255, 30, 70),
            width=max(1, round(side / 96)),
        )
        zoom = crop.resize((width, height), resample=Image.Resampling.NEAREST)
        exact = scene.crop((x0, y0, x1, y1)).resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )
        ImageDraw.Draw(exact).rectangle(
            (0, 0, width - 1, height - 1),
            outline=(255, 30, 70),
            width=line_width,
        )
        separator_width = max(4, width // 64)
        confirmation = Image.new(
            "RGB", (width * 3 + separator_width * 2, height), (15, 18, 24)
        )
        confirmation.paste(full, (0, 0))
        confirmation.paste(zoom, (width + separator_width, 0))
        confirmation.paste(exact, (width * 2 + separator_width * 2, 0))

        identity = hashlib.sha256(
            f"{observation_id}|{tool_call_id}|{roi_pixels}".encode("utf-8")
        ).hexdigest()[:20]
        output_dir = Path(run_dir) / "artifacts" / "tool_evidence"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"metric_depth_roi_{identity}.png"
        confirmation.save(output_path, format="PNG")
    except (OSError, ValueError, UnidentifiedImageError):
        return (), ()

    path = str(output_path)
    return (
        (path,),
        (
            {
                "type": "image",
                "role": "metric_depth_roi_confirmation",
                "observation_id": observation_id,
                "tool_call_id": str(tool_call_id or ""),
                "artifact_ref": path,
                "path": path,
                "mime_type": "image/png",
                "usage": (
                    "Left panel is the full CURRENT RGB, middle is a magnified "
                    "neighborhood, and right contains only the sampled ROI "
                    "pixels. The red rectangle is the exact sampled ROI. "
                    "Identify what is actually inside it before using the "
                    "distance. If it is not wholly inside the intended visible "
                    "surface, call the metric-depth tool again with a corrected "
                    "ROI."
                ),
            },
        ),
    )


def _frame_path(frame: dict) -> Path | None:
    raw = frame.get("path") or frame.get("artifact_path") or frame.get("artifact_ref")
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    if path.is_file():
        return path
    candidate = Path.cwd() / path
    return candidate if candidate.is_file() else None


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
