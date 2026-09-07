"""Robust metric-depth sampling for a tight normalized image region."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.plugins.contracts.skill import SkillCapabilityResult
from src.depth_display import display_normalized_roi_to_scene_pixels


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

    region = depth[y0:y1, x0:x1]
    valid_mask = np.isfinite(region) & (region > 0.0)
    valid = region[valid_mask].astype(np.float64)
    valid_count = int(valid.size)
    coverage = valid_count / int(region.size)
    if valid_count < 9 or coverage < 0.5:
        raise ValueError(
            "requested ROI has insufficient valid metric depth "
            f"(valid_pixels={valid_count}, coverage={coverage:.3f})"
        )

    raw_median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - raw_median)))
    if mad > _MAX_COHERENT_MAD_M:
        raise ValueError(
            "requested ROI crosses inconsistent depth surfaces; shrink and "
            f"recenter it inside one visible surface (mad_m={mad:.3f})"
        )
    threshold = max(0.25, 3.0 * mad)
    inliers = valid[np.abs(valid - raw_median) <= threshold]
    inlier_count = int(inliers.size)
    if inlier_count < 9 or inlier_count < int(np.ceil(valid_count * 0.5)):
        raise ValueError(
            "requested ROI has insufficient robust depth inliers "
            f"(inlier_pixels={inlier_count}, valid_pixels={valid_count})"
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
