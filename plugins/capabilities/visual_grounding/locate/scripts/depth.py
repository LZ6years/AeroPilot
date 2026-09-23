"""Robust candidate-scoped metric-depth estimation for visual grounding."""

from __future__ import annotations

from src.perception.frame_payload import load_depth_frame

import numpy as np

from plugins.capabilities.visual_grounding.locate.scripts.contracts import (
    CandidateMask,
    DepthEstimate,
    GroundedCandidate,
    VisualGroundingSettings,
)


def load_metric_depth(frame: dict) -> np.ndarray:
    return load_depth_frame(frame, require_metric=True)


def estimate_candidate_depth(
    depth: np.ndarray,
    candidate: GroundedCandidate,
    candidate_mask: CandidateMask | None,
    settings: VisualGroundingSettings,
) -> DepthEstimate:
    height, width = depth.shape
    box = _clamped_box(candidate.box_xyxy, width, height)
    sample_mask, source = _sampling_mask(
        candidate_mask, box, width, height, settings
    )
    sample_count = int(np.count_nonzero(sample_mask))
    if sample_count < settings.minimum_valid_depth_pixels:
        return DepthEstimate(None, "insufficient_region", {"sampling_source": source, "box_pixels": list(box), "sample_pixel_count": sample_count})

    values = depth[sample_mask]
    valid = values[np.isfinite(values) & (values > 0)].astype(np.float64)
    valid_count = int(valid.size)
    coverage = valid_count / sample_count
    audit = {
        "sampling_source": source,
        "box_pixels": list(box),
        "sample_pixel_count": sample_count,
        "valid_pixel_count": valid_count,
        "valid_coverage": coverage,
    }
    if valid_count < settings.minimum_valid_depth_pixels or coverage < settings.minimum_valid_depth_coverage:
        return DepthEstimate(None, "insufficient_valid_depth", audit)

    raw_median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - raw_median)))
    audit.update({"raw_median_m": raw_median, "mad_m": mad})
    if mad > settings.maximum_coherent_mad_m:
        return DepthEstimate(None, "inconsistent_depth_surfaces", audit)
    threshold = max(0.25, 3.0 * mad)
    inliers = valid[np.abs(valid - raw_median) <= threshold]
    inlier_count = int(inliers.size)
    audit["inlier_pixel_count"] = inlier_count
    if inlier_count < settings.minimum_valid_depth_pixels or inlier_count < int(np.ceil(valid_count * 0.5)):
        return DepthEstimate(None, "insufficient_depth_inliers", audit)
    return DepthEstimate(round(float(np.median(inliers)) + 1e-12, 1), "ok", audit)


def _sampling_mask(
    candidate_mask: CandidateMask | None,
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    settings: VisualGroundingSettings,
) -> tuple[np.ndarray, str]:
    if candidate_mask is not None:
        mask = np.asarray(candidate_mask.mask, dtype=bool)
        if mask.shape == (height, width) and np.any(mask):
            iterations = max(1, int(round(min(box[2] - box[0], box[3] - box[1]) * settings.mask_erosion_ratio)))
            eroded = _erode(mask, iterations)
            if np.count_nonzero(eroded) >= settings.minimum_valid_depth_pixels:
                return eroded, "eroded_mask"
    x0, y0, x1, y1 = box
    inset_x = int(round((x1 - x0) * settings.box_inset_ratio))
    inset_y = int(round((y1 - y0) * settings.box_inset_ratio))
    x0, x1 = x0 + inset_x, x1 - inset_x
    y0, y1 = y0 + inset_y, y1 - inset_y
    mask = np.zeros((height, width), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask, "box_interior"


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_and.reduce(
            [padded[y : y + result.shape[0], x : x + result.shape[1]] for y in range(3) for x in range(3)]
        )
        if not np.any(result):
            break
    return result


def _clamped_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    left = max(0, min(width - 1, int(np.floor(min(x0, x1)))))
    top = max(0, min(height - 1, int(np.floor(min(y0, y1)))))
    right = max(left + 1, min(width, int(np.ceil(max(x0, x1)))))
    bottom = max(top + 1, min(height, int(np.ceil(max(y0, y1)))))
    return left, top, right, bottom
