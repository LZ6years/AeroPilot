"""Orchestrate replaceable grounding, segmentation, and metric-depth stages."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from src.perception.frame_payload import frame_bytes, load_rgb_frame

from plugins.capabilities.visual_grounding.locate.scripts.backends.grounding_dino import (
    GroundingDinoBackend,
)
from plugins.capabilities.visual_grounding.locate.scripts.backends.sam import SamBackend
from plugins.capabilities.visual_grounding.locate.scripts.contracts import (
    CandidateMask,
    GroundedCandidate,
    GroundingBackend,
    SegmentationBackend,
    VisualGroundingSettings,
)
from plugins.capabilities.visual_grounding.locate.scripts.depth import (
    _clamped_box,
    estimate_candidate_depth,
    load_metric_depth,
)


class VisualGroundingService:
    """Keep model-specific work behind capability-local protocols."""

    def __init__(
        self,
        grounding_backend: GroundingBackend,
        segmentation_backend: SegmentationBackend | None,
        settings: VisualGroundingSettings,
    ) -> None:
        self.grounding_backend = grounding_backend
        self.segmentation_backend = segmentation_backend
        self.settings = settings

    def locate(
        self,
        *,
        observation: dict[str, Any],
        observation_id: str,
        query: str,
        include_distance: bool,
        run_dir: str | None,
        tool_call_id: str | None,
    ) -> dict[str, Any]:
        latest_id = str(observation.get("observation_id") or "")
        if observation_id != latest_id:
            raise ValueError(
                "visual grounding only accepts the latest observation_id; "
                f"requested={observation_id!r}, latest={latest_id!r}"
            )
        frames = observation.get("frames")
        frames = frames if isinstance(frames, dict) else {}
        rgb_frame = frames.get("rgb")
        if not isinstance(rgb_frame, dict):
            raise ValueError("latest observation has no RGB frame")
        image = load_rgb_frame(rgb_frame)
        width, height = image.size
        # PIL-backed detectors accept a seekable stream as well as a path.
        # Keep memory-only observations in memory, including segmentation.
        rgb_path = BytesIO(frame_bytes(rgb_frame))

        raw_candidates = self.grounding_backend.locate(rgb_path, query)
        # Replacement backends obey the same JSON-configured public threshold.
        eligible = [c for c in raw_candidates if np.isfinite(c.confidence)
                    and c.confidence >= self.settings.box_threshold]
        candidates = sorted(
            eligible,
            key=lambda item: item.confidence,
            reverse=True,
        )[: self.settings.max_candidates]
        masks: dict[int, CandidateMask] = {}
        segmentation_error: str | None = None
        depth: np.ndarray | None = None
        depth_frame: dict[str, Any] | None = None
        depth_error: str | None = None
        if include_distance and candidates:
            if self.segmentation_backend is not None:
                try:
                    masks = {
                        item.candidate_index: item
                        for item in self.segmentation_backend.segment(rgb_path, candidates)
                    }
                except Exception as exc:
                    # A detector box remains a valid conservative fallback.
                    segmentation_error = f"{type(exc).__name__}: {exc}"
            raw_depth_frame = frames.get("depth")
            if isinstance(raw_depth_frame, dict):
                depth_frame = raw_depth_frame
                try:
                    depth = load_metric_depth(raw_depth_frame)
                    if depth.shape != (height, width):
                        raise ValueError("RGB and metric-depth frame dimensions do not match")
                except Exception as exc:
                    depth_error = f"{type(exc).__name__}: {exc}"
                    depth = None
            else:
                depth_error = "latest observation has no metric-depth frame"

        model_candidates: list[dict[str, Any]] = []
        audit_candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            candidate_id = f"candidate_{index + 1:02d}"
            box = _clamped_box(candidate.box_xyxy, width, height)
            normalized = [
                round(box[0] / width, 4),
                round(box[1] / height, 4),
                round(box[2] / width, 4),
                round(box[3] / height, 4),
            ]
            model_item: dict[str, Any] = {
                "candidate_id": candidate_id,
                "label": candidate.label,
                "confidence": round(float(candidate.confidence), 3),
                "box_normalized": normalized,
                "distance_m": None,
                "distance_status": "not_requested",
            }
            audit_item: dict[str, Any] = {
                "candidate_id": candidate_id,
                "label": candidate.label,
                "confidence": float(candidate.confidence),
                "box_pixels": list(box),
            }
            if include_distance:
                if depth is None:
                    model_item["distance_status"] = "metric_depth_unavailable"
                    audit_item["depth_error"] = depth_error
                else:
                    estimate = estimate_candidate_depth(
                        depth,
                        candidate,
                        masks.get(index),
                        self.settings,
                    )
                    model_item["distance_m"] = estimate.distance_m
                    model_item["distance_status"] = estimate.status
                    audit_item.update(estimate.audit)
            model_candidates.append(model_item)
            audit_candidates.append(audit_item)

        artifact_refs, model_inputs = _write_candidate_evidence(
            rgb_path=rgb_path,
            run_dir=run_dir,
            observation_id=observation_id,
            tool_call_id=tool_call_id,
            candidates=candidates,
            masks=masks,
        )
        return {
            "data": {
                "observation_id": observation_id,
                "query": query,
                "include_distance": include_distance,
                "candidates": model_candidates,
                "audit_details": {
                    "box_threshold": self.settings.box_threshold,
                    "backend_candidate_count": len(raw_candidates),
                    "filtered_candidate_count": len(raw_candidates) - len(eligible),
                    "rgb_frame_id": rgb_frame.get("id"),
                    "depth_frame_id": depth_frame.get("id") if depth_frame else None,
                    "grounding_backend": self.settings.grounding_backend,
                    "segmentation_backend": self.settings.segmentation_backend,
                    "segmentation_error": segmentation_error,
                    "depth_error": depth_error,
                    "candidates": audit_candidates,
                },
            },
            "artifact_refs": artifact_refs,
            "model_inputs": model_inputs,
        }


_SERVICE_CACHE: dict[str, VisualGroundingService] = {}


def service_from_settings(raw_settings: dict[str, Any]) -> VisualGroundingService:
    """Build one cached default service solely from capability.json settings."""

    cache_key = json.dumps(raw_settings, sort_keys=True, ensure_ascii=False)
    cached = _SERVICE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    settings = VisualGroundingSettings.from_dict(raw_settings)
    if settings.grounding_backend != "grounding_dino":
        raise ValueError(f"unsupported grounding backend: {settings.grounding_backend}")
    grounding = GroundingDinoBackend(
        settings.grounding_model,
        settings.device,
        box_threshold=settings.box_threshold,
        text_threshold=settings.text_threshold,
    )
    segmentation: SegmentationBackend | None
    if settings.segmentation_backend == "sam":
        segmentation = SamBackend(settings.segmentation_model, settings.device)
    elif settings.segmentation_backend in {"", "none"}:
        segmentation = None
    else:
        raise ValueError(f"unsupported segmentation backend: {settings.segmentation_backend}")
    service = VisualGroundingService(grounding, segmentation, settings)
    _SERVICE_CACHE[cache_key] = service
    return service


def _write_candidate_evidence(
    *,
    rgb_path: Path | BinaryIO,
    run_dir: str | None,
    observation_id: str,
    tool_call_id: str | None,
    candidates: list[GroundedCandidate],
    masks: dict[int, CandidateMask],
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    if not run_dir or not candidates:
        return (), ()
    with Image.open(rgb_path) as source:
        image = source.convert("RGB")
    overlay = image.copy()
    pixels = np.asarray(overlay).copy()
    colors = [(255, 40, 70), (30, 180, 255), (255, 180, 30), (120, 230, 90), (190, 90, 255)]
    for index, mask_result in masks.items():
        if index >= len(candidates):
            continue
        mask = np.asarray(mask_result.mask, dtype=bool)
        if mask.shape != pixels.shape[:2]:
            continue
        color = np.asarray(colors[index % len(colors)], dtype=np.float32)
        pixels[mask] = np.asarray(pixels[mask] * 0.55 + color * 0.45, dtype=np.uint8)
    overlay = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(min(image.size) / 100))
    for index, candidate in enumerate(candidates):
        color = colors[index % len(colors)]
        x0, y0, x1, y1 = _clamped_box(candidate.box_xyxy, *image.size)
        draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=color, width=line_width)
        draw.text((x0 + line_width, y0 + line_width), f"candidate_{index + 1:02d}", fill=color)
    identity = hashlib.sha256(
        f"{observation_id}|{tool_call_id}|{[(item.box_xyxy, item.label) for item in candidates]}".encode("utf-8")
    ).hexdigest()[:20]
    output_dir = Path(run_dir) / "artifacts" / "tool_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"visual_grounding_{identity}.png"
    overlay.save(output_path, format="PNG")
    path = str(output_path)
    return (
        (path,),
        (
            {
                "type": "image",
                "role": "visual_grounding_candidates",
                "observation_id": observation_id,
                "tool_call_id": str(tool_call_id or ""),
                "artifact_ref": path,
                "path": path,
                "mime_type": "image/png",
                "usage": (
                    "Candidate IDs, detector boxes, and available segmentation masks. "
                    "Confirm which candidate matches the instruction; detector confidence alone is not semantic proof."
                ),
            },
        ),
    )
