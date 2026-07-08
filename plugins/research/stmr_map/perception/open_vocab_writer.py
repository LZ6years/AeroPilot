"""Write GroundingDINO/SAM open-vocabulary masks into run-local memory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from plugins.research.custom_map.perception.semantic_palette import SEMANTIC_CLASSES, color_key, semantic_class_for_ade_label

from .sam_backend import SamMaskResult
from .tfidf_filter import TfidfMatch


@dataclass(frozen=True)
class OpenVocabMaskResult:
    """Artifacts produced by the STMR open-vocabulary perception stage."""

    observation_id: str
    rgb_path: Path
    mask_path: Path
    overlay_path: Path
    boxes_path: Path
    label_map: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def write_open_vocab_mask(
    run_dir: str | Path,
    rgb_path: str | Path,
    observation_id: str,
    masks: list[SamMaskResult],
    matches: list[TfidfMatch],
    *,
    source: str = "stmr.grounding_dino_sam",
    force: bool = True,
    dynamic_target_candidates: list[dict[str, Any]] | None = None,
    rejected_masks: list[dict[str, Any]] | None = None,
    kept_non_primary_masks: list[dict[str, Any]] | None = None,
    primary_dynamic_track: dict[str, Any] | None = None,
) -> OpenVocabMaskResult:
    """Render filtered SAM masks and append a ``stmr_open_vocab_mask`` frame."""

    run_dir = Path(run_dir)
    rgb_path = Path(rgb_path)
    artifacts = run_dir / "artifacts"
    mask_dir = artifacts / "stmr_open_vocab_mask"
    overlay_dir = artifacts / "stmr_open_vocab_overlay"
    box_dir = artifacts / "stmr_grounding_boxes"
    for directory in (mask_dir, overlay_dir, box_dir):
        directory.mkdir(parents=True, exist_ok=True)
    keep_by_index = {item.detection_index: item for item in matches if item.keep}
    rgb = Image.open(rgb_path).convert("RGB")
    mask_image = Image.new("RGB", rgb.size, (0, 0, 0))
    overlay = rgb.copy()
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    boxes_image = rgb.copy()
    boxes_draw = ImageDraw.Draw(boxes_image)
    label_map: dict[str, dict[str, Any]] = {}
    kept: list[dict[str, Any]] = []

    for index, mask_result in enumerate(masks):
        match = keep_by_index.get(index)
        if match is None:
            continue
        semantic = semantic_class_for_ade_label(match.landmark)
        if semantic.semantic_id == SEMANTIC_CLASSES["unknown"].semantic_id:
            semantic = _dynamic_landmark_semantic(match.landmark)
        label_map[color_key(semantic.color)] = {
            "id": semantic.semantic_id,
            "label": semantic.label,
            "category": semantic.category,
            "source": source,
            "confidence": "grounding_dino_sam_tfidf",
            "object_name": match.landmark,
        }
        _paint_mask(mask_image, mask_result.mask, semantic.color)
        _paint_overlay(overlay_draw, mask_result.mask, semantic.color)
        x0, y0, x1, y1 = mask_result.detection.box_xyxy
        boxes_draw.rectangle((x0, y0, x1, y1), outline=semantic.color, width=3)
        boxes_draw.text((x0, max(0, y0 - 12)), match.landmark, fill=semantic.color)
        kept.append(
            {
                "phrase": match.phrase,
                "landmark": match.landmark,
                "similarity": match.similarity,
                "score": match.score,
                "box_xyxy": [round(value, 2) for value in mask_result.detection.box_xyxy],
                "semantic_id": semantic.semantic_id,
                "label": semantic.label,
                "iou_score": mask_result.iou_score,
                "mask_pixel_count": int(np.asarray(mask_result.mask, dtype=bool).sum()),
            }
        )

    stem = _artifact_stem(rgb_path, observation_id)
    mask_path = mask_dir / f"{stem}.png"
    overlay_path = overlay_dir / f"{stem}.png"
    boxes_path = box_dir / f"{stem}.png"
    mask_image.save(mask_path)
    overlay.save(overlay_path)
    boxes_image.save(boxes_path)
    metadata = {
        "source": source,
        "rgb_path": str(rgb_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "boxes_path": str(boxes_path),
        "tfidf_matches": [match.__dict__ for match in matches],
        "kept_masks": kept,
        "dynamic_target_candidates": dynamic_target_candidates or [],
        "rejected_masks": rejected_masks or [],
        "kept_non_primary_masks": kept_non_primary_masks or [],
        "primary_dynamic_track": primary_dynamic_track,
        "label_map": label_map,
    }
    if force or not _has_open_vocab_frame(run_dir, observation_id):
        _append_frame(run_dir, observation_id, mask_path, metadata)
    return OpenVocabMaskResult(observation_id, rgb_path, mask_path, overlay_path, boxes_path, label_map, metadata)


def _paint_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (image.height, image.width):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
    pixels = image.load()
    ys, xs = np.where(mask)
    for x, y in zip(xs, ys):
        pixels[int(x), int(y)] = color


def _paint_overlay(draw: ImageDraw.ImageDraw, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    # Drawing every point is slow but acceptable for standalone research probes.
    rgba = (*color, 95)
    for x, y in zip(xs, ys):
        draw.point((int(x), int(y)), fill=rgba)


def _append_frame(run_dir: Path, observation_id: str, mask_path: Path, metadata: dict[str, Any]) -> None:
    memory_dir = run_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    frame_path = memory_dir / "frames.jsonl"
    line_number = len(frame_path.read_text(encoding="utf-8").splitlines()) + 1 if frame_path.exists() else 1
    frame = {
        "id": f"stmr_open_vocab_{line_number:04d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "observation_id": observation_id,
        "modality": "stmr_open_vocab_mask",
        "path": str(mask_path),
        "metadata": metadata,
    }
    with frame_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(frame, ensure_ascii=False) + "\n")


def _has_open_vocab_frame(run_dir: Path, observation_id: str) -> bool:
    path = run_dir / "memory" / "frames.jsonl"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("observation_id") == observation_id and item.get("modality") == "stmr_open_vocab_mask":
            return True
    return False


def _dynamic_landmark_semantic(label: str) -> Any:
    semantic_id = 100 + (sum(ord(char) for char in label) % 900)
    color = (
        80 + semantic_id * 37 % 150,
        80 + semantic_id * 53 % 150,
        80 + semantic_id * 71 % 150,
    )
    return type(
        "DynamicSemantic",
        (),
        {
            "semantic_id": semantic_id,
            "label": label,
            "category": "instruction_landmark",
            "color": color,
        },
    )()


def _artifact_stem(rgb_path: Path, observation_id: str) -> str:
    safe_observation = observation_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{safe_observation}_{rgb_path.stem}"
