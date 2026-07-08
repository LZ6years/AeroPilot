"""Generate dense semantic masks from RGB frames for custom_map.

This module turns closed-set segmentation predictions into the same RGB mask
and color label-map contract consumed by the existing custom_map projection
code. It does not write memory by itself; segmentation_writer owns persistence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .segformer_backend import DEFAULT_SEGFORMER_MODEL, SegFormerBackend, SegmentationPrediction
from .semantic_palette import color_key, semantic_class_for_ade_label, semantic_label_map


class SegmentationBackend(Protocol):
    def predict(self, rgb_path: str | Path) -> SegmentationPrediction:
        ...


@dataclass
class ClosedSetSegmentationConfig:
    model_name: str = DEFAULT_SEGFORMER_MODEL
    device: str = "auto"
    source: str = "segformer_ade20k"
    overlay_alpha: float = 0.48
    mask_dir_name: str = "semantic_mask"
    overlay_dir_name: str = "semantic_overlay"


@dataclass
class ClosedSetSegmentationResult:
    observation_id: str | None
    rgb_path: Path
    mask_path: Path
    overlay_path: Path
    label_map: dict[str, dict[str, Any]]
    stats: dict[str, Any]


def run_closed_set_segmentation(
    rgb_path: str | Path,
    output_dir: str | Path,
    *,
    observation_id: str | None = None,
    config: ClosedSetSegmentationConfig | None = None,
    backend: SegmentationBackend | None = None,
) -> ClosedSetSegmentationResult:
    config = config or ClosedSetSegmentationConfig()
    rgb_path = Path(rgb_path)
    output_dir = Path(output_dir)
    mask_dir = output_dir / config.mask_dir_name
    overlay_dir = output_dir / config.overlay_dir_name
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    backend = backend or SegFormerBackend(config.model_name, config.device)
    prediction = backend.predict(rgb_path)

    rgb_image = Image.open(rgb_path).convert("RGB")
    class_ids = _ensure_image_size(prediction.class_ids, rgb_image.size)
    id_to_label = prediction.id_to_label
    mask_array, stats = _semantic_mask_array(class_ids, id_to_label, config.source)

    stem = _artifact_stem(rgb_path, observation_id)
    mask_path = mask_dir / f"{stem}.png"
    overlay_path = overlay_dir / f"{stem}.png"
    Image.fromarray(mask_array, mode="RGB").save(mask_path)
    _overlay(rgb_image, mask_array, config.overlay_alpha).save(overlay_path)

    stats.update(
        {
            "source": config.source,
            "model_name": prediction.model_name,
            "device": prediction.device,
            "rgb_path": str(rgb_path),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "observation_id": observation_id,
        }
    )
    return ClosedSetSegmentationResult(
        observation_id=observation_id,
        rgb_path=rgb_path,
        mask_path=mask_path,
        overlay_path=overlay_path,
        label_map=semantic_label_map(config.source),
        stats=stats,
    )


def _semantic_mask_array(
    class_ids: np.ndarray,
    id_to_label: dict[int, str],
    source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = class_ids.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)
    custom_counts: Counter[str] = Counter()
    ade_counts: Counter[str] = Counter()
    for ade_id in np.unique(class_ids):
        label = id_to_label.get(int(ade_id), f"ade_class_{int(ade_id)}")
        semantic = semantic_class_for_ade_label(label)
        mask = class_ids == int(ade_id)
        output[mask] = semantic.color
        count = int(mask.sum())
        ade_counts[label] += count
        custom_counts[semantic.label] += count
    return output, {
        "pixel_count": int(height * width),
        "custom_class_pixel_counts": dict(sorted(custom_counts.items())),
        "ade_label_pixel_counts": dict(sorted(ade_counts.items())),
        "label_map_source": source,
        "mask_colors": sorted({color_key(tuple(int(channel) for channel in pixel)) for row in output for pixel in row}),
    }


def _ensure_image_size(class_ids: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if class_ids.shape == (height, width):
        return class_ids.astype(np.int32)
    image = Image.fromarray(class_ids.astype(np.int32), mode="I")
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.int32)


def _overlay(rgb_image: Image.Image, mask_array: np.ndarray, alpha: float) -> Image.Image:
    mask_image = Image.fromarray(mask_array, mode="RGB")
    return Image.blend(rgb_image, mask_image, max(0.0, min(1.0, alpha)))


def _artifact_stem(rgb_path: Path, observation_id: str | None) -> str:
    safe_observation = (observation_id or "").replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"{safe_observation}_{rgb_path.stem}" if safe_observation else rgb_path.stem
