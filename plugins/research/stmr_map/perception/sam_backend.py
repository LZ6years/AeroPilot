"""SAM box-prompt segmentation backend for STMR masks."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from plugins.research.stmr_map.profiling import StageContext, StageProfiler

from .grounding_dino_backend import DetectionResult


DEFAULT_SAM_MODEL = "facebook/sam-vit-base"


@dataclass(frozen=True)
class SamMaskResult:
    """A binary SAM mask associated with one grounded detection."""

    detection: DetectionResult
    mask: np.ndarray
    iou_score: float | None = None


class SamBackend:
    """Run Hugging Face SAM using GroundingDINO boxes as prompts."""

    def __init__(
        self,
        model_name: str = DEFAULT_SAM_MODEL,
        device: str = "auto",
        *,
        profiler: StageProfiler | None = None,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.profiler = profiler
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def warmup(self, *, profiler: StageProfiler | None = None) -> None:
        """Load the processor and model once for reuse across cases."""

        self._load(profiler=profiler)

    def segment(
        self,
        rgb_path: str | Path,
        detections: list[DetectionResult],
        *,
        profiler: StageProfiler | None = None,
    ) -> list[SamMaskResult]:
        """Generate one binary mask per detection box."""

        if not detections:
            return []
        profiler = profiler or self.profiler
        torch = self._torch()
        processor, model = self._load(profiler=profiler)
        with _profile(profiler, "sam_preprocess", detection_count=len(detections)) as stage:
            image = Image.open(rgb_path).convert("RGB")
            boxes = [[list(item.box_xyxy) for item in detections]]
            inputs = processor(images=image, input_boxes=boxes, return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            stage.update(image_width=image.width, image_height=image.height)
        with _profile(profiler, "sam_inference", detection_count=len(detections)):
            with torch.no_grad():
                outputs = model(**inputs)
        with _profile(profiler, "sam_mask_postprocess", detection_count=len(detections)) as stage:
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
            )[0]
            iou_scores = getattr(outputs, "iou_scores", None)
            if iou_scores is not None:
                iou_scores = iou_scores.detach().cpu().numpy()
            results: list[SamMaskResult] = []
            for index, detection in enumerate(detections):
                item_masks = masks[index]
                if hasattr(item_masks, "numpy"):
                    item_masks = item_masks.numpy()
                item_masks = np.asarray(item_masks)
                if item_masks.ndim == 3:
                    mask_index = 0
                    if iou_scores is not None and index < iou_scores.shape[1]:
                        mask_index = int(np.argmax(iou_scores[0, index]))
                    mask = item_masks[mask_index]
                    score = float(iou_scores[0, index, mask_index]) if iou_scores is not None else None
                else:
                    mask = item_masks
                    score = None
                results.append(SamMaskResult(detection=detection, mask=np.asarray(mask, dtype=bool), iou_score=score))
            stage.update(mask_count=len(results))
        return results

    def _load(self, *, profiler: StageProfiler | None = None) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        profiler = profiler or self.profiler
        with _profile(profiler, "sam_model_load", cold_start=True, model_name=self.model_name):
            try:
                from transformers import SamModel, SamProcessor
            except ModuleNotFoundError as exc:
                raise RuntimeError("SAM backend requires transformers") from exc
            self._processor = SamProcessor.from_pretrained(self.model_name)
            self._model = SamModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
        return self._processor, self._model

    def _resolve_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        torch = self._torch()
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("SAM backend requires torch") from exc
        return torch


def _profile(
    profiler: StageProfiler | None,
    stage: str,
    *,
    cold_start: bool = False,
    **metadata: Any,
) -> Any:
    if profiler is None:
        return nullcontext(StageContext(metadata))
    return profiler.stage(stage, cold_start=cold_start, metadata=metadata)
