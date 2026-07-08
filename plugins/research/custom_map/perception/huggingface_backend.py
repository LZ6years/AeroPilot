"""Generic Hugging Face semantic segmentation backend for custom_map probes.

This backend keeps the same SegmentationPrediction contract as SegFormerBackend
while allowing manual research probes to compare Cityscapes and Mapillary
Mask2Former models without changing the custom_map projection pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .segformer_backend import SegmentationPrediction


CITYSCAPES_SEGFORMER_B2 = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
CITYSCAPES_MASK2FORMER_LARGE = "facebook/mask2former-swin-large-cityscapes-semantic"
MAPILLARY_MASK2FORMER_LARGE = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
ADE_SEGFORMER_B5 = "nvidia/segformer-b5-finetuned-ade-640-640"
ADE_MASK2FORMER_LARGE = "facebook/mask2former-swin-large-ade-semantic"
ADE_ONEFORMER_DINAT_LARGE = "shi-labs/oneformer_ade20k_dinat_large"
CITYSCAPES_ONEFORMER_DINAT_LARGE = "shi-labs/oneformer_cityscapes_dinat_large"
ADE_ONEFORMER_SWIN_LARGE = "shi-labs/oneformer_ade20k_swin_large"
CITYSCAPES_ONEFORMER_SWIN_LARGE = "shi-labs/oneformer_cityscapes_swin_large"


class HuggingFaceSegmentationBackend:
    """Run supported Hugging Face semantic segmentation models on RGB images."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        *,
        resize_long_edge: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.resize_long_edge = resize_long_edge
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def predict(self, rgb_path: str | Path) -> SegmentationPrediction:
        torch = self._torch()
        processor, model = self._load()
        original = Image.open(rgb_path).convert("RGB")
        image = _resize_long_edge(original, self.resize_long_edge)
        if _model_family(self.model_name) == "oneformer":
            inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        else:
            inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_size = [original.size[::-1]]
        class_ids = processor.post_process_semantic_segmentation(outputs, target_sizes=target_size)[0]
        id_to_label = {int(key): str(value) for key, value in getattr(model.config, "id2label", {}).items()}
        return SegmentationPrediction(
            class_ids=class_ids.detach().cpu().numpy().astype(np.int32),
            id_to_label=id_to_label,
            device=self.device,
            model_name=self.model_name,
        )

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import (
                AutoImageProcessor,
                Mask2FormerForUniversalSegmentation,
                OneFormerForUniversalSegmentation,
                OneFormerProcessor,
                SegformerForSemanticSegmentation,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError("Hugging Face segmentation comparison requires transformers") from exc

        family = _model_family(self.model_name)
        if family == "oneformer":
            self._processor = OneFormerProcessor.from_pretrained(self.model_name)
            self._model = OneFormerForUniversalSegmentation.from_pretrained(self.model_name)
        else:
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        if family == "mask2former":
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
        elif family == "segformer":
            self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
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
            raise RuntimeError("Hugging Face segmentation comparison requires torch") from exc
        return torch


def _model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "oneformer" in lowered:
        return "oneformer"
    if "mask2former" in lowered:
        return "mask2former"
    return "segformer"


def _resize_long_edge(image: Image.Image, long_edge: int | None) -> Image.Image:
    if long_edge is None or long_edge <= 0:
        return image
    width, height = image.size
    current_long_edge = max(width, height)
    if current_long_edge <= long_edge:
        return image
    scale = float(long_edge) / float(current_long_edge)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(target, Image.Resampling.BILINEAR)
