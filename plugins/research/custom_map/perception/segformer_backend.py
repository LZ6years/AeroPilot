"""Hugging Face SegFormer backend for closed-set custom_map preprocessing.

The backend is optional and imported lazily. Unit tests can inject a fake
backend, while manual probes can use this class to run real GPU inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


@dataclass
class SegmentationPrediction:
    class_ids: np.ndarray
    id_to_label: dict[int, str]
    device: str
    model_name: str


class SegFormerBackend:
    """Run SegFormer semantic segmentation on RGB images."""

    def __init__(self, model_name: str = DEFAULT_SEGFORMER_MODEL, device: str = "auto") -> None:
        self.model_name = model_name
        self.requested_device = device
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
        image = Image.open(rgb_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_size = [image.size[::-1]]
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
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ModuleNotFoundError as exc:
            raise RuntimeError("SegFormer preprocessing requires transformers to be installed") from exc

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
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
            raise RuntimeError("SegFormer preprocessing requires torch to be installed") from exc
        return torch
