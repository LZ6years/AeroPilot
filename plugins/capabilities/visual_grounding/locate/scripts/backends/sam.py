"""SAM adapter isolated inside the visual-grounding capability."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from typing import Any, Sequence

import numpy as np
import threading
from src.perception.inference_resources import serialized_inference
from PIL import Image

from plugins.capabilities.visual_grounding.locate.scripts.contracts import CandidateMask, GroundedCandidate


class SamBackend:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.requested_device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._device_lock = threading.RLock()

    def segment(self, rgb_path: Path | BinaryIO, candidates: Sequence[GroundedCandidate]) -> list[CandidateMask]:
        if not candidates:
            return []
        return self._segment(rgb_path, candidates)

    @serialized_inference
    def _segment(self, rgb_path: Path | BinaryIO, candidates: Sequence[GroundedCandidate]) -> list[CandidateMask]:
        torch = self._torch()
        processor, model = self._load()
        with Image.open(rgb_path) as source:
            image = source.convert("RGB")
        inputs = processor(images=image, input_boxes=[[list(item.box_xyxy) for item in candidates]], return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        scores = getattr(outputs, "iou_scores", None)
        scores = scores.detach().cpu().numpy() if scores is not None else None
        results = []
        for index, item in enumerate(masks):
            raw = item.numpy() if hasattr(item, "numpy") else np.asarray(item)
            raw = np.asarray(raw)
            selected = 0
            if raw.ndim == 3 and scores is not None:
                selected = int(np.argmax(scores[0, index]))
            mask = raw[selected] if raw.ndim == 3 else raw
            confidence = float(scores[0, index, selected]) if scores is not None else None
            results.append(CandidateMask(index, np.asarray(mask, dtype=bool), confidence))
        return results

    @property
    def device(self) -> str:
        with self._device_lock:
            if self._device is None:
                resolved = self.requested_device if self.requested_device != "auto" else ("cuda" if self._torch().cuda.is_available() else "cpu")
                self._device = f"cuda:{self._torch().cuda.current_device()}" if resolved == "cuda" else resolved
            return self._device

    @serialized_inference
    def _load(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            try:
                from transformers import SamModel, SamProcessor
            except ModuleNotFoundError as exc:
                raise RuntimeError("SAM backend requires transformers") from exc
            processor = SamProcessor.from_pretrained(self.model_name)
            model = SamModel.from_pretrained(self.model_name)
            model.eval()
            model.to(self.device)
            self._processor, self._model = processor, model
        return self._processor, self._model

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("SAM backend requires torch") from exc
        return torch
