"""GroundingDINO adapter isolated inside the visual-grounding capability."""

from __future__ import annotations

import inspect
import threading
from src.perception.inference_resources import serialized_inference
from pathlib import Path
from typing import BinaryIO
from typing import Any

from PIL import Image

from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.contracts import GroundedCandidate


class GroundingDinoBackend:
    def __init__(self, model_name: str, device: str, *, box_threshold: float, text_threshold: float) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._device_lock = threading.RLock()

    @serialized_inference
    def locate(self, rgb_path: Path | BinaryIO, query: str) -> list[GroundedCandidate]:
        torch = self._torch()
        processor, model = self._load()
        with Image.open(rgb_path) as source:
            image = source.convert("RGB")
        inputs = processor(images=image, text=query.rstrip(". ") + ".", return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        post_process = getattr(processor, "post_process_grounded_object_detection", None)
        if post_process is None:
            post_process = getattr(processor, "post_process_grounded_object_detection_with_logits", None)
        if post_process is None:
            raise RuntimeError("GroundingDINO processor does not expose grounded object post-processing")
        kwargs: dict[str, Any] = {"text_threshold": self.text_threshold, "target_sizes": torch.tensor([image.size[::-1]], device=self.device)}
        if "box_threshold" in inspect.signature(post_process).parameters:
            kwargs["box_threshold"] = self.box_threshold
        else:
            kwargs["threshold"] = self.box_threshold
        result = post_process(outputs, inputs.get("input_ids"), **kwargs)[0]
        labels = result.get("text_labels") if "text_labels" in result else result.get("labels", [])
        candidates = []
        for box, score, label in zip(result.get("boxes", []), result.get("scores", []), labels):
            raw_box = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            raw_score = score.detach().cpu().item() if hasattr(score, "detach") else score
            candidates.append(GroundedCandidate(tuple(float(value) for value in raw_box[:4]), str(label), float(raw_score)))
        return candidates

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
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            except ModuleNotFoundError as exc:
                raise RuntimeError("GroundingDINO backend requires transformers") from exc
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name)
            model.eval()
            model.to(self.device)
            self._processor, self._model = processor, model
        return self._processor, self._model

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("GroundingDINO backend requires torch") from exc
        return torch
