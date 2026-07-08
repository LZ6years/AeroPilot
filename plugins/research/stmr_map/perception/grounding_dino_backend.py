"""GroundingDINO backend for STMR open-vocabulary detection."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"


@dataclass(frozen=True)
class DetectionResult:
    """A grounded phrase box in absolute pixel coordinates."""

    box_xyxy: tuple[float, float, float, float]
    phrase: str
    score: float
    label: str | None = None


class GroundingDinoBackend:
    """Run Hugging Face GroundingDINO zero-shot object detection."""

    def __init__(
        self,
        model_name: str = DEFAULT_GROUNDING_DINO_MODEL,
        device: str = "auto",
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def detect(self, rgb_path: str | Path, prompts: list[str]) -> list[DetectionResult]:
        """Detect all prompt phrases in one RGB image."""

        if not prompts:
            return []
        torch = self._torch()
        processor, model = self._load()
        image = Image.open(rgb_path).convert("RGB")
        text = ". ".join(prompts) + "."
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        post_process = getattr(processor, "post_process_grounded_object_detection", None)
        if post_process is None:
            post_process = getattr(processor, "post_process_grounded_object_detection_with_logits", None)
        if post_process is None:
            raise RuntimeError("GroundingDINO processor does not expose grounded object post-processing")
        kwargs = {
            "text_threshold": self.text_threshold,
            "target_sizes": target_sizes,
        }
        if "box_threshold" in inspect.signature(post_process).parameters:
            kwargs["box_threshold"] = self.box_threshold
        else:
            kwargs["threshold"] = self.box_threshold
        results = post_process(outputs, inputs.get("input_ids"), **kwargs)[0]
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("labels", results.get("text_labels", []))
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            raw_box = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            detections.append(
                DetectionResult(
                    box_xyxy=tuple(float(value) for value in raw_box[:4]),  # type: ignore[arg-type]
                    phrase=str(label),
                    score=float(score.detach().cpu().item() if hasattr(score, "detach") else score),
                    label=str(label),
                )
            )
        return detections

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError("GroundingDINO backend requires transformers") from exc
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name)
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
            raise RuntimeError("GroundingDINO backend requires torch") from exc
        return torch
