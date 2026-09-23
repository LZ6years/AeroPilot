"""Internal replaceable contracts for the visual-grounding capability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class VisualGroundingSettings:
    grounding_backend: str
    grounding_model: str
    segmentation_backend: str
    segmentation_model: str
    device: str
    box_threshold: float
    text_threshold: float
    max_candidates: int
    mask_erosion_ratio: float
    box_inset_ratio: float
    minimum_valid_depth_pixels: int
    minimum_valid_depth_coverage: float
    maximum_coherent_mad_m: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualGroundingSettings":
        """Validate capability-local JSON settings without consulting Runtime."""

        settings = cls(
            grounding_backend=str(data.get("grounding_backend", "")).strip(),
            grounding_model=str(data.get("grounding_model", "")).strip(),
            segmentation_backend=str(data.get("segmentation_backend", "")).strip(),
            segmentation_model=str(data.get("segmentation_model", "")).strip(),
            device=str(data.get("device", "auto")).strip() or "auto",
            box_threshold=float(data.get("box_threshold", 0.25)),
            text_threshold=float(data.get("text_threshold", 0.25)),
            max_candidates=int(data.get("max_candidates", 5)),
            mask_erosion_ratio=float(data.get("mask_erosion_ratio", 0.08)),
            box_inset_ratio=float(data.get("box_inset_ratio", 0.2)),
            minimum_valid_depth_pixels=int(data.get("minimum_valid_depth_pixels", 9)),
            minimum_valid_depth_coverage=float(data.get("minimum_valid_depth_coverage", 0.5)),
            maximum_coherent_mad_m=float(data.get("maximum_coherent_mad_m", 0.25)),
        )
        if not settings.grounding_backend or not settings.grounding_model:
            raise ValueError("grounding backend and model must be configured")
        if settings.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if not 0 <= settings.mask_erosion_ratio < 0.5:
            raise ValueError("mask_erosion_ratio must be in [0, 0.5)")
        if not 0 <= settings.box_inset_ratio < 0.5:
            raise ValueError("box_inset_ratio must be in [0, 0.5)")
        if settings.minimum_valid_depth_pixels < 1:
            raise ValueError("minimum_valid_depth_pixels must be positive")
        if not 0 < settings.minimum_valid_depth_coverage <= 1:
            raise ValueError("minimum_valid_depth_coverage must be in (0, 1]")
        if settings.maximum_coherent_mad_m <= 0:
            raise ValueError("maximum_coherent_mad_m must be positive")
        return settings


@dataclass(frozen=True)
class GroundedCandidate:
    box_xyxy: tuple[float, float, float, float]
    label: str
    confidence: float


@dataclass(frozen=True)
class CandidateMask:
    candidate_index: int
    mask: np.ndarray
    confidence: float | None = None


@dataclass(frozen=True)
class DepthEstimate:
    distance_m: float | None
    status: str
    audit: dict[str, Any]


class GroundingBackend(Protocol):
    def locate(self, rgb_path: Path | BinaryIO, query: str) -> list[GroundedCandidate]: ...


class SegmentationBackend(Protocol):
    def segment(
        self, rgb_path: Path | BinaryIO, candidates: Sequence[GroundedCandidate]
    ) -> list[CandidateMask]: ...
