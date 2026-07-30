"""Define run-level observation choices independently of simulator deployment.

``MapPackage`` answers where and how a simulator is reached.  This module
answers which canonical observations a Benchmark, Runtime, or Demo wants for
one run.  Keeping the two configurations separate lets the same deployed scene
serve different evaluation protocols without embedding benchmark names or
episode truth in the simulator layer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODALITIES = ("rgb", "depth", "pose")
DEFAULT_CAPTURE_SOURCES = {
    "rgb": "scene",
    "depth": "depth_perspective",
    "segmentation": "segmentation",
    "mask": "segmentation",
}


@dataclass(frozen=True)
class ObservationProfile:
    """Describe canonical modalities and their simulator capture sources.

    Capture-source values deliberately remain strings here.  The concrete
    Backend owns their meaning and validation, so adding a future simulator
    does not require teaching EnvClient its SDK enums.
    """

    modalities: tuple[str, ...] = DEFAULT_MODALITIES
    capture_sources: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CAPTURE_SOURCES)
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationProfile":
        """Validate a user-facing run profile without assuming a Backend."""

        raw_modalities = data.get("modalities", DEFAULT_MODALITIES)
        if not isinstance(raw_modalities, (list, tuple)) or not raw_modalities:
            raise ValueError("ObservationProfile modalities must be a non-empty list")
        modalities = tuple(str(value).strip().lower() for value in raw_modalities)
        if any(not value for value in modalities):
            raise ValueError("ObservationProfile modalities cannot contain empty values")
        if len(set(modalities)) != len(modalities):
            raise ValueError("ObservationProfile modalities must be unique")

        raw_sources = data.get("capture_sources", {})
        if not isinstance(raw_sources, dict):
            raise ValueError("ObservationProfile capture_sources must be an object")
        capture_sources = dict(DEFAULT_CAPTURE_SOURCES)
        for modality, source in raw_sources.items():
            key = str(modality).strip().lower()
            value = str(source).strip().lower()
            if not key or not value:
                raise ValueError(
                    "ObservationProfile capture_sources keys and values cannot be empty"
                )
            capture_sources[key] = value
        return cls(modalities=modalities, capture_sources=capture_sources)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""

        payload = asdict(self)
        payload["modalities"] = list(self.modalities)
        return payload


def load_observation_profile(
    path: str | Path | None,
) -> ObservationProfile:
    """Load a UTF-8 JSON profile, or return the generic default profile."""

    if path is None:
        return ObservationProfile()
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"Observation profile not found: {profile_path}")
    payload = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Observation profile root must be a JSON object")
    return ObservationProfile.from_dict(payload)
