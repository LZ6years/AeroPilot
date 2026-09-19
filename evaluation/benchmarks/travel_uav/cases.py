"""Immutable TravelUAV case and dataset-provenance schemas.

Cases are loaded from the compact index produced by :mod:`dataset`.  Hidden
goal, reference-path, and spawn facts are intentionally kept out of the
planner-safe metadata projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TravelUavTargetObject:
    """One episode-specific object placement in AirSim NED coordinates."""

    asset_name: str
    object_name: str
    position_ned: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale: tuple[float, float, float]
    physics_enabled: bool = False
    is_blueprint: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TravelUavTargetObject":
        return cls(
            asset_name=_text(data.get("asset_name"), "target.asset_name"),
            object_name=_text(data.get("object_name"), "target.object_name"),
            position_ned=_floats(data.get("position_ned"), 3, "target.position_ned"),
            rotation_xyzw=_floats(
                data.get("rotation_xyzw"), 4, "target.rotation_xyzw"
            ),
            scale=_positive_floats(data.get("scale"), 3, "target.scale"),
            physics_enabled=bool(data.get("physics_enabled", False)),
            is_blueprint=bool(data.get("is_blueprint", False)),
        )


@dataclass(frozen=True)
class TravelUavCase:
    """One deduplicated TravelUAV episode in official NED conventions."""

    episode_id: str
    sequence_id: str
    map_name: str
    split: str
    instruction: str
    raw_instruction: str
    start_position_ned: tuple[float, float, float]
    start_rotation_xyzw: tuple[float, float, float, float]
    reference_path_ned: tuple[tuple[float, float, float], ...]
    reference_path_length_m: float
    target: TravelUavTargetObject
    source_json: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TravelUavCase":
        path_value = data.get("reference_path_ned")
        if not isinstance(path_value, list) or not path_value:
            raise ValueError("TravelUAV reference_path_ned must be non-empty")
        path = tuple(
            _floats(row, 3, "reference_path_ned row") for row in path_value
        )
        length = float(data.get("reference_path_length_m", -1.0))
        if length < 0.0:
            raise ValueError("TravelUAV reference_path_length_m must be non-negative")
        return cls(
            episode_id=_text(data.get("episode_id"), "episode_id"),
            sequence_id=_text(data.get("sequence_id"), "sequence_id"),
            map_name=_text(data.get("map_name"), "map_name"),
            split=_text(data.get("split"), "split"),
            instruction=_text(data.get("instruction"), "instruction"),
            raw_instruction=_text(
                data.get("raw_instruction", data.get("instruction")),
                "raw_instruction",
            ),
            start_position_ned=_floats(
                data.get("start_position_ned"), 3, "start_position_ned"
            ),
            start_rotation_xyzw=_floats(
                data.get("start_rotation_xyzw"), 4, "start_rotation_xyzw"
            ),
            reference_path_ned=path,
            reference_path_length_m=length,
            target=TravelUavTargetObject.from_dict(_object(data.get("target"), "target")),
            source_json=_text(data.get("source_json"), "source_json"),
        )

    @property
    def difficulty(self) -> str:
        return "easy" if self.reference_path_length_m <= 250.0 else "hard"

    def planner_safe_metadata(self) -> dict[str, Any]:
        """Return identity fields only; no target or route truth is exposed."""

        return {
            "episode_id": self.episode_id,
            "sequence_id": self.sequence_id,
            "map_name": self.map_name,
            "dataset_split": self.split,
        }

    def evaluation_truth(self) -> dict[str, Any]:
        return {
            "target": asdict(self.target),
            "reference_path_ned": [list(point) for point in self.reference_path_ned],
            "reference_path_length_m": self.reference_path_length_m,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["difficulty"] = self.difficulty
        return data


@dataclass(frozen=True)
class TravelUavDatasetProvenance:
    """Identity of one compact index and its selected source split."""

    dataset_version: str
    index_file: str
    index_sha256: str
    source_split_file: str
    source_split_sha256: str
    split: str
    episode_count: int
    map_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["map_names"] = list(self.map_names)
        return data


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"TravelUAV {label} must be an object")
    return dict(value)


def _text(value: Any, label: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"TravelUAV {label} cannot be empty")
    return text


def _floats(value: Any, length: int, label: str) -> tuple[float, ...]:
    import math

    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"TravelUAV {label} must contain {length} values")
    parsed = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"TravelUAV {label} values must be finite")
    return parsed


def _positive_floats(value: Any, length: int, label: str) -> tuple[float, ...]:
    parsed = _floats(value, length, label)
    if any(item <= 0.0 for item in parsed):
        raise ValueError(f"TravelUAV {label} values must be positive")
    return parsed
