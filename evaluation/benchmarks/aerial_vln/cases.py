"""Load AerialVLN episodes without importing AirSim or Habitat."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AerialVlnCase:
    """One AerialVLN episode in its original dataset conventions."""

    episode_id: str
    trajectory_id: str
    scene_id: str
    instruction: str
    start_position: tuple[float, float, float]
    start_rotation_wxyz: tuple[float, float, float, float]
    goal_position: tuple[float, float, float]
    reference_path: tuple[tuple[float, ...], ...]
    official_actions: tuple[Any, ...]
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AerialVlnCase":
        instruction_data = data.get("instruction")
        instruction = (
            instruction_data.get("instruction_text")
            if isinstance(instruction_data, dict)
            else instruction_data
        )
        goals = data.get("goals")
        goal = goals[0] if isinstance(goals, list) and goals else {}
        return cls(
            episode_id=_required_text(data, "episode_id"),
            trajectory_id=_required_text(data, "trajectory_id"),
            scene_id=_required_text(data, "scene_id"),
            instruction=_required_value_text(instruction, "instruction_text"),
            start_position=_fixed_floats(
                data.get("start_position"), 3, "start_position"
            ),
            start_rotation_wxyz=_fixed_floats(
                data.get("start_rotation"), 4, "start_rotation"
            ),
            goal_position=_fixed_floats(
                goal.get("position") if isinstance(goal, dict) else None,
                3,
                "goals[0].position",
            ),
            reference_path=_path(data.get("reference_path")),
            official_actions=tuple(data.get("actions") or ()),
            raw=dict(data),
        )

    def planner_safe_metadata(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "scene_id": self.scene_id,
        }

    def evaluation_truth(self) -> dict[str, Any]:
        return {
            "goal_position": list(self.goal_position),
            "reference_path": [list(row) for row in self.reference_path],
            "official_actions": list(self.official_actions),
        }

    def to_dict(self, *, include_evaluation_truth: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_evaluation_truth:
            for key in (
                "goal_position",
                "reference_path",
                "official_actions",
                "raw",
            ):
                data.pop(key, None)
        return data


def load_aerial_vln_cases(
    dataset_root: str | Path,
    split: str,
    *,
    max_cases: int | None = None,
    scene_filter: list[str] | tuple[str, ...] | None = None,
) -> list[AerialVlnCase]:
    """Load one split from JSON or JSON.GZ and apply stable filters."""

    path = resolve_split_path(dataset_root, split)
    payload = _read_json(path)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list):
        raise ValueError(f"AerialVLN split {path} requires an episodes list")
    allowed = {str(value) for value in (scene_filter or ())}
    cases = [
        AerialVlnCase.from_dict(dict(item))
        for item in episodes
        if isinstance(item, dict)
        and (not allowed or str(item.get("scene_id")) in allowed)
    ]
    if max_cases is not None:
        cases = cases[: max(0, int(max_cases))]
    return cases


def resolve_split_path(dataset_root: str | Path, split: str) -> Path:
    root = Path(dataset_root)
    candidates = (
        root / f"{split}.json.gz",
        root / f"{split}.json",
        root / split / f"{split}.json.gz",
        root / split / f"{split}.json",
        root / split / "episodes.json.gz",
        root / split / "episodes.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"AerialVLN split {split!r} not found under {root}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _required_text(data: dict[str, Any], key: str) -> str:
    return _required_value_text(data.get(key), key)


def _required_value_text(value: Any, label: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"AerialVLN episode missing {label}")
    return text


def _fixed_floats(
    value: Any, length: int, label: str
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(
            f"AerialVLN {label} must contain exactly {length} values"
        )
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AerialVLN {label} values must be numeric") from exc


def _path(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("AerialVLN reference_path must be a non-empty list")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            raise ValueError(
                "AerialVLN reference_path rows require at least XYZ"
            )
        try:
            rows.append(tuple(float(item) for item in row))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "AerialVLN reference_path values must be numeric"
            ) from exc
    return tuple(rows)
