"""Classify and select AerialVLN episodes using evaluation-only path truth.

This module deliberately stays independent from Runtime and simulator code.
Reference paths, difficulty labels, and sampling metadata are benchmark truth
for reporting and case selection; they must never enter Planner context.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

from evaluation.benchmarks.aerial_vln.cases import AerialVlnCase


DIFFICULTIES = ("low", "medium", "high")


@dataclass(frozen=True)
class AerialVlnDifficultyAssignment:
    """One episode's path-length class within one complete split."""

    episode_id: str
    trajectory_id: str
    scene_id: str
    reference_path_length_m: float
    difficulty: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AerialVlnCaseSelection:
    """Immutable classification population and selected evaluation subset."""

    q33_m: float
    q67_m: float
    population_case_count: int
    population_scene_ids: tuple[str, ...]
    requested_difficulties: tuple[str, ...]
    limit_per_difficulty: int | None
    assignments: tuple[AerialVlnDifficultyAssignment, ...]
    selected_assignments: tuple[AerialVlnDifficultyAssignment, ...]

    @property
    def assignment_by_episode(
        self,
    ) -> dict[str, AerialVlnDifficultyAssignment]:
        return {item.episode_id: item for item in self.assignments}

    @property
    def bucket_counts(self) -> dict[str, int]:
        return {
            label: sum(
                item.difficulty == label for item in self.assignments
            )
            for label in DIFFICULTIES
        }

    @property
    def selected_counts(self) -> dict[str, int]:
        return {
            label: sum(
                item.difficulty == label
                for item in self.selected_assignments
            )
            for label in DIFFICULTIES
        }

    @property
    def selected_scene_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {item.scene_id for item in self.selected_assignments},
                key=_scene_sort_key,
            )
        )

    def assignments_dict(self) -> dict[str, object]:
        """Return the complete population classification artifact."""

        return {
            "method": _classification_method(),
            "thresholds_m": {"q33": self.q33_m, "q67": self.q67_m},
            "population_case_count": self.population_case_count,
            "population_scene_ids": list(self.population_scene_ids),
            "bucket_counts": self.bucket_counts,
            "assignments": [
                item.to_dict() for item in self.assignments
            ],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the selected subset and reproducible sampling policy."""

        selected_by_difficulty = {
            label: [
                item.to_dict()
                for item in self.selected_assignments
                if item.difficulty == label
            ]
            for label in DIFFICULTIES
        }
        return {
            "method": _selection_method(self.limit_per_difficulty),
            "classification_method": _classification_method(),
            "thresholds_m": {"q33": self.q33_m, "q67": self.q67_m},
            "population_case_count": self.population_case_count,
            "population_scene_ids": list(self.population_scene_ids),
            "bucket_counts": self.bucket_counts,
            "requested_difficulties": list(
                self.requested_difficulties
            ),
            "limit_per_difficulty": self.limit_per_difficulty,
            "selected_counts": self.selected_counts,
            "selected_scene_ids": list(self.selected_scene_ids),
            "selected_by_difficulty": selected_by_difficulty,
            "cases_by_episode": {
                item.episode_id: item.to_dict()
                for item in self.selected_assignments
            },
        }


def classify_and_select_aerial_vln_cases(
    cases: Sequence[AerialVlnCase],
    *,
    difficulties: Sequence[str] = DIFFICULTIES,
    limit_per_difficulty: int | None = None,
) -> tuple[list[AerialVlnCase], AerialVlnCaseSelection]:
    """Classify one complete population, then select requested buckets.

    Thresholds always come from every supplied case. Callers therefore own the
    important boundary of passing one complete split population before applying
    difficulty filters or limits. Different splits are intentionally classified
    independently.
    """

    if not cases:
        raise ValueError("AerialVLN difficulty classification requires cases")
    requested = tuple(str(item) for item in difficulties)
    _validate_difficulties(requested)
    if limit_per_difficulty is not None and limit_per_difficulty <= 0:
        raise ValueError("limit_per_difficulty must be positive or null")

    rows = sorted(
        (
            (
                case,
                reference_path_length_m(case.reference_path),
            )
            for case in cases
        ),
        key=lambda item: (
            item[1],
            _scene_sort_key(item[0].scene_id),
            item[0].episode_id,
        ),
    )
    lengths = [row[1] for row in rows]
    q33 = _linear_quantile(lengths, 1.0 / 3.0)
    q67 = _linear_quantile(lengths, 2.0 / 3.0)
    assignments = tuple(
        AerialVlnDifficultyAssignment(
            episode_id=case.episode_id,
            trajectory_id=case.trajectory_id,
            scene_id=case.scene_id,
            reference_path_length_m=length,
            difficulty=_difficulty(length, q33=q33, q67=q67),
        )
        for case, length in rows
    )

    selected_assignments: list[AerialVlnDifficultyAssignment] = []
    for label in requested:
        bucket = tuple(
            item for item in assignments if item.difficulty == label
        )
        if limit_per_difficulty is None:
            chosen = bucket
        else:
            chosen = _scene_balanced_sample(
                bucket,
                min(limit_per_difficulty, len(bucket)),
            )
        selected_assignments.extend(chosen)

    selection = AerialVlnCaseSelection(
        q33_m=q33,
        q67_m=q67,
        population_case_count=len(assignments),
        population_scene_ids=tuple(
            sorted({case.scene_id for case in cases}, key=_scene_sort_key)
        ),
        requested_difficulties=requested,
        limit_per_difficulty=limit_per_difficulty,
        assignments=assignments,
        selected_assignments=tuple(selected_assignments),
    )
    cases_by_episode = {case.episode_id: case for case in cases}
    return (
        [
            cases_by_episode[item.episode_id]
            for item in selection.selected_assignments
        ],
        selection,
    )


def reference_path_length_m(
    points: Iterable[Iterable[float]],
) -> float:
    """Return XYZ length after dropping consecutive duplicate positions."""

    deduplicated: list[tuple[float, float, float]] = []
    for point in points:
        values = tuple(float(value) for value in point)
        if len(values) < 3:
            raise ValueError("reference path rows require XYZ")
        xyz = values[:3]
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("reference path XYZ values must be finite")
        if not deduplicated or xyz != deduplicated[-1]:
            deduplicated.append(xyz)
    if not deduplicated:
        raise ValueError("reference path must contain at least one XYZ point")
    return sum(
        math.dist(previous, current)
        for previous, current in zip(
            deduplicated, deduplicated[1:], strict=False
        )
    )


def _scene_balanced_sample(
    bucket: Sequence[AerialVlnDifficultyAssignment],
    count: int,
) -> tuple[AerialVlnDifficultyAssignment, ...]:
    """Cover scenes first while tracking evenly spaced length percentiles."""

    if count <= 0:
        return ()
    if count >= len(bucket):
        return tuple(bucket)

    target_lengths = [
        _linear_quantile(
            [item.reference_path_length_m for item in bucket],
            (index + 0.5) / count,
        )
        for index in range(count)
    ]
    selected_per_scene: defaultdict[str, int] = defaultdict(int)
    remaining = list(bucket)
    selected: list[AerialVlnDifficultyAssignment] = []
    for target_length in target_lengths:
        minimum_scene_count = min(
            selected_per_scene[item.scene_id] for item in remaining
        )
        eligible = [
            item
            for item in remaining
            if selected_per_scene[item.scene_id] == minimum_scene_count
        ]
        chosen = min(
            eligible,
            key=lambda item: (
                abs(item.reference_path_length_m - target_length),
                _scene_sort_key(item.scene_id),
                item.episode_id,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        selected_per_scene[chosen.scene_id] += 1
    return tuple(selected)


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _difficulty(length: float, *, q33: float, q67: float) -> str:
    if length <= q33:
        return "low"
    if length <= q67:
        return "medium"
    return "high"


def _validate_difficulties(difficulties: tuple[str, ...]) -> None:
    if not difficulties:
        raise ValueError("difficulties cannot be empty")
    unknown = sorted(set(difficulties).difference(DIFFICULTIES))
    if unknown:
        raise ValueError(
            "Unknown AerialVLN difficulties: " + ", ".join(unknown)
        )
    if len(set(difficulties)) != len(difficulties):
        raise ValueError("AerialVLN difficulties must be unique")


def _scene_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    if text.isdigit():
        return 0, int(text)
    return 1, text


def _classification_method() -> str:
    return (
        "remove consecutive duplicate XYZ points, sum Euclidean segments, "
        "compute global linear q33/q67 over the complete input population"
    )


def _selection_method(limit_per_difficulty: int | None) -> str:
    if limit_per_difficulty is None:
        return "all cases from each requested difficulty bucket"
    return (
        "scene-balanced deterministic sampling toward evenly spaced path-"
        "length percentiles within each requested difficulty bucket"
    )
