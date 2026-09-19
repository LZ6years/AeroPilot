"""Deterministic TravelUAV split and difficulty selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from evaluation.benchmarks.travel_uav.cases import TravelUavCase


TRAVEL_UAV_SPLITS = (
    "test_seen",
    "test_unseen_map",
    "test_unseen_object",
)
TRAVEL_UAV_DIFFICULTIES = ("easy", "hard")


@dataclass(frozen=True)
class TravelUavCaseSelection:
    """Durable record of filtering and deterministic sampling."""

    split: str
    difficulties: tuple[str, ...]
    limit_per_difficulty: int | None
    requested_case_ids: tuple[str, ...]
    population_counts: dict[str, int]
    selected_counts: dict[str, int]
    selected_case_ids: tuple[str, ...]
    selected_map_names: tuple[str, ...]
    strategy: str

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "difficulties",
            "requested_case_ids",
            "selected_case_ids",
            "selected_map_names",
        ):
            data[key] = list(data[key])
        return data


def select_travel_uav_cases(
    cases: Iterable[TravelUavCase],
    *,
    split: str,
    difficulties: tuple[str, ...] = TRAVEL_UAV_DIFFICULTIES,
    limit_per_difficulty: int | None = None,
    case_ids: tuple[str, ...] | None = None,
) -> tuple[list[TravelUavCase], TravelUavCaseSelection]:
    """Select exact IDs or length-spread cases within each difficulty."""

    if split not in TRAVEL_UAV_SPLITS:
        raise ValueError(f"Unsupported TravelUAV split: {split}")
    unknown = sorted(set(difficulties).difference(TRAVEL_UAV_DIFFICULTIES))
    if unknown or not difficulties or len(set(difficulties)) != len(difficulties):
        raise ValueError("TravelUAV difficulties must be unique easy/hard values")
    if limit_per_difficulty is not None and limit_per_difficulty <= 0:
        raise ValueError("TravelUAV limit_per_difficulty must be positive or null")
    population = sorted(
        (case for case in cases if case.split == split),
        key=lambda item: (item.map_name, item.episode_id),
    )
    by_id = {case.episode_id: case for case in population}
    if len(by_id) != len(population):
        raise ValueError(f"TravelUAV split {split} contains duplicate episode_id")
    population_counts = {
        label: sum(case.difficulty == label for case in population)
        for label in TRAVEL_UAV_DIFFICULTIES
    }
    requested = tuple(case_ids or ())
    if len(set(requested)) != len(requested):
        raise ValueError("TravelUAV case_ids must be unique")
    if requested:
        missing = [case_id for case_id in requested if case_id not in by_id]
        if missing:
            raise ValueError("Unknown TravelUAV case_ids: " + ", ".join(missing))
        selected = [by_id[case_id] for case_id in requested]
        strategy = "exact_case_ids"
    else:
        selected = []
        for difficulty in difficulties:
            bucket = [case for case in population if case.difficulty == difficulty]
            selected.extend(_length_spread(bucket, limit_per_difficulty))
        strategy = (
            "all_per_difficulty"
            if limit_per_difficulty is None
            else "deterministic_length_spread"
        )
    selection = TravelUavCaseSelection(
        split=split,
        difficulties=tuple(difficulties),
        limit_per_difficulty=limit_per_difficulty,
        requested_case_ids=requested,
        population_counts=population_counts,
        selected_counts={
            label: sum(case.difficulty == label for case in selected)
            for label in TRAVEL_UAV_DIFFICULTIES
        },
        selected_case_ids=tuple(case.episode_id for case in selected),
        selected_map_names=tuple(sorted({case.map_name for case in selected})),
        strategy=strategy,
    )
    return selected, selection


def _length_spread(
    cases: list[TravelUavCase], limit: int | None
) -> list[TravelUavCase]:
    ordered = sorted(
        cases,
        key=lambda item: (
            item.reference_path_length_m,
            item.map_name,
            item.episode_id,
        ),
    )
    if limit is None or limit >= len(ordered):
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(index * (len(ordered) - 1) / (limit - 1))
        for index in range(limit)
    }
    # Rounding is monotonic but retain a defensive fill for tiny buckets.
    chosen = [ordered[index] for index in sorted(indices)]
    if len(chosen) < limit:
        chosen_ids = {case.episode_id for case in chosen}
        chosen.extend(
            case for case in ordered if case.episode_id not in chosen_ids
        )
    return chosen[:limit]
