"""TravelUAV split, difficulty, and stable selection tests."""

import pytest

from evaluation.benchmarks.travel_uav.selection import select_travel_uav_cases
from tests.support.travel_uav import case


def test_selection_keeps_split_and_easy_hard_boundary():
    cases = [
        case("a", length=250),
        case("b", length=251),
        case("c", split="test_unseen_map", map_name="ModularPark", length=10),
    ]
    selected, summary = select_travel_uav_cases(
        cases, split="test_seen", difficulties=("easy", "hard")
    )

    assert [item.episode_id for item in selected] == ["a", "b"]
    assert summary.population_counts == {"easy": 1, "hard": 1}


def test_exact_case_ids_preserve_order_and_fail_before_execution():
    cases = [case("a"), case("b", length=300)]
    selected, _ = select_travel_uav_cases(
        cases,
        split="test_seen",
        case_ids=("b", "a"),
    )
    assert [item.episode_id for item in selected] == ["b", "a"]
    with pytest.raises(ValueError, match="Unknown TravelUAV case_ids"):
        select_travel_uav_cases(cases, split="test_seen", case_ids=("missing",))


def test_limited_selection_spreads_path_lengths_deterministically():
    cases = [case(str(index), length=float(index * 10)) for index in range(1, 7)]
    selected, _ = select_travel_uav_cases(
        cases,
        split="test_seen",
        difficulties=("easy",),
        limit_per_difficulty=3,
    )
    assert [item.reference_path_length_m for item in selected] == [10, 30, 60]
