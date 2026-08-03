"""AerialVLN loader and configuration tests."""

from __future__ import annotations

import gzip
import json

import pytest

from evaluation.benchmarks.aerial_vln.cases import (
    AerialVlnCase,
    load_aerial_vln_cases,
)
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from tests.manual.aerial_vln_remote_difficulty_batch import (
    reference_path_length_m,
    select_difficulty_cases,
)


def test_case_loader_reads_nested_instruction_and_official_truth():
    case = AerialVlnCase.from_dict(episode("1"))

    assert case.episode_id == "ep_1"
    assert case.instruction == "Fly to the plaza"
    assert case.start_rotation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert case.goal_position == (10.0, 20.0, -5.0)
    assert case.official_actions == (1, 2, 0)
    assert "goal_position" not in case.planner_safe_metadata()


def test_split_loader_supports_gzip_filters_and_limits(tmp_path):
    path = tmp_path / "val_seen.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"episodes": [episode("1"), episode("2")]}, handle)

    cases = load_aerial_vln_cases(
        tmp_path,
        "val_seen",
        scene_filter=["2"],
        max_cases=1,
    )

    assert [case.scene_id for case in cases] == ["2"]


def test_config_requires_explicit_dataset_root_and_valid_scene_map():
    with pytest.raises(ValueError, match="dataset_root"):
        AerialVlnEvalConfig.from_dict({})
    with pytest.raises(ValueError, match="scene_map"):
        AerialVlnEvalConfig.from_dict(
            {"dataset_root": "data", "scene_map": {}}
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": "data",
                "initialization_min_settle_sec": -1,
            }
        )


def test_difficulty_sampling_uses_deduplicated_path_length_tertiles():
    assert reference_path_length_m(
        [(0, 0, 0), (0, 0, 0), (3, 4, 0)]
    ) == 5.0
    cases = [
        AerialVlnCase.from_dict(
            {
                **episode("1"),
                "episode_id": f"ep_{length:02d}",
                "trajectory_id": f"traj_{length:02d}",
                "reference_path": [[0, 0, 0], [length, 0, 0]],
            }
        )
        for length in range(1, 10)
    ]

    selected, manifest = select_difficulty_cases(
        cases, cases_per_difficulty=1
    )

    assert [case.episode_id for case in selected] == [
        "ep_02",
        "ep_05",
        "ep_08",
    ]
    assert manifest["bucket_counts"] == {
        "low": 3,
        "medium": 3,
        "high": 3,
    }


def episode(scene_id: str) -> dict:
    return {
        "episode_id": f"ep_{scene_id}",
        "trajectory_id": f"traj_{scene_id}",
        "scene_id": scene_id,
        "instruction": {"instruction_text": "Fly to the plaza"},
        "start_position": [1.0, 2.0, -3.0],
        "start_rotation": [1.0, 0.0, 0.0, 0.0],
        "goals": [{"position": [10.0, 20.0, -5.0]}],
        "reference_path": [
            [1.0, 2.0, -3.0],
            [10.0, 20.0, -5.0],
        ],
        "actions": [1, 2, 0],
    }
