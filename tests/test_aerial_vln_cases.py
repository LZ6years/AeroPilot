"""AerialVLN loader and configuration tests."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.benchmarks.aerial_vln.cases import (
    AerialVlnCase,
    load_aerial_vln_dataset,
    load_aerial_vln_cases,
)
from evaluation.benchmarks.aerial_vln.config import (
    AERIAL_VLN_S_SPLITS,
    AerialVlnEvalConfig,
    aerial_vln_split_label,
)
from evaluation.benchmarks.aerial_vln.selection import (
    classify_and_select_aerial_vln_cases,
    reference_path_length_m,
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


def test_dataset_provenance_records_family_hash_population_and_scenes(
    tmp_path,
):
    path = tmp_path / "val_seen.json"
    path.write_text(
        json.dumps({"episodes": [episode("10"), episode("2")]}),
        encoding="utf-8",
    )

    cases, provenance = load_aerial_vln_dataset(
        tmp_path,
        "val_seen",
        dataset_family="aerialvln-s",
        dataset_version="official",
    )

    assert len(cases) == 2
    assert provenance.dataset_family == "aerialvln-s"
    assert provenance.split == "val_seen"
    assert provenance.scene_ids == ("2", "10")
    assert provenance.split_sha256 == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_dataset_provenance_is_bound_to_each_selected_split(tmp_path):
    hashes = {}
    for split, scene_id in (("val_seen", "1"), ("val_unseen", "2")):
        path = tmp_path / f"{split}.json"
        path.write_text(
            json.dumps({"episodes": [episode(scene_id)]}),
            encoding="utf-8",
        )
        _, provenance = load_aerial_vln_dataset(
            tmp_path,
            split,
            dataset_family="aerialvln-s",
        )
        hashes[split] = provenance.split_sha256
        assert provenance.split == split
        assert provenance.split_file == f"{split}.json"
        assert provenance.scene_ids == (scene_id,)

    assert hashes["val_seen"] != hashes["val_unseen"]


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
    with pytest.raises(ValueError, match="64 hex"):
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": "data",
                "expected_split_sha256": "not-a-hash",
            }
        )


@pytest.mark.parametrize(
    "override, message",
    [
        ({"max_cases": 1}, "max_cases"),
        ({"scene_filter": ["1"]}, "scene_filter"),
    ],
)
def test_aerial_vln_s_rejects_preclassification_filters(
    override, message
):
    with pytest.raises(ValueError, match=message):
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": "data",
                "dataset_family": "aerialvln-s",
                **override,
            }
        )


@pytest.mark.parametrize(
    "split, label",
    [
        ("val_seen", "Validation Seen"),
        ("val_unseen", "Validation Unseen"),
    ],
)
def test_aerial_vln_s_accepts_official_validation_splits(split, label):
    config = AerialVlnEvalConfig.from_dict(
        {
            "dataset_root": "data",
            "dataset_family": "aerialvln-s",
            "split": split,
        }
    )

    assert config.split == split
    assert split in AERIAL_VLN_S_SPLITS
    assert aerial_vln_split_label(split) == label


@pytest.mark.parametrize("split", ["train", "test", "val_unseen_half"])
def test_aerial_vln_s_rejects_non_validation_splits(split):
    with pytest.raises(ValueError, match="val_seen or val_unseen"):
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": "data",
                "dataset_family": "aerialvln-s",
                "split": split,
            }
        )


def test_aerial_vln_s_default_split_remains_val_seen():
    config = AerialVlnEvalConfig.from_dict(
        {
            "dataset_root": "data",
            "dataset_family": "aerialvln-s",
        }
    )

    assert config.split == "val_seen"


@pytest.mark.parametrize(
    "filename, split, limit",
    [
        ("aerial_vln_eval.example.json", "val_seen", 3),
        (
            "aerial_vln_s_val_seen_full.example.json",
            "val_seen",
            None,
        ),
        (
            "aerial_vln_s_val_unseen.example.json",
            "val_unseen",
            3,
        ),
        (
            "aerial_vln_s_val_unseen_full.example.json",
            "val_unseen",
            None,
        ),
    ],
)
def test_official_split_example_configs(filename, split, limit):
    config_path = (
        Path("evaluation/benchmarks/aerial_vln/configs") / filename
    )

    config = AerialVlnEvalConfig.from_file(config_path)

    assert config.dataset_family == "aerialvln-s"
    assert config.split == split
    assert config.limit_per_difficulty == limit


def test_config_validates_difficulties_and_per_bucket_limit():
    config = AerialVlnEvalConfig.from_dict(
        {
            "dataset_root": "data",
            "dataset_family": "aerialvln-s",
            "difficulties": ["medium", "high"],
            "limit_per_difficulty": 3,
        }
    )

    assert config.difficulties == ("medium", "high")
    assert config.limit_per_difficulty == 3
    with pytest.raises(ValueError, match="Unknown"):
        AerialVlnEvalConfig.from_dict(
            {"dataset_root": "data", "difficulties": ["extreme"]}
        )
    with pytest.raises(ValueError, match="positive or null"):
        AerialVlnEvalConfig.from_dict(
            {"dataset_root": "data", "limit_per_difficulty": 0}
        )


def test_difficulty_sampling_uses_global_deduplicated_path_tertiles():
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

    selected, selection = classify_and_select_aerial_vln_cases(
        cases, limit_per_difficulty=1
    )

    assert [case.episode_id for case in selected] == [
        "ep_02",
        "ep_05",
        "ep_08",
    ]
    assert selection.bucket_counts == {
        "low": 3,
        "medium": 3,
        "high": 3,
    }
    assert selection.q33_m == 11.0 / 3.0
    assert selection.q67_m == 19.0 / 3.0


def test_limited_sampling_covers_scenes_before_reusing_them():
    cases = []
    for index in range(1, 13):
        scene_id = str(((index - 1) % 3) + 1)
        cases.append(
            AerialVlnCase.from_dict(
                {
                    **episode(scene_id),
                    "episode_id": f"ep_{index:02d}",
                    "trajectory_id": f"traj_{index:02d}",
                    "reference_path": [[0, 0, 0], [index, 0, 0]],
                }
            )
        )

    selected, selection = classify_and_select_aerial_vln_cases(
        cases, limit_per_difficulty=3
    )

    assert len(selected) == 9
    for label in ("low", "medium", "high"):
        rows = [
            item
            for item in selection.selected_assignments
            if item.difficulty == label
        ]
        assert {item.scene_id for item in rows} == {"1", "2", "3"}


def test_limited_sampling_is_deterministic_when_limit_is_below_scene_count():
    cases = [
        AerialVlnCase.from_dict(
            {
                **episode(str(index)),
                "episode_id": f"ep_{index}",
                "trajectory_id": f"traj_{index}",
                "reference_path": [[0, 0, 0], [index, 0, 0]],
            }
        )
        for index in range(1, 10)
    ]

    first, first_selection = classify_and_select_aerial_vln_cases(
        cases,
        difficulties=("low",),
        limit_per_difficulty=2,
    )
    second, second_selection = classify_and_select_aerial_vln_cases(
        list(reversed(cases)),
        difficulties=("low",),
        limit_per_difficulty=2,
    )

    assert [case.episode_id for case in first] == [
        case.episode_id for case in second
    ]
    assert len({case.scene_id for case in first}) == 2
    assert first_selection.to_dict() == second_selection.to_dict()


def test_limited_sampling_targets_path_length_quantiles_not_row_ranks():
    lengths = (
        1.0,
        2.0,
        100.0,
        1000.0,
        1001.0,
        1002.0,
        2000.0,
        2001.0,
        2002.0,
    )
    cases = [
        AerialVlnCase.from_dict(
            {
                **episode("1"),
                "episode_id": f"ep_{index}",
                "trajectory_id": f"traj_{index}",
                "reference_path": [[0, 0, 0], [length, 0, 0]],
            }
        )
        for index, length in enumerate(lengths)
    ]

    selected, _ = classify_and_select_aerial_vln_cases(
        cases,
        difficulties=("low",),
        limit_per_difficulty=1,
    )

    # The low bucket is [1, 2, 100]. Its median length is 2, even though
    # distances are highly skewed.
    assert [case.episode_id for case in selected] == ["ep_1"]


def test_selection_supports_one_bucket_all_and_limit_above_population():
    cases = [
        AerialVlnCase.from_dict(
            {
                **episode(str(index)),
                "episode_id": f"ep_{index}",
                "trajectory_id": f"traj_{index}",
                "reference_path": [[0, 0, 0], [index, 0, 0]],
            }
        )
        for index in range(1, 10)
    ]

    selected_high, selection_high = (
        classify_and_select_aerial_vln_cases(
            cases,
            difficulties=("high",),
            limit_per_difficulty=None,
        )
    )
    selected_all, _ = classify_and_select_aerial_vln_cases(
        cases, limit_per_difficulty=20
    )

    assert len(selected_high) == selection_high.bucket_counts["high"]
    assert all(
        item.difficulty == "high"
        for item in selection_high.selected_assignments
    )
    assert len(selected_all) == len(cases)


def test_equal_length_selection_uses_stable_scene_and_episode_ties():
    cases = [
        AerialVlnCase.from_dict(
            {
                **episode(scene_id),
                "episode_id": episode_id,
                "trajectory_id": f"traj_{episode_id}",
                "reference_path": [[0, 0, 0], [5, 0, 0]],
            }
        )
        for scene_id, episode_id in (
            ("2", "ep_b"),
            ("1", "ep_c"),
            ("1", "ep_a"),
        )
    ]

    selected, selection = classify_and_select_aerial_vln_cases(
        cases,
        difficulties=("low",),
        limit_per_difficulty=2,
    )

    assert [case.episode_id for case in selected] == ["ep_a", "ep_b"]
    assert selection.bucket_counts == {
        "low": 3,
        "medium": 0,
        "high": 0,
    }


def test_case_rejects_non_finite_dataset_coordinates():
    payload = episode("1")
    payload["reference_path"] = [[0, 0, 0], [float("nan"), 0, 0]]

    with pytest.raises(ValueError, match="finite"):
        AerialVlnCase.from_dict(payload)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda payload: payload.pop("trajectory_id"), "trajectory_id"),
        (
            lambda payload: payload.update(reference_path=[]),
            "non-empty",
        ),
        (
            lambda payload: payload.update(reference_path=[[0, 0]]),
            "at least XYZ",
        ),
    ],
)
def test_case_rejects_missing_or_invalid_required_schema(
    mutation, message
):
    payload = episode("1")
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        AerialVlnCase.from_dict(payload)


def test_dataset_loader_rejects_duplicate_episode_ids(tmp_path):
    path = tmp_path / "val_seen.json"
    path.write_text(
        json.dumps({"episodes": [episode("1"), episode("1")]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate episode_id"):
        load_aerial_vln_dataset(tmp_path, "val_seen")


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
