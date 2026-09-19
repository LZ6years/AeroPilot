"""TravelUAV official-data index regression tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from evaluation.benchmarks.travel_uav.dataset import (
    build_travel_uav_index,
    load_travel_uav_index,
    normalize_instruction,
)


FIXTURES = Path("tests/fixtures/travel_uav")


def test_build_index_deduplicates_frame_rows_and_resolves_spawn_area(tmp_path):
    dataset = tmp_path / "dataset"
    for map_name, sequence, distance in (
        ("MapSeen", "seq_easy", 30),
        ("MapSeen", "seq_hard", 300),
        ("Carla_Town03", "seq_unseen_map", 30),
        ("MapSeen", "seq_unseen_object", 30),
    ):
        episode = dataset / map_name / sequence
        episode.mkdir(parents=True)
        merged = json.loads((FIXTURES / "merged_data.json").read_text())
        merged["trajectory_raw_detailed"][-1]["position"][0] = distance
        (episode / "merged_data.json").write_text(json.dumps(merged), encoding="utf-8")
        shutil.copyfile(FIXTURES / "mark.json", episode / "mark.json")
    output = tmp_path / "travel_uav_index.json"
    result = build_travel_uav_index(
        dataset_root=dataset,
        seen_split_path=FIXTURES / "seen_valset.json",
        unseen_split_path=FIXTURES / "unseen_valset.json",
        map_spawn_area_path=FIXTURES / "map_spawnarea_info.json",
        output_path=output,
        dataset_version="fixture-v1",
    )

    assert len(result["episodes"]) == 4
    seen, provenance = load_travel_uav_index(output, split="test_seen")
    assert len(seen) == 2
    assert {item.difficulty for item in seen} == {"easy", "hard"}
    assert seen[0].instruction == "Find the red vehicle near the road."
    assert seen[0].target.asset_name == "SM_RedVehicle"
    assert seen[0].target.scale == (1.25, 1.25, 1.25)
    assert provenance.source_split_file == "seen_valset.json"


def test_instruction_normalization_only_removes_leading_image_placeholder():
    assert normalize_instruction(" <image>\nFly toward <image> sign ") == "Fly toward <image> sign"


def test_travel_uav_cli_scripts_can_run_directly_from_repo_root():
    for script in (
        "evaluation/benchmarks/travel_uav/scripts/build_index.py",
        "evaluation/benchmarks/travel_uav/scripts/run_eval.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
