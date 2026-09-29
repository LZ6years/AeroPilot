"""Small builders shared by TravelUAV unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmarks.travel_uav.cases import TravelUavCase


def case(
    episode_id: str = "MapSeen__seq",
    *,
    split: str = "test_seen",
    map_name: str = "MapSeen",
    length: float = 100.0,
) -> TravelUavCase:
    return TravelUavCase.from_dict(
        {
            "episode_id": episode_id,
            "sequence_id": episode_id.split("__")[-1],
            "map_name": map_name,
            "split": split,
            "instruction": "Find the red vehicle.",
            "raw_instruction": "<image>\nFind the red vehicle.",
            "start_position_ned": [1, 2, -3],
            "start_rotation_xyzw": [0, 0, 0, 1],
            "reference_path_ned": [[1, 2, -3], [length + 1, 2, -3]],
            "reference_path_length_m": length,
            "target": {
                "asset_name": "SM_Target",
                "object_name": f"flyagent_target_{episode_id}",
                "position_ned": [length + 1, 2, -3],
                "rotation_xyzw": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            },
            "source_json": f"{map_name}/{episode_id}/merged_data.json",
        }
    )


def write_index(path: Path, cases: list[TravelUavCase]) -> Path:
    payload = {
        "schema_version": 1,
        "dataset_version": "fixture",
        "sources": {
            "test_seen": {"file": "seen_valset.json", "sha256": "a" * 64},
            "test_unseen": {"file": "unseen_valset.json", "sha256": "b" * 64},
        },
        "spawn_area_file": {"file": "map_spawnarea_info.json", "sha256": "c" * 64},
        "episodes": [item.to_dict() for item in cases],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_map_config(path: Path, map_names: tuple[str, ...] = ("MapSeen",)) -> Path:
    maps = []
    for map_name in map_names:
        maps.append(
            {
                "map_id": f"travel_{map_name}",
                "provider": "fixture",
                "scene_id": map_name,
                "root": None,
                "connection": {"mode": "local"},
                "launch": {"manual": True, "existing_process_policy": "attach"},
                "control": {"protocol": "airsim"},
                "airsim": {
                    "mode": "Multirotor",
                    "vehicle_name": "Drone_1",
                    "camera_name": "FrontCamera",
                },
            }
        )
    path.write_text(json.dumps({"maps": maps}), encoding="utf-8")
    return path
