"""Build and load the compact TravelUAV benchmark index.

Official split files contain one row per frame and reference episode-local
``merged_data.json`` files.  This module deduplicates those rows, resolves the
official target-placement rule, and writes only the fields needed by the
benchmark runner.  It never imports AirSim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from evaluation.benchmarks.travel_uav.cases import (
    TravelUavCase,
    TravelUavDatasetProvenance,
)


INDEX_SCHEMA_VERSION = 1
UNSEEN_MAP_NAMES = frozenset({"Carla_Town03", "ModularPark"})


def build_travel_uav_index(
    *,
    dataset_root: str | Path,
    seen_split_path: str | Path,
    unseen_split_path: str | Path,
    map_spawn_area_path: str | Path,
    output_path: str | Path,
    dataset_version: str = "official",
) -> dict[str, Any]:
    """Create a portable episode index from official TravelUAV files."""

    # Resolve once so a relative dataset root and an absolute discovered
    # episode path still share a stable base for portable source_json values.
    root = Path(dataset_root).resolve()
    seen_path = Path(seen_split_path)
    unseen_path = Path(unseen_split_path)
    spawn_path = Path(map_spawn_area_path)
    spawn_areas = _json_object(spawn_path)
    episodes: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    for source_name, split_path in (
        ("test_seen", seen_path),
        ("test_unseen", unseen_path),
    ):
        rows = _json_list(split_path)
        unique_json_paths = _deduplicate_json_paths(rows, split_path)
        sources[source_name] = {
            "file": split_path.name,
            "sha256": sha256_file(split_path),
            "frame_rows": len(rows),
            "unique_episodes": len(unique_json_paths),
        }
        for source_json in unique_json_paths:
            merged_path = _resolve_episode_file(root, source_json)
            map_name = merged_path.parent.parent.name
            sequence_id = merged_path.parent.name
            split = source_name
            if source_name == "test_unseen":
                split = (
                    "test_unseen_map"
                    if map_name in UNSEEN_MAP_NAMES
                    else "test_unseen_object"
                )
            episodes.append(
                _build_episode(
                    merged_path,
                    root=root,
                    map_name=map_name,
                    sequence_id=sequence_id,
                    split=split,
                    spawn_areas=spawn_areas,
                )
            )
    episode_ids = [item["episode_id"] for item in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("TravelUAV compact index produced duplicate episode_id")
    episodes.sort(key=lambda item: (item["split"], item["map_name"], item["episode_id"]))
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "dataset_version": str(dataset_version),
        "sources": sources,
        "spawn_area_file": {
            "file": spawn_path.name,
            "sha256": sha256_file(spawn_path),
        },
        "episodes": episodes,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_travel_uav_index(
    index_path: str | Path,
    *,
    split: str,
) -> tuple[list[TravelUavCase], TravelUavDatasetProvenance]:
    """Load all cases for one logical split and verify index structure."""

    path = Path(index_path)
    payload = _json_object(path)
    if int(payload.get("schema_version", -1)) != INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported TravelUAV index schema_version: {payload.get('schema_version')}"
        )
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("TravelUAV index requires an episodes list")
    all_cases = [TravelUavCase.from_dict(_require_object(item, "episode")) for item in rows]
    cases = [case for case in all_cases if case.split == split]
    if not cases:
        raise ValueError(f"TravelUAV index contains no cases for split {split!r}")
    source_key = "test_seen" if split == "test_seen" else "test_unseen"
    sources = _require_object(payload.get("sources"), "sources")
    source = _require_object(sources.get(source_key), f"sources.{source_key}")
    return cases, TravelUavDatasetProvenance(
        dataset_version=str(payload.get("dataset_version", "unspecified")),
        index_file=path.name,
        index_sha256=sha256_file(path),
        source_split_file=str(source.get("file", "")),
        source_split_sha256=str(source.get("sha256", "")),
        split=split,
        episode_count=len(cases),
        map_names=tuple(sorted({case.map_name for case in cases})),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_episode(
    merged_path: Path,
    *,
    root: Path,
    map_name: str,
    sequence_id: str,
    split: str,
    spawn_areas: dict[str, Any],
) -> dict[str, Any]:
    merged = _json_object(merged_path)
    trajectory = merged.get("trajectory_raw_detailed")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"TravelUAV episode {merged_path} has no trajectory_raw_detailed")
    path = [
        list(_finite_tuple(_require_object(row, "trajectory row").get("position"), 3, "position"))
        for row in trajectory
    ]
    start = _require_object(trajectory[0], "first trajectory row")
    orientation = _finite_tuple(start.get("orientation"), 4, "orientation")
    conversations = merged.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"TravelUAV episode {merged_path} has no conversations")
    raw_instruction = str(_require_object(conversations[0], "conversation").get("value", ""))
    instruction = normalize_instruction(raw_instruction)
    mark_path = merged_path.with_name("mark.json")
    mark = _json_object(mark_path)
    target_mark = _require_object(mark.get("target"), "mark.target")
    marked_position = _finite_tuple(target_mark.get("position"), 3, "mark.target.position")
    area_rows = spawn_areas.get(map_name)
    if not isinstance(area_rows, list):
        raise ValueError(f"TravelUAV spawn-area metadata missing map {map_name!r}")
    area = _closest_spawn_area(marked_position, area_rows)
    episode_id = _safe_episode_id(map_name, sequence_id)
    return {
        "episode_id": episode_id,
        "sequence_id": sequence_id,
        "map_name": map_name,
        "split": split,
        "instruction": instruction,
        "raw_instruction": raw_instruction,
        "start_position_ned": path[0],
        "start_rotation_xyzw": list(orientation),
        "reference_path_ned": path,
        "reference_path_length_m": _path_length(path),
        "target": {
            "asset_name": str(area[16]),
            "object_name": f"flyagent_target_{episode_id}",
            "position_ned": [float(area[9]), float(area[10]), float(area[11])],
            "rotation_xyzw": [
                float(area[13]),
                float(area[14]),
                float(area[15]),
                float(area[12]),
            ],
            "scale": [float(area[17])] * 3,
            "physics_enabled": False,
            "is_blueprint": False,
        },
        "source_json": merged_path.relative_to(root).as_posix(),
    }


def normalize_instruction(value: str) -> str:
    """Remove only the dataset's leading multimodal placeholder."""

    text = str(value).lstrip("\ufeff")
    text = re.sub(r"^\s*<image>\s*", "", text, count=1)
    text = text.strip()
    if not text:
        raise ValueError("TravelUAV instruction is empty after removing <image>")
    return text


def _deduplicate_json_paths(rows: list[Any], source: Path) -> list[str]:
    values: set[str] = set()
    for index, row in enumerate(rows):
        item = _require_object(row, f"{source.name} row {index}")
        value = str(item.get("json", "")).strip().replace("\\", "/")
        if not value:
            raise ValueError(f"TravelUAV split {source} row {index} missing json")
        values.add(value)
    return sorted(values)


def _resolve_episode_file(root: Path, source_json: str) -> Path:
    raw = Path(source_json)
    candidates = [root / raw]
    parts = raw.as_posix().strip("/").split("/")
    if len(parts) >= 3:
        candidates.append(root.joinpath(*parts[-3:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"TravelUAV episode file {source_json!r} not found under {root}"
    )


def _closest_spawn_area(position: tuple[float, ...], rows: list[Any]) -> list[Any]:
    candidates = [row for row in rows if isinstance(row, list) and len(row) >= 18]
    if not candidates:
        raise ValueError("TravelUAV spawn-area map has no usable 18-field row")

    def distance(row: list[Any]) -> float:
        anchor = (float(row[0]) + 1.0, float(row[1]) + 1.0, float(row[2]) + 0.5)
        return math.sqrt(sum((position[i] - anchor[i]) ** 2 for i in range(3)))

    selected = min(candidates, key=distance)
    _finite_tuple([selected[i] for i in (9, 10, 11, 13, 14, 15, 12, 17)], 8, "spawn area")
    if not str(selected[16]).strip() or float(selected[17]) <= 0.0:
        raise ValueError("TravelUAV selected spawn area has invalid asset or scale")
    return selected


def _path_length(points: list[list[float]]) -> float:
    return sum(
        math.sqrt(sum((points[index][axis] - points[index - 1][axis]) ** 2 for axis in range(3)))
        for index in range(1, len(points))
    )


def _safe_episode_id(map_name: str, sequence_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{map_name}__{sequence_id}").strip("_")


def _finite_tuple(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"TravelUAV {label} must contain {length} values")
    parsed = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"TravelUAV {label} must contain finite values")
    return parsed


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return _require_object(value, str(path))


def _json_list(path: Path) -> list[Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ValueError(f"TravelUAV {path} must contain a JSON list")
    return value


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"TravelUAV {label} must be an object")
    return dict(value)
