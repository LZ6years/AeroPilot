"""Persist custom_map segmentation artifacts into run-local memory.

The writer appends a frame record for an existing observation and merges the
segmentation label map into memory/map/mask_label_map.json. It avoids invoking
the context module; rendering remains a separate step.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .closed_set_segmentation import ClosedSetSegmentationResult


def append_segmentation_result(
    run_dir: str | Path,
    result: ClosedSetSegmentationResult,
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    run_dir = Path(run_dir)
    if result.observation_id is None:
        raise ValueError("Segmentation result must include observation_id before writing memory")
    if not force and _has_segmentation_frame(run_dir, result.observation_id):
        return None
    merge_mask_label_map(run_dir, result.label_map)
    return _append_frame(run_dir, result)


def merge_mask_label_map(run_dir: str | Path, label_map: dict[str, dict[str, Any]]) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "memory" / "map" / "mask_label_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"colors": {}, "source": "custom_map_segmentation_sources"}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            data.update(loaded)
    colors = data.get("colors") if isinstance(data.get("colors"), dict) else {}
    colors.update({str(key): dict(value) for key, value in label_map.items()})
    data["colors"] = colors
    data["updated_by"] = "custom_map.closed_set_segmentation"
    data["updated_at"] = _timestamp()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _append_frame(run_dir: Path, result: ClosedSetSegmentationResult) -> dict[str, Any]:
    memory_dir = run_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / "frames.jsonl"
    line_number = len(path.read_text(encoding="utf-8").splitlines()) + 1 if path.exists() else 1
    timestamp = _timestamp()
    frame = {
        "id": f"segframe_{line_number:04d}",
        "timestamp": timestamp,
        "observation_id": result.observation_id,
        "modality": "segmentation",
        "path": str(result.mask_path),
        "metadata": {
            "path": str(result.mask_path),
            "mock": False,
            "source": result.stats.get("source", "segformer_ade20k"),
            "overlay_path": str(result.overlay_path),
            "label_map": result.label_map,
            "segmentation_stats": result.stats,
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(frame, ensure_ascii=False) + "\n")
    return {"id": frame["id"], "ref": f"memory/frames.jsonl#{line_number}", "timestamp": timestamp}


def _has_segmentation_frame(run_dir: Path, observation_id: str) -> bool:
    path = run_dir / "memory" / "frames.jsonl"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("observation_id") == observation_id and str(item.get("modality", "")).lower() == "segmentation":
            return True
    return False


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
