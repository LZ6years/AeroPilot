"""Read run-local memory records for custom_map without exposing raw logs.

The context module needs more than the rolling recent refs because map evidence
is cumulative. This reader prefers the append-only JSONL files under a run
directory and falls back to recent refs when a full run directory is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.memory.flight_memory import FlightMemory

from .schemas import ActionRecord, CustomMapRecords, FrameRecord, PosePoint


def read_custom_map_records(memory: FlightMemory) -> CustomMapRecords:
    reader = memory.memory_reader
    run_dir = Path(reader.run_dir) if reader is not None and getattr(reader, "run_dir", None) else None
    warnings: list[str] = []

    if run_dir is not None and (run_dir / "memory").exists():
        actions = [_action_record(item) for item in _read_jsonl(run_dir / "memory" / "actions.jsonl")]
        poses = [_pose_point(item) for item in _read_jsonl(run_dir / "memory" / "poses.jsonl")]
        frames = [_frame_record(item, run_dir) for item in _read_jsonl(run_dir / "memory" / "frames.jsonl")]
    elif reader is not None:
        warnings.append("custom_map_full_run_memory_unavailable:using_recent_refs")
        actions = [_action_record(item) for item in reader.read_refs(memory.recent_action_refs)]
        poses = [_pose_point(item) for item in reader.read_refs(memory.recent_pose_refs)]
        frames = [_frame_record(item, run_dir) for item in reader.read_refs(memory.recent_frame_refs)]
    else:
        warnings.append("custom_map_memory_reader_missing")
        actions = []
        poses = []
        frames = []

    return CustomMapRecords(
        run_dir=run_dir,
        actions=[item for item in actions if item is not None],
        poses=[item for item in poses if item is not None],
        frames=[item for item in frames if item is not None],
        poses_by_observation_id=_poses_by_observation_id([item for item in poses if item is not None]),
        warnings=warnings,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _action_record(item: dict[str, Any] | None) -> ActionRecord | None:
    if not isinstance(item, dict):
        return None
    result = item.get("result", {})
    ok = result.get("ok") if isinstance(result, dict) else None
    return ActionRecord(
        id=str(item.get("id")) if item.get("id") is not None else None,
        tool=str(item.get("tool") or ""),
        args=dict(item.get("args", {})) if isinstance(item.get("args"), dict) else {},
        ok=bool(ok) if ok is not None else None,
        timestamp=str(item.get("timestamp")) if item.get("timestamp") is not None else None,
    )


def _pose_point(item: dict[str, Any] | None) -> PosePoint | None:
    if not isinstance(item, dict) or not isinstance(item.get("pose"), dict):
        return None
    pose = item["pose"]
    x = _float_or_none(pose.get("x"))
    y = _float_or_none(pose.get("y"))
    if x is None or y is None:
        return None
    return PosePoint(
        x=x,
        y=y,
        z=_float_or_none(pose.get("z")),
        yaw_deg=_float_or_none(pose.get("yaw_deg")),
        timestamp=str(item.get("timestamp")) if item.get("timestamp") is not None else None,
        source=str(item.get("source")) if item.get("source") is not None else None,
        action_id=str(item.get("action_id")) if item.get("action_id") is not None else None,
        observation_id=str(item.get("observation_id")) if item.get("observation_id") is not None else None,
    )


def _frame_record(item: dict[str, Any] | None, run_dir: Path | None) -> FrameRecord | None:
    if not isinstance(item, dict):
        return None
    raw_path = item.get("path")
    path = _resolve_path(raw_path, run_dir) if isinstance(raw_path, str) and raw_path else None
    metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
    return FrameRecord(
        id=str(item.get("id")) if item.get("id") is not None else None,
        observation_id=str(item.get("observation_id")) if item.get("observation_id") is not None else None,
        modality=str(item.get("modality") or ""),
        path=path,
        metadata=metadata,
        timestamp=str(item.get("timestamp")) if item.get("timestamp") is not None else None,
    )


def _resolve_path(raw_path: str, run_dir: Path | None) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if run_dir is not None:
        run_path = run_dir / path
        if run_path.exists():
            return run_path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else path


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _poses_by_observation_id(poses: list[PosePoint]) -> dict[str, PosePoint]:
    return {
        pose.observation_id: pose
        for pose in poses
        if pose.observation_id
    }
