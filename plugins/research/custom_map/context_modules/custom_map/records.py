"""Read run-local memory records for custom_map without exposing raw logs.

The context module needs more than the rolling recent refs because map evidence
is cumulative. This reader prefers the append-only JSONL files under a run
directory and falls back to recent refs when a full run directory is unavailable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.memory.flight_memory import FlightMemory

from .schemas import ActionRecord, CustomMapRecords, FrameRecord, PosePoint


# These are the positive-Z-up metric frames currently emitted by production
# Runtime paths. Raw AirSim NED is intentionally excluded: STMR projection must
# only receive poses after the adapter has converted them to canonical axes.
SUPPORTED_STMR_POSE_FRAMES = {
    "canonical_unspecified",
    "unrealcv_world_m",
    "uav_flow_initial_local_canonical_m",
}


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

    parsed_poses = [item for item in poses if item is not None]
    _validate_pose_coordinate_frames(parsed_poses)
    return CustomMapRecords(
        run_dir=run_dir,
        actions=[item for item in actions if item is not None],
        poses=parsed_poses,
        frames=[item for item in frames if item is not None],
        poses_by_observation_id=_poses_by_observation_id(parsed_poses),
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
    position = pose.get("position") if isinstance(pose.get("position"), dict) else pose
    x = _float_or_none(position.get("x"))
    y = _float_or_none(position.get("y"))
    if x is None or y is None:
        return None
    yaw_deg = _pose_yaw_deg(pose)
    coordinate_frame = str(
        item.get("coordinate_frame") or "canonical_unspecified"
    ).strip()
    if coordinate_frame not in SUPPORTED_STMR_POSE_FRAMES:
        raise ValueError(
            "STMR requires a supported positive-Z-up metric pose frame; "
            f"received {coordinate_frame!r}"
        )
    return PosePoint(
        x=x,
        y=y,
        z=_float_or_none(position.get("z")),
        yaw_deg=yaw_deg,
        timestamp=str(item.get("timestamp")) if item.get("timestamp") is not None else None,
        source=str(item.get("source")) if item.get("source") is not None else None,
        action_id=str(item.get("action_id")) if item.get("action_id") is not None else None,
        observation_id=str(item.get("observation_id")) if item.get("observation_id") is not None else None,
        coordinate_frame=coordinate_frame,
    )


def _frame_record(item: dict[str, Any] | None, run_dir: Path | None) -> FrameRecord | None:
    if not isinstance(item, dict):
        return None
    metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
    # AirSim artifacts are recorded in metadata while older/custom-map records
    # use the top-level path. Keep one resolver so online STMR can consume both.
    raw_path = item.get("path") or metadata.get("artifact_path")
    path = _resolve_path(raw_path, run_dir) if isinstance(raw_path, str) and raw_path else None
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
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _pose_yaw_deg(pose: dict[str, Any]) -> float | None:
    """Read flat degree or nested degree/radian Euler representations."""

    direct = _float_or_none(pose.get("yaw_deg"))
    if direct is not None:
        return direct
    euler_deg = pose.get("euler_deg")
    if isinstance(euler_deg, dict):
        nested_deg = _float_or_none(euler_deg.get("yaw"))
        if nested_deg is not None:
            return nested_deg
    euler_rad = pose.get("euler_rad")
    if isinstance(euler_rad, dict):
        nested_rad = _float_or_none(euler_rad.get("yaw"))
        if nested_rad is not None:
            return math.degrees(nested_rad)
    return None


def _validate_pose_coordinate_frames(poses: list[PosePoint]) -> None:
    frames = {pose.coordinate_frame for pose in poses}
    if len(frames) > 1:
        raise ValueError(
            "STMR cannot combine mixed pose coordinate frames: "
            + ", ".join(sorted(frames))
        )


def _poses_by_observation_id(poses: list[PosePoint]) -> dict[str, PosePoint]:
    return {
        pose.observation_id: pose
        for pose in poses
        if pose.observation_id
    }
