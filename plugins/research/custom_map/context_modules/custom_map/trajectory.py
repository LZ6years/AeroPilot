"""Compress raw action and pose history into map-ready trajectory evidence."""

from __future__ import annotations

import math
from typing import Any

from .schemas import ActionRecord, CustomMapRecords, PosePoint, TrajectorySummary


def build_trajectory(records: CustomMapRecords) -> TrajectorySummary:
    # Runtime observation poses form the camera/map timeline and exclude
    # preflight helpers such as connect/hover/set_pose. Older custom-map runs
    # recorded only action-result poses, so keep that representation as a
    # compatibility fallback when no observation-linked pose exists.
    observation_path = [
        pose for pose in records.poses if pose.observation_id is not None
    ]
    path = observation_path or list(records.poses)
    action_pose_by_id = {
        pose.action_id: pose
        for pose in records.poses
        if pose.action_id
    }
    failed_markers = [
        action_pose_by_id[action.id]
        for action in records.actions
        if action.id in action_pose_by_id and action.ok is False
    ]
    segments = _segments_from_actions(records.actions)
    summary = {
        "pose_count": len(path),
        "segment_count": len(segments),
        "failed_action_count": sum(1 for action in records.actions if action.ok is False),
        "approx_path_distance_m": round(_path_distance(path), 3),
        "latest_heading_deg": path[-1].yaw_deg if path and path[-1].yaw_deg is not None else None,
        "history_compressed": True,
    }
    return TrajectorySummary(path=path, failed_markers=failed_markers, segments=segments, summary=summary)


def _segments_from_actions(actions: list[ActionRecord]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for action in actions:
        kind = _action_kind(action)
        if not segments or not _can_merge(segments[-1], action, kind):
            segments.append(_new_segment(action, kind))
            continue
        segment = dict(segments[-1])
        segment["count"] = int(segment.get("count", 1)) + 1
        segment["ok"] = segment.get("ok") is not False and action.ok is not False
        segment["last_action_id"] = action.id
        if kind == "yaw":
            segment["yaw_sum_deg"] = round(float(segment.get("yaw_sum_deg", 0.0)) + _yaw_deg(action), 3)
        elif kind == "move":
            segment["duration_sum_sec"] = round(float(segment.get("duration_sum_sec", 0.0)) + _duration_sec(action), 3)
        segments[-1] = segment
    return segments


def _new_segment(action: ActionRecord, kind: str) -> dict[str, Any]:
    segment: dict[str, Any] = {
        "kind": kind,
        "tool": action.tool,
        "count": 1,
        "ok": action.ok,
        "first_action_id": action.id,
        "last_action_id": action.id,
    }
    if kind == "yaw":
        segment["yaw_sum_deg"] = round(_yaw_deg(action), 3)
    elif kind == "move":
        segment["duration_sum_sec"] = round(_duration_sec(action), 3)
        segment["body_velocity_signature"] = _move_signature(action)
    return segment


def _can_merge(segment: dict[str, Any], action: ActionRecord, kind: str) -> bool:
    if segment.get("kind") != kind or segment.get("tool") != action.tool:
        return False
    if segment.get("ok") is False or action.ok is False:
        return False
    if kind == "move":
        return segment.get("body_velocity_signature") == _move_signature(action)
    return kind in {"yaw", "hover", "other"}


def _action_kind(action: ActionRecord) -> str:
    if action.tool == "uav.move_body":
        return "move"
    if action.tool == "uav.turn_relative":
        return "yaw"
    if action.tool == "uav.hover":
        return "hover"
    return "other"


def _yaw_deg(action: ActionRecord) -> float:
    return _float_arg(action, "yaw_deg")


def _duration_sec(action: ActionRecord) -> float:
    return _float_arg(action, "duration")


def _float_arg(action: ActionRecord, key: str) -> float:
    try:
        return float(action.args.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _move_signature(action: ActionRecord) -> str:
    return "|".join(_component_sign(action.args.get(key)) for key in ("vx", "vy", "vz"))


def _component_sign(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if numeric > 0:
        return "+"
    if numeric < 0:
        return "-"
    return "0"


def _path_distance(path: list[PosePoint]) -> float:
    total = 0.0
    for left, right in zip(path, path[1:]):
        total += math.hypot(right.x - left.x, right.y - left.y)
    return total
