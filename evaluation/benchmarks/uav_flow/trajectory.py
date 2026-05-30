"""Trajectory conversion helpers for UAV-Flow benchmark metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase

M_TO_CM = 100.0


def world_to_local(target: list[float], init_pos: list[float]) -> list[float]:
    """Convert UAV-Flow world coordinates into the first-frame local frame."""

    x0, y0, z0 = init_pos[0:3]
    yaw0 = init_pos[4]
    dx = target[0] - x0
    dy = target[1] - y0
    dz = target[2] - z0
    theta = -math.radians(yaw0)
    x_rel = dx * math.cos(theta) - dy * math.sin(theta)
    y_rel = dx * math.sin(theta) + dy * math.cos(theta)
    return [x_rel, y_rel, dz]


def world_pose_to_model_pose(world_pose_cm: list[float], initial_pos_cm: list[float]) -> dict[str, float]:
    """Create model-facing local pose in meters without exposing target truth."""

    local_cm = world_to_local(world_pose_cm[:3], initial_pos_cm)
    yaw_delta = _normalize_yaw(world_pose_cm[4] - initial_pos_cm[4])
    return {
        "x": round(local_cm[0] / M_TO_CM, 6),
        "y": round(local_cm[1] / M_TO_CM, 6),
        "z": round(local_cm[2] / M_TO_CM, 6),
        "pitch_deg": float(world_pose_cm[3]),
        "yaw_deg": round(yaw_delta, 6),
        "roll_deg": float(world_pose_cm[5]),
        "coordinate_frame": "uav_flow_initial_local_m",
    }


def model_pose_to_uav_flow_state(pose: dict[str, Any]) -> dict[str, list[list[float]]]:
    """Convert a UAV-Agent local meter pose into UAV-Flow trajectory state shape."""

    return {
        "state": [
            [
                round(float(pose.get("x", 0.0)) * M_TO_CM, 6),
                round(float(pose.get("y", 0.0)) * M_TO_CM, 6),
                round(float(pose.get("z", 0.0)) * M_TO_CM, 6),
            ],
            [
                float(pose.get("pitch_deg", 0.0)),
                float(pose.get("yaw_deg", 0.0)),
                float(pose.get("roll_deg", 0.0)),
            ],
        ]
    }


def export_trajectory(
    case: UavFlowCase,
    pose_history: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write UAV-Flow-compatible trajectory JSON and metadata."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    states = [model_pose_to_uav_flow_state(pose) for pose in pose_history]
    path.write_text(json.dumps(states, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "path": str(path),
        "point_count": len(states),
        "unit": "cm",
        "coordinate_frame": "uav_flow_initial_local",
        "source_pose_frame": "uav_agent_local_m",
        "initial_pos": list(case.initial_pos),
    }


def target_local_from_case(case: UavFlowCase) -> list[float] | None:
    if not case.target_pos:
        return None
    return world_to_local(case.target_pos, case.initial_pos)


def _normalize_yaw(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized

