"""UAV body-frame motion and phase-local measurement helpers.

The current projection assumes yaw=0 means world +x is UAV forward and yaw=90
means world +y is UAV forward. The real-engine calibration demo verifies this
assumption before tightening completion tolerances.
"""

from __future__ import annotations

import math
from typing import Any

from src.uav.frames import normalize_yaw_delta_deg, pose_value, yaw_deg


def body_velocity_to_world_delta(
    vx: float,
    vy: float,
    vz: float,
    duration: float,
    origin_yaw_deg: float,
) -> dict[str, float]:
    """Integrate body-frame velocity into a world-frame delta."""

    yaw_rad = math.radians(float(origin_yaw_deg))
    dx = (float(vx) * math.cos(yaw_rad) - float(vy) * math.sin(yaw_rad)) * float(duration)
    dy = (float(vx) * math.sin(yaw_rad) + float(vy) * math.cos(yaw_rad)) * float(duration)
    dz = float(vz) * float(duration)
    return {"dx": dx, "dy": dy, "dz": dz}


def pose_measurements(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Measure world and phase-local displacement between two poses."""

    dx = _value(after, "x") - _value(before, "x")
    dy = _value(after, "y") - _value(before, "y")
    dz = _value(after, "z") - _value(before, "z")
    yaw_delta = yaw_deg(after) - yaw_deg(before) if before and after else None
    yaw_rad = math.radians(yaw_deg(before))
    forward = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
    right = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
    return {
        "world_delta": {"dx": round(dx, 3), "dy": round(dy, 3), "dz": round(dz, 3)},
        "pose_delta": {"dx": round(dx, 3), "dy": round(dy, 3), "dz": round(dz, 3)},
        "forward_displacement_m": round(forward, 3),
        "right_displacement_m": round(right, 3),
        "up_displacement_m": round(-dz, 3),
        "planar_displacement_m": round((dx * dx + dy * dy) ** 0.5, 3),
        "vertical_delta_m": round(dz, 3),
        "yaw_delta_deg": round(normalize_yaw_delta_deg(yaw_delta), 3) if yaw_delta is not None else None,
    }


def _value(pose: dict[str, Any], key: str) -> float:
    value = pose_value(pose, key)
    return 0.0 if value is None else float(value)
