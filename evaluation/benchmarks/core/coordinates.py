"""Pure coordinate math shared by dataset-specific adapters."""

from __future__ import annotations

import math
from typing import Sequence


def quaternion_wxyz_to_euler_deg(
    quaternion: Sequence[float],
) -> tuple[float, float, float]:
    """Return yaw, pitch, roll degrees for a normalized WXYZ quaternion."""

    if len(quaternion) != 4:
        raise ValueError("WXYZ quaternion must contain four values")
    values = tuple(float(value) for value in quaternion)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("WXYZ quaternion values must be finite")
    w, x, y, z = values
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("WXYZ quaternion norm must be positive")
    w, x, y, z = (value / norm for value in values)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


def ned_to_canonical_position(position: Sequence[float]) -> tuple[float, float, float]:
    """Convert AirSim NED meters to canonical positive-up meters."""

    if len(position) < 3:
        raise ValueError("NED position must contain at least three values")
    x, y, z = (float(position[index]) for index in range(3))
    if any(not math.isfinite(value) for value in (x, y, z)):
        raise ValueError("NED position values must be finite")
    return x, y, -z
