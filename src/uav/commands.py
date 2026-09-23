"""Shared UAV primitive command argument normalization.

Backends still own simulator-specific execution, but direction names such as
``forward`` or ``up`` should resolve to the same body-frame velocity contract
everywhere. Model-facing code uses FlyAgent's canonical vertical convention:
positive ``vz`` means up and negative ``vz`` means down. Backends pass a
different ``vertical_sign`` only when converting that canonical command into a
simulator-specific raw velocity.
"""

from __future__ import annotations

from typing import Any


AIRSIM_NED_VERTICAL_SIGN = -1.0
CANONICAL_VERTICAL_SIGN = 1.0
UNREALCV_VERTICAL_SIGN = 1.0

# Compact model-facing meanings for FlyAgent's canonical body-frame arguments.
# Execution adapters and planner context must share these conventions.
MOVE_BODY_ARGUMENT_SEMANTICS = {
    "vx": "Body-frame velocity in m/s: positive moves forward; negative moves backward.",
    "vy": "Body-frame velocity in m/s: positive moves right; negative moves left.",
    "vz": "Body-frame velocity in m/s: positive moves up; negative moves down.",
    "duration": "Positive duration in seconds.",
}


def move_body_velocity_from_args(
    args: dict[str, Any],
    action_config: Any,
    *,
    vertical_sign: float = CANONICAL_VERTICAL_SIGN,
) -> tuple[float, float, float, float]:
    """Resolve move_body args into body-frame ``vx, vy, vz, duration``."""

    duration = float(args.get("duration", action_config.move_duration_sec))
    direction = str(args.get("direction", "")).lower()
    if direction:
        vx, vy, vz = velocity_for_direction(direction, action_config, vertical_sign=vertical_sign)
        return vx, vy, vz, duration
    return (
        float(args.get("vx", action_config.linear_velocity_mps)),
        float(args.get("vy", 0.0)),
        vertical_sign * float(args.get("vz", 0.0)),
        duration,
    )


def velocity_for_direction(
    direction: str,
    action_config: Any,
    *,
    vertical_sign: float = CANONICAL_VERTICAL_SIGN,
) -> tuple[float, float, float]:
    """Return body-frame velocity for a named movement direction."""

    speed = float(action_config.linear_velocity_mps)
    vertical_speed = float(action_config.vertical_velocity_mps)
    direction_map = {
        "forward": (speed, 0.0, 0.0),
        "backward": (-speed, 0.0, 0.0),
        "left": (0.0, -speed, 0.0),
        "right": (0.0, speed, 0.0),
        "up": (0.0, 0.0, vertical_sign * vertical_speed),
        "down": (0.0, 0.0, -vertical_sign * vertical_speed),
    }
    try:
        return direction_map[str(direction).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported move_body direction: {direction}") from exc
