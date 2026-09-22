"""UAV pose and coordinate-frame helpers shared across runtime layers.

These functions normalize the pose shapes produced by dry-run adapters,
AirSim, UnrealCV, run summaries, and context modules. They intentionally avoid
calling simulator SDKs; adapters own I/O, while this module owns pure pose/yaw
interpretation.
"""

from __future__ import annotations

import math
from typing import Any


SIGNED_YAW_SEMANTICS = "Signed relative yaw in degrees: negative turns left; positive turns right."


def pose_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Extract a pose dictionary from a raw observation-like payload."""

    pose = observation.get("pose") if isinstance(observation, dict) else None
    return dict(pose) if isinstance(pose, dict) else {}


def pose_value(pose: dict[str, Any], key: str) -> float | None:
    """Read a scalar pose value from flat, nested, or orientation pose shapes."""

    if not isinstance(pose, dict):
        return None
    if key in pose:
        return _safe_float(pose.get(key))
    if key in {"x", "y", "z"}:
        position = pose.get("position")
        if isinstance(position, dict):
            return _safe_float(position.get(key))
    orientation = pose.get("orientation")
    if isinstance(orientation, dict) and key in orientation:
        return _safe_float(orientation.get(key))
    return None


def yaw_deg(pose: dict[str, Any]) -> float:
    """Return yaw in degrees from supported pose shapes."""

    if not isinstance(pose, dict):
        return 0.0
    if "yaw_deg" in pose:
        return float(pose.get("yaw_deg") or 0.0)
    orientation = pose.get("orientation")
    if isinstance(orientation, dict) and "yaw_deg" in orientation:
        return float(orientation.get("yaw_deg") or 0.0)
    euler_rad = pose.get("euler_rad")
    if isinstance(euler_rad, dict) and "yaw" in euler_rad:
        return math.degrees(float(euler_rad.get("yaw") or 0.0))
    euler_deg = pose.get("euler_deg")
    if isinstance(euler_deg, dict) and "yaw" in euler_deg:
        return float(euler_deg.get("yaw") or 0.0)
    return 0.0


def normalize_yaw_deg(value: float) -> float:
    """Wrap an absolute heading into [-180, 180)."""

    return (float(value) + 180.0) % 360.0 - 180.0


def position_distance_m(before: dict[str, Any] | None, after: dict[str, Any] | None) -> float | None:
    """Measure unrounded XYZ distance; unavailable/nonfinite poses stay unknown."""

    if not before or not after:
        return None
    first = [pose_value(before, axis) for axis in ("x", "y", "z")]
    second = [pose_value(after, axis) for axis in ("x", "y", "z")]
    if any(value is None or not math.isfinite(value) for value in first + second):
        return None
    return math.dist(first, second)


def normalize_yaw_delta_deg(delta: float) -> float:
    """Normalize a yaw delta to [-180, 180], preserving +180 for positive turns."""

    normalized = normalize_yaw_deg(delta)
    return 180.0 if normalized == -180.0 and delta > 0 else normalized


def pose_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None]:
    """Return a compact pose delta with normalized yaw difference."""

    return {
        "dx": _subtract(pose_value(after, "x"), pose_value(before, "x")),
        "dy": _subtract(pose_value(after, "y"), pose_value(before, "y")),
        "dz": _subtract(pose_value(after, "z"), pose_value(before, "z")),
        "dyaw_deg": _yaw_delta(before, after),
    }


def clean_pose_for_model(pose: dict[str, Any]) -> dict[str, Any]:
    """Remove high-rate raw fields that should not drive model phase judgment."""

    data = dict(pose) if isinstance(pose, dict) else {}
    data.pop("velocity", None)
    return data


def velocity_norm_mps(pose: dict[str, Any]) -> float | None:
    """Return speed norm from a pose velocity field, if present."""

    if not isinstance(pose, dict):
        return None
    velocity = pose.get("velocity")
    if not isinstance(velocity, dict):
        return None
    vx = _safe_float(velocity.get("x"))
    vy = _safe_float(velocity.get("y"))
    vz = _safe_float(velocity.get("z"))
    if vx is None or vy is None or vz is None:
        return None
    return (vx * vx + vy * vy + vz * vz) ** 0.5


def _yaw_delta(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    if not before or not after:
        return None
    return round(normalize_yaw_delta_deg(yaw_deg(after) - yaw_deg(before)), 3)


def _subtract(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return round(float(after) - float(before), 3)


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
