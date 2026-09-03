"""Enrich canonical simulator poses with optional ground clearance.

This module is deliberately internal to simulator Backends.  EnvClient keeps
its frozen v1 observation API while each Backend remains responsible for
turning deployment-specific calibration into canonical, optional pose data.
An unavailable or unreliable ground reference must stay absent rather than be
represented by a fabricated zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class GroundClearanceResolver:
    """Resolve AGL from a calibrated landed vehicle-origin height.

    ``landed_origin_z_m`` is the canonical positive-up world Z of the
    controlled vehicle origin when it rests on the local landing reference.
    Keeping the model-origin offset inside this single calibration value avoids
    exposing separate ground-plane and mesh-offset parameters to callers.
    """

    landed_origin_z_m: float | None = None

    def enrich_pose(self, pose: dict[str, Any]) -> dict[str, Any]:
        """Return a pose copy with ``agl_m`` only when calibration is usable."""

        enriched = dict(pose) if isinstance(pose, dict) else {}
        if self.landed_origin_z_m is None:
            return enriched
        z = _finite_float(enriched.get("z"))
        if z is None:
            return enriched
        enriched["agl_m"] = max(0.0, z - self.landed_origin_z_m)
        return enriched


def enrich_pose_with_agl(
    pose: dict[str, Any],
    *,
    landed_origin_z_m: float | None,
) -> dict[str, Any]:
    """Convenience wrapper used by Backends at their pose response boundary."""

    return GroundClearanceResolver(landed_origin_z_m).enrich_pose(pose)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
