"""Define FlyAgent's simulator-independent UAV primitive interface.

Primitive adapters expose FlyAgent's canonical pose and velocity convention:
positive Z means upward motion, negative Z means downward motion. Simulator
backends that use another convention must convert at their own boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol



@dataclass(frozen=True)
class PrimitiveResult:
    ok: bool
    message: str = ""
    data: dict[str, Any] | None = None
    # Runtime-only control evidence; never part of model-visible tool data.
    execution_state: dict[str, Any] = field(default_factory=dict)


class UavPrimitiveAdapter(Protocol):
    """Thin interface for UAV primitive control.

    This layer intentionally exposes only primitive flight and raw observation
    capabilities. It must not contain semantic tasks such as "search target" or
    "face landmark".
    """

    def connect(self) -> PrimitiveResult:
        # Connect the primitive UAV control channel.
        ...

    def takeoff(self) -> PrimitiveResult:
        # Lift the UAV into a controllable airborne state.
        ...

    def hover(self) -> PrimitiveResult:
        # Hold the UAV at its current pose.
        ...

    def land(self) -> PrimitiveResult:
        # Land the UAV safely.
        ...

    def move_body(self, vx: float, vy: float, vz: float, duration: float) -> PrimitiveResult:
        # Move using body-frame velocity for a short duration.
        ...

    def turn_relative(self, yaw_deg: float) -> PrimitiveResult:
        # Rotate the UAV by a relative yaw angle.
        ...

    def set_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        ignore_collision: bool = True,
    ) -> PrimitiveResult:
        # Teleport/place the UAV at an absolute simulator pose.
        ...

    def ascend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        # Move upward using vertical speed for a bounded duration.
        ...

    def descend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        # Move downward using vertical speed for a bounded duration.
        ...

    def get_pose(self) -> PrimitiveResult:
        # Return raw pose and motion state.
        ...

    def get_frames(
        self,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> PrimitiveResult:
        # Return raw image/depth frame references or data.
        ...
