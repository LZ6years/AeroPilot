"""Define FlyAgent's UAV primitive interface and dry-run implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PrimitiveResult:
    ok: bool
    message: str = ""
    data: dict[str, Any] | None = None


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

    def ascend(self, meters: float) -> PrimitiveResult:
        # Move upward by a bounded distance.
        ...

    def descend(self, meters: float) -> PrimitiveResult:
        # Move downward by a bounded distance.
        ...

    def get_pose(self) -> PrimitiveResult:
        # Return raw pose and motion state.
        ...

    def get_frames(self, modalities: list[str]) -> PrimitiveResult:
        # Return raw image/depth frame references or data.
        ...


class NullUavPrimitiveAdapter:
    """Deterministic primitive adapter for dry-run validation.

    It is useful before a real AirSim/UE scene is available and keeps the same
    interface that real adapters must implement.
    """

    def __init__(self) -> None:
        # Initialize a deterministic in-memory UAV state.
        self.connected = False
        self.flying = False
        self.yaw_deg = 0.0
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}

    def connect(self) -> PrimitiveResult:
        # Mark the dry-run UAV as connected.
        self.connected = True
        return PrimitiveResult(True, "null UAV connected")

    def takeoff(self) -> PrimitiveResult:
        # Simulate takeoff by moving to a small negative NED altitude.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        self.flying = True
        self.position["z"] = -2.0
        return PrimitiveResult(True, "null UAV takeoff", {"pose": self._pose()})

    def hover(self) -> PrimitiveResult:
        # Return the current pose without changing simulated state.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        return PrimitiveResult(True, "null UAV hover", {"pose": self._pose()})

    def land(self) -> PrimitiveResult:
        # Simulate landing by returning to ground altitude.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        self.flying = False
        self.position["z"] = 0.0
        return PrimitiveResult(True, "null UAV land", {"pose": self._pose()})

    def move_body(self, vx: float, vy: float, vz: float, duration: float) -> PrimitiveResult:
        # Integrate body-frame velocity into the simple pose state.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        self.position["x"] += vx * duration
        self.position["y"] += vy * duration
        self.position["z"] += vz * duration
        return PrimitiveResult(True, "null UAV move_body", {"pose": self._pose()})

    def turn_relative(self, yaw_deg: float) -> PrimitiveResult:
        # Add the relative yaw to the simulated heading.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        self.yaw_deg += yaw_deg
        return PrimitiveResult(True, "null UAV turn_relative", {"pose": self._pose()})

    def ascend(self, meters: float) -> PrimitiveResult:
        # Convert ascend into negative-Z NED motion.
        return self.move_body(0.0, 0.0, -abs(meters), 1.0)

    def descend(self, meters: float) -> PrimitiveResult:
        # Convert descend into positive-Z NED motion.
        return self.move_body(0.0, 0.0, abs(meters), 1.0)

    def get_pose(self) -> PrimitiveResult:
        # Return the current dry-run pose.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        return PrimitiveResult(True, "null UAV pose", {"pose": self._pose()})

    def get_frames(self, modalities: list[str]) -> PrimitiveResult:
        # Return mock frame metadata for requested modalities.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        frames = {modality: {"artifact": None, "mock": True} for modality in modalities}
        return PrimitiveResult(True, "null UAV frames", {"frames": frames})

    def _pose(self) -> dict[str, float]:
        # Build a serializable pose dictionary.
        return {**self.position, "yaw_deg": self.yaw_deg}
