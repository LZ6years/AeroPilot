"""Deterministic UAV test double. Never imported by production execution."""

from src.adapters.uav_primitives import PrimitiveResult
from src.uav import body_velocity_to_world_delta

class FakeUavPrimitiveAdapter:
    """Deterministic primitive adapter for test validation.

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
        # Mark the test UAV as connected.
        self.connected = True
        return PrimitiveResult(True, "fake UAV connected")

    def takeoff(self) -> PrimitiveResult:
        # Simulate takeoff by moving to a small positive canonical altitude.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        self.flying = True
        self.position["z"] = 2.0
        return PrimitiveResult(True, "fake UAV takeoff", {"pose": self._pose()})

    def hover(self) -> PrimitiveResult:
        # Return the current pose without changing simulated state.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        return PrimitiveResult(True, "fake UAV hover", {"pose": self._pose()})

    def land(self) -> PrimitiveResult:
        # Simulate landing by returning to ground altitude.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        self.flying = False
        self.position["z"] = 0.0
        return PrimitiveResult(True, "fake UAV land", {"pose": self._pose()})

    def move_body(self, vx: float, vy: float, vz: float, duration: float) -> PrimitiveResult:
        # Integrate body-frame velocity into the simple pose state.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        delta = body_velocity_to_world_delta(vx, vy, vz, duration, self.yaw_deg)
        self.position["x"] += delta["dx"]
        self.position["y"] += delta["dy"]
        self.position["z"] += delta["dz"]
        return PrimitiveResult(True, "fake UAV move_body", {"pose": self._pose()})

    def turn_relative(self, yaw_deg: float) -> PrimitiveResult:
        # Add the relative yaw to the simulated heading.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        self.yaw_deg += yaw_deg
        return PrimitiveResult(True, "fake UAV turn_relative", {"pose": self._pose()})

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
        # Directly place the test UAV at an absolute pose.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        self.position = {"x": x, "y": y, "z": z}
        self.yaw_deg = yaw_deg
        return PrimitiveResult(
            True,
            "fake UAV set_pose",
            {
                "pose": {
                    **self._pose(),
                    "pitch_deg": pitch_deg,
                    "roll_deg": roll_deg,
                    "ignore_collision": ignore_collision,
                }
            },
        )

    def ascend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        # Convert ascend into positive-Z canonical velocity over time.
        return self.move_body(0.0, 0.0, abs(speed_mps), duration)

    def descend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        # Convert descend into negative-Z canonical velocity over time.
        return self.move_body(0.0, 0.0, -abs(speed_mps), duration)

    def get_pose(self) -> PrimitiveResult:
        # Return the current test pose.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        return PrimitiveResult(True, "fake UAV pose", {"pose": self._pose()})

    def get_frames(
        self,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> PrimitiveResult:
        # Return mock frame metadata for requested modalities.
        if not self.connected:
            return PrimitiveResult(False, "UAV is not connected")
        frames = {modality: {"artifact": None, "mock": True} for modality in modalities}
        return PrimitiveResult(True, "fake UAV frames", {"frames": frames})

    def _pose(self) -> dict[str, float]:
        # Build a serializable pose dictionary.
        return {**self.position, "yaw_deg": self.yaw_deg}
