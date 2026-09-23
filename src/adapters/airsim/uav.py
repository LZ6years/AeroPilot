"""Implement FlyAgent's own AirSim UAV primitive adapter.

The public adapter contract uses FlyAgent's canonical vertical frame:
positive Z and positive vertical velocity mean up. AirSim's SDK uses NED Z, so
this adapter converts at the SDK boundary and never exposes NED values to
runtime/model-facing layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.uav import yaw_deg

from ..uav_primitives import PrimitiveResult
from .client import close_airsim_client
from .capture import AirSimCapture


from .control import (
    AirSimControlStrategy,
    POSE_KINEMATIC_CONTROL,
    PAUSED_MULTIROTOR_CONTROL,
    PoseKinematicControlStrategy,
    PausedMultirotorControlStrategy,
    resolve_airsim_control_strategy,
)


class AirSimUavAdapter:
    """FlyAgent-owned AirSim primitive adapter.

    This is the real adapter shape to use when AirSim is available. It is not
    coupled to any benchmark's map, UI, task, or evaluation modules.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = "Drone_1",
        camera_name: str = "front_0",
        mode: str = "Multirotor",
        control_strategy: str | None = None,
        artifact_dir: str | Path | None = None,
        save_artifacts: bool = False,
    ) -> None:
        # Store AirSim connection and vehicle identity settings.
        self.host = host
        self.port = port
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.mode = mode
        self.control_strategy = resolve_airsim_control_strategy(
            mode,
            control_strategy,
        )
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = save_artifacts
        self.capture = AirSimCapture(camera_name, vehicle_name, self.artifact_dir, save_artifacts)
        self._client: Any | None = None
        strategy_type = {
            POSE_KINEMATIC_CONTROL: PoseKinematicControlStrategy,
            PAUSED_MULTIROTOR_CONTROL: PausedMultirotorControlStrategy,
        }[self.control_strategy]
        self._control: AirSimControlStrategy = strategy_type(
            self.host,
            self.port,
            self.vehicle_name,
            lambda: self._airsim(),
        )

    def connect(self) -> PrimitiveResult:
        # Open the mode-specific AirSim client.
        self.close()
        self._client = self._control.connect()
        return PrimitiveResult(
            True,
            f"AirSim {self.control_strategy} client connected",
            {"control_strategy": self.control_strategy},
        )

    def close(self) -> None:
        """Close the local RPC client without affecting the remote UE process."""

        if self._client is None:
            return
        close_airsim_client(self._client)
        self._client = None

    def reset(self) -> PrimitiveResult:
        """Reset AirSim state and reconnect the configured control mode."""

        client = self._require_client()
        try:
            result = self._control.reset(client)
        finally:
            self.close()
        reconnect = self.connect()
        if not reconnect.ok:
            return reconnect
        return result

    def takeoff(self) -> PrimitiveResult:
        return self._control.takeoff(self._require_client())

    def hover(self) -> PrimitiveResult:
        return self._control.hover(self._require_client())

    def land(self) -> PrimitiveResult:
        return self._control.land(self._require_client())

    def move_body(self, vx: float, vy: float, vz: float, duration: float) -> PrimitiveResult:
        return self._control.move_body(
            self._require_client(), vx, vy, vz, duration
        )

    def turn_relative(self, yaw_deg: float) -> PrimitiveResult:
        return self._control.turn_relative(self._require_client(), yaw_deg)

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
        return self._control.set_pose(
            self._require_client(),
            x,
            y,
            z,
            yaw_deg,
            pitch_deg,
            roll_deg,
            ignore_collision,
        )

    def ascend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        # Use canonical positive-Z velocity for upward motion.
        return self.move_body(0.0, 0.0, abs(speed_mps), duration)

    def descend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        # Use canonical negative-Z velocity for downward motion.
        return self.move_body(0.0, 0.0, -abs(speed_mps), duration)

    def get_pose(self) -> PrimitiveResult:
        return self._control.get_pose(self._require_client())

    @staticmethod
    def _canonical_position(position: Any) -> dict[str, float]:
        """Convert AirSim NED position into FlyAgent's positive-up pose."""

        return {
            "x": position.x_val,
            "y": position.y_val,
            "z": -position.z_val,
        }

    def get_frames(self, modalities: list[str], capture_sources: dict[str, str] | None = None) -> PrimitiveResult:
        """Delegate validated file/memory capture to the capture component."""
        return self.capture.get_frames(self._require_client(), self._airsim(), modalities, capture_sources)

    def _current_yaw_deg(self) -> float:
        # Read the current yaw in degrees for relative turns.
        pose = self.get_pose()
        return yaw_deg(pose.data or {})

    def _require_client(self) -> Any:
        # Fail early when flight commands are issued before connect.
        if self._client is None:
            raise RuntimeError("AirSim client is not connected")
        return self._client

    @staticmethod
    def _airsim() -> Any:
        # Import AirSim lazily so injected SDK tests do not require the package.
        import airsim

        return airsim

    def _is_computer_vision(self) -> bool:
        # Detect AirSim ComputerVision mode, which has observations but no UAV flight APIs.
        return self.mode.lower() in {"computervision", "computer_vision", "cv"}
