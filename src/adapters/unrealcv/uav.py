"""Adapt UnrealCV protocol I/O to canonical UAV primitives.

Scene/process ownership stays in SimulationServer. This adapter receives only
the resolved endpoint and protocol configuration, and owns its SDK connection.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.uav import body_velocity_to_world_delta
from src.uav.frames import normalize_yaw_deg
from ..uav_primitives import PrimitiveResult
from .control import UnrealCvPoseControl, create_pose_control
from .capture import UnrealCvCapture
from .client import close_unrealcv_client


class UnrealCvUavAdapter:
    """Compose pose control and capture without depending on scene schemas."""

    def __init__(self, host: str, port: int, config: dict[str, Any], *,
                 artifact_dir: str | Path | None = None,
                 save_artifacts: bool = False) -> None:
        self.host, self.port = host, int(port)
        self.config = deepcopy(config)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = save_artifacts
        self.client: Any | None = None
        self.control: UnrealCvPoseControl | None = None
        self.capture: UnrealCvCapture | None = None

    def connect(self) -> PrimitiveResult:
        self._connect()
        return PrimitiveResult(True, "UnrealCV connected")

    def close(self) -> None:
        """Release only our SDK client; subsequent primitives reconnect lazily."""
        close_unrealcv_client(self.client)
        self.client = self.control = self.capture = None

    def takeoff(self) -> PrimitiveResult:
        return PrimitiveResult(False, "UnrealCV does not support takeoff",
                               {"pose": self.get_pose().data})

    def land(self) -> PrimitiveResult:
        return PrimitiveResult(False, "UnrealCV does not support land",
                               {"pose": self.get_pose().data})

    def hover(self) -> PrimitiveResult:
        pose = self._get_pose()
        self._set_pose(pose)
        return PrimitiveResult(True, "UnrealCV pose held", {"pose": self._pose_to_model(pose)})

    def move_body(self, vx: float, vy: float, vz: float, duration: float) -> PrimitiveResult:
        pose = self._get_pose()
        delta = body_velocity_to_world_delta(vx, vy, vz, duration, pose[4])
        for index, axis in enumerate(("dx", "dy", "dz")):
            pose[index] += delta[axis]
        self._set_pose(pose)
        return PrimitiveResult(True, "UnrealCV location updated", {"pose": self._pose_to_model(pose)})

    def turn_relative(self, yaw_deg: float) -> PrimitiveResult:
        pose = self._get_pose()
        wrapped = normalize_yaw_deg(pose[4] + yaw_deg)
        pose[4] = 180.0 if wrapped == -180.0 else wrapped
        self._set_pose(pose)
        return PrimitiveResult(True, "UnrealCV yaw updated", {"pose": self._pose_to_model(pose)})

    def set_pose(self, x: float, y: float, z: float, yaw_deg: float = 0.0,
                 pitch_deg: float = 0.0, roll_deg: float = 0.0,
                 ignore_collision: bool = True) -> PrimitiveResult:
        # UnrealCV placement has no portable collision-sweep flag.
        pose = [x, y, z, pitch_deg, yaw_deg, roll_deg]
        self._set_pose(pose)
        return PrimitiveResult(True, "UnrealCV pose set", {"pose": self._pose_to_model(pose)})

    def ascend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        return self.move_body(0.0, 0.0, abs(speed_mps), duration)

    def descend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        return self.move_body(0.0, 0.0, -abs(speed_mps), duration)

    def get_pose(self) -> PrimitiveResult:
        return PrimitiveResult(True, "UnrealCV pose", self._pose_to_model(self._get_pose()))

    def get_frames(self, modalities: list[str],
                   capture_sources: dict[str, str] | None = None) -> PrimitiveResult:
        self._connect()
        if self.capture is None:
            raise RuntimeError("UnrealCV capture was not initialized")
        return PrimitiveResult(True, "UnrealCV frames", {"frames": self.capture.capture(modalities, capture_sources)})

    def _connect(self) -> None:
        """Open the client lazily and initialize the configured strategy once."""

        if self.client is not None:
            return

        from unrealcv import Client

        client = Client((self.host, self.port))
        try:
            client.connect()
            if not client.isconnected():
                raise ConnectionError(
                    f"Could not connect to UnrealCV at {self.host}:{self.port}"
                )
            control = create_pose_control(client, self.config)
            control.prepare()
            actor = dict(self.config.get("actor", {}))
            units_per_meter = (
                float(actor.get("units_per_meter", 100.0))
                if self.config.get("control_target") == "actor"
                else 1.0
            )
            capture = UnrealCvCapture(
                client,
                camera_id=str(self.config.get("camera_id", 0)),
                units_per_meter=units_per_meter,
                artifact_dir=self.artifact_dir,
                save_artifacts=self.save_artifacts,
                sync_camera=control.sync_camera,
            )
        except Exception as exc:
            close_unrealcv_client(client, original_error=exc)
            raise
        self.client = client
        self.control = control
        self.capture = capture

    def _get_pose(self) -> list[float]:
        self._connect()
        return list(self._require_control().get_pose())

    def _set_pose(self, pose: list[float]) -> None:
        self._connect()
        self._require_control().set_pose(pose)

    def _pose_to_model(self, pose: list[float]) -> dict[str, float | str]:
        coordinate_frame = (
            self.control.coordinate_frame
            if self.control is not None
            else (
                "unrealcv_world_m"
                if str(self.config.get("control_target", "camera")) == "actor"
                else "unrealcv_camera_native"
            )
        )
        model_pose: dict[str, float | str] = {
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "pitch_deg": float(pose[3]),
            "yaw_deg": float(pose[4]),
            "roll_deg": float(pose[5]),
            "coordinate_frame": coordinate_frame,
        }
        return model_pose

    def _require_control(self) -> UnrealCvPoseControl:
        if self.control is None:
            raise RuntimeError("UnrealCV pose control was not initialized")
        return self.control
