"""Implement FlyAgent's own AirSim UAV primitive adapter."""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Any

from .uav_primitives import PrimitiveResult


class AirSimUavAdapter:
    """FlyAgent-owned AirSim primitive adapter.

    This is the real adapter shape to use when AirSim is available. It is not
    coupled to AirNavigation's map, UI, or task modules.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = "Drone_1",
        camera_name: str = "front_0",
        mode: str = "Multirotor",
        artifact_dir: str | Path | None = None,
        save_artifacts: bool = False,
    ) -> None:
        # Store AirSim connection and vehicle identity settings.
        self.host = host
        self.port = port
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.mode = mode
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = save_artifacts
        self._frame_index = 0
        self._client: Any | None = None

    def connect(self) -> PrimitiveResult:
        # Open the AirSim client and enable flight control when available.
        airsim = self._airsim()
        if self._is_computer_vision():
            self._client = airsim.VehicleClient(ip=self.host, port=self.port, timeout_value=10)
            self._client.confirmConnection()
            return PrimitiveResult(True, "AirSim ComputerVision client connected")
        self._client = airsim.MultirotorClient(ip=self.host, port=self.port, timeout_value=10)
        self._client.confirmConnection()
        self._client.enableApiControl(True, self.vehicle_name)
        self._client.armDisarm(True, self.vehicle_name)
        return PrimitiveResult(True, "AirSim UAV connected")

    def takeoff(self) -> PrimitiveResult:
        # Execute AirSim's blocking takeoff primitive.
        self._require_client().takeoffAsync(vehicle_name=self.vehicle_name).join()
        return PrimitiveResult(True, "AirSim UAV takeoff")

    def hover(self) -> PrimitiveResult:
        # Ask AirSim to hold the current vehicle pose.
        self._require_client().hoverAsync(vehicle_name=self.vehicle_name).join()
        return PrimitiveResult(True, "AirSim UAV hover")

    def land(self) -> PrimitiveResult:
        # Execute AirSim's blocking land primitive.
        self._require_client().landAsync(vehicle_name=self.vehicle_name).join()
        return PrimitiveResult(True, "AirSim UAV land")

    def move_body(self, vx: float, vy: float, vz: float, duration: float) -> PrimitiveResult:
        # Send a short body-frame velocity command.
        self._require_client().moveByVelocityBodyFrameAsync(
            vx, vy, vz, duration, vehicle_name=self.vehicle_name
        ).join()
        return PrimitiveResult(True, "AirSim UAV move_body")

    def turn_relative(self, yaw_deg: float) -> PrimitiveResult:
        # Convert a relative yaw command into AirSim's absolute yaw command.
        current_yaw = self._current_yaw_deg()
        self._require_client().rotateToYawAsync(
            current_yaw + yaw_deg, vehicle_name=self.vehicle_name
        ).join()
        return PrimitiveResult(True, "AirSim UAV turn_relative")

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
        # Place the AirSim vehicle/camera at an absolute NED pose for fast resets.
        airsim = self._airsim()
        pose = airsim.Pose(
            airsim.Vector3r(x, y, z),
            airsim.to_quaternion(
                math.radians(pitch_deg),
                math.radians(roll_deg),
                math.radians(yaw_deg),
            ),
        )
        self._require_client().simSetVehiclePose(
            pose,
            ignore_collision=ignore_collision,
            vehicle_name=self.vehicle_name,
        )
        return PrimitiveResult(
            True,
            "AirSim UAV set_pose",
            {
                "pose": {
                    "position": {"x": x, "y": y, "z": z},
                    "euler_deg": {
                        "pitch": pitch_deg,
                        "roll": roll_deg,
                        "yaw": yaw_deg,
                    },
                    "ignore_collision": ignore_collision,
                }
            },
        )

    def ascend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        # Use NED negative-Z velocity for upward motion over a fixed duration.
        return self.move_body(0.0, 0.0, -abs(speed_mps), duration)

    def descend(self, speed_mps: float, duration: float) -> PrimitiveResult:
        # Use NED positive-Z velocity for downward motion over a fixed duration.
        return self.move_body(0.0, 0.0, abs(speed_mps), duration)

    def get_pose(self) -> PrimitiveResult:
        # Read AirSim kinematics and normalize them into dictionaries.
        if self._is_computer_vision():
            pose = self._require_client().simGetVehiclePose(vehicle_name=self.vehicle_name)
            pitch, roll, yaw = self._airsim().to_eularian_angles(pose.orientation)
            return PrimitiveResult(
                True,
                "AirSim ComputerVision pose",
                {
                    "position": {
                        "x": pose.position.x_val,
                        "y": pose.position.y_val,
                        "z": pose.position.z_val,
                    },
                    "euler_rad": {"pitch": pitch, "roll": roll, "yaw": yaw},
                    "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
            )
        state = self._require_client().getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        airsim = self._airsim()
        pitch, roll, yaw = airsim.to_eularian_angles(kin.orientation)
        data = {
            "position": {
                "x": kin.position.x_val,
                "y": kin.position.y_val,
                "z": kin.position.z_val,
            },
            "euler_rad": {"pitch": pitch, "roll": roll, "yaw": yaw},
            "velocity": {
                "x": kin.linear_velocity.x_val,
                "y": kin.linear_velocity.y_val,
                "z": kin.linear_velocity.z_val,
            },
        }
        return PrimitiveResult(True, "AirSim UAV pose", data)

    def get_frames(self, modalities: list[str]) -> PrimitiveResult:
        # Convert requested modalities into AirSim image requests.
        airsim = self._airsim()
        requests = []
        for modality in modalities:
            image_type = self._image_type(modality)
            pixels_as_float = self._is_depth_modality(modality)
            compress = self.save_artifacts and not pixels_as_float
            requests.append(airsim.ImageRequest(self.camera_name, image_type, pixels_as_float, compress))
        responses = self._require_client().simGetImages(requests, vehicle_name=self.vehicle_name)
        frames = {
            modality: self._frame_metadata(modality, response)
            for modality, response in zip(modalities, responses)
        }
        return PrimitiveResult(True, "AirSim UAV frames", {"frames": frames})

    def _frame_metadata(self, modality: str, response: Any) -> dict[str, Any]:
        # Build frame metadata and optionally save raw frame data as local artifacts.
        metadata = {
            "width": response.width,
            "height": response.height,
            "image_type": str(self._image_type(modality)),
            "bytes": len(response.image_data_uint8 or b""),
            "float_count": len(response.image_data_float or []),
        }
        if self.save_artifacts and self.artifact_dir is not None:
            artifact_path = self._save_frame_artifact(modality, response)
            if artifact_path is not None:
                metadata["artifact_path"] = str(artifact_path)
        return metadata

    def _save_frame_artifact(self, modality: str, response: Any) -> Path | None:
        # Save RGB bytes or depth float arrays for later local inspection.
        self._frame_index += 1
        normalized = modality.lower()
        if self._is_depth_modality(normalized) and response.image_data_float:
            import numpy as np

            path = self.artifact_dir / "depth" / f"frame_{self._frame_index:06d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            buffer = BytesIO()
            np.save(buffer, np.array(response.image_data_float, dtype=np.float32))
            path.write_bytes(buffer.getvalue())
            return path
        if response.image_data_uint8:
            suffix = "png" if normalized in {"rgb", "scene", "color"} else "bin"
            folder = "rgb" if normalized in {"rgb", "scene", "color"} else normalized
            path = self.artifact_dir / folder / f"frame_{self._frame_index:06d}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(response.image_data_uint8))
            return path
        return None

    def _current_yaw_deg(self) -> float:
        # Read the current yaw in degrees for relative turns.
        pose = self.get_pose()
        yaw_rad = float(pose.data["euler_rad"]["yaw"]) if pose.data else 0.0
        return yaw_rad * 180.0 / 3.141592653589793

    def _require_client(self) -> Any:
        # Fail early when flight commands are issued before connect.
        if self._client is None:
            raise RuntimeError("AirSim client is not connected")
        return self._client

    @staticmethod
    def _airsim() -> Any:
        # Import AirSim lazily so dry-run tests do not require the package.
        import airsim

        return airsim

    def _image_type(self, modality: str) -> Any:
        # Map FlyAgent modality names onto AirSim image enum values.
        airsim = self._airsim()
        mapping = {
            "rgb": airsim.ImageType.Scene,
            "scene": airsim.ImageType.Scene,
            "depth": airsim.ImageType.DepthPerspective,
            "depth_perspective": airsim.ImageType.DepthPerspective,
            "segmentation": airsim.ImageType.Segmentation,
            "mask": airsim.ImageType.Segmentation,
        }
        return mapping.get(modality.lower(), airsim.ImageType.Scene)

    @staticmethod
    def _is_depth_modality(modality: str) -> bool:
        # Detect depth-like modalities that should request float pixels from AirSim.
        return modality.lower() in {"depth", "depth_perspective"}

    def _is_computer_vision(self) -> bool:
        # Detect AirSim ComputerVision mode, which has observations but no UAV flight APIs.
        return self.mode.lower() in {"computervision", "computer_vision", "cv"}
