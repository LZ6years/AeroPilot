"""Implement FlyAgent's own AirSim UAV primitive adapter."""

from __future__ import annotations

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
    ) -> None:
        # Store AirSim connection and vehicle identity settings.
        self.host = host
        self.port = port
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self._client: Any | None = None

    def connect(self) -> PrimitiveResult:
        # Open the AirSim client and enable API control.
        airsim = self._airsim()
        self._client = airsim.MultirotorClient(ip=self.host, port=self.port)
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

    def ascend(self, meters: float) -> PrimitiveResult:
        # Use NED negative-Z velocity for upward motion.
        return self.move_body(0.0, 0.0, -abs(meters), 1.0)

    def descend(self, meters: float) -> PrimitiveResult:
        # Use NED positive-Z velocity for downward motion.
        return self.move_body(0.0, 0.0, abs(meters), 1.0)

    def get_pose(self) -> PrimitiveResult:
        # Read AirSim kinematics and normalize them into dictionaries.
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
            pixels_as_float = modality.lower() in {"depth", "depth_perspective"}
            requests.append(airsim.ImageRequest(self.camera_name, image_type, pixels_as_float, False))
        responses = self._require_client().simGetImages(requests, vehicle_name=self.vehicle_name)
        frames = {
            modality: {
                "width": response.width,
                "height": response.height,
                "image_type": str(self._image_type(modality)),
                "bytes": len(response.image_data_uint8 or b""),
                "float_count": len(response.image_data_float or []),
            }
            for modality, response in zip(modalities, responses)
        }
        return PrimitiveResult(True, "AirSim UAV frames", {"frames": frames})

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
