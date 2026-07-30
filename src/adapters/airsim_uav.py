"""Implement FlyAgent's own AirSim UAV primitive adapter.

The public adapter contract uses FlyAgent's canonical vertical frame:
positive Z and positive vertical velocity mean up. AirSim's SDK uses NED Z, so
this adapter converts at the SDK boundary and never exposes NED values to
runtime/model-facing layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

from src.uav import body_velocity_to_world_delta, yaw_deg

from .uav_primitives import PrimitiveResult


class AirSimControlStrategy(Protocol):
    """Define control-mode behavior while frame capture stays in the adapter."""

    def connect(self) -> Any: ...
    def reset(self, client: Any) -> PrimitiveResult: ...
    def takeoff(self, client: Any) -> PrimitiveResult: ...
    def hover(self, client: Any) -> PrimitiveResult: ...
    def land(self, client: Any) -> PrimitiveResult: ...
    def move_body(
        self, client: Any, vx: float, vy: float, vz: float, duration: float
    ) -> PrimitiveResult: ...
    def turn_relative(self, client: Any, yaw_delta_deg: float) -> PrimitiveResult: ...
    def set_pose(
        self,
        client: Any,
        x: float,
        y: float,
        z: float,
        yaw_value_deg: float,
        pitch_deg: float,
        roll_deg: float,
        ignore_collision: bool,
    ) -> PrimitiveResult: ...
    def get_pose(self, client: Any) -> PrimitiveResult: ...


class _AirSimStrategyBase:
    """Share endpoint, vehicle identity, and canonical pose conversion."""

    def __init__(
        self,
        host: str,
        port: int,
        vehicle_name: str,
        airsim_provider: Callable[[], Any],
    ) -> None:
        self.host = host
        self.port = port
        self.vehicle_name = vehicle_name
        self._airsim_provider = airsim_provider

    def _airsim(self) -> Any:
        return self._airsim_provider()

    def _pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw_value_deg: float,
        pitch_deg: float,
        roll_deg: float,
    ) -> Any:
        airsim = self._airsim()
        return airsim.Pose(
            airsim.Vector3r(x, y, -z),
            airsim.to_quaternion(
                math.radians(pitch_deg),
                math.radians(roll_deg),
                math.radians(yaw_value_deg),
            ),
        )

    def set_pose(
        self,
        client: Any,
        x: float,
        y: float,
        z: float,
        yaw_value_deg: float,
        pitch_deg: float,
        roll_deg: float,
        ignore_collision: bool,
    ) -> PrimitiveResult:
        client.simSetVehiclePose(
            self._pose(x, y, z, yaw_value_deg, pitch_deg, roll_deg),
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
                        "yaw": yaw_value_deg,
                    },
                    "ignore_collision": ignore_collision,
                }
            },
        )

    def _canonical_pose_result(
        self, pose: Any, message: str, velocity: dict[str, float] | None = None
    ) -> PrimitiveResult:
        pitch, roll, yaw = self._airsim().to_eularian_angles(pose.orientation)
        return PrimitiveResult(
            True,
            message,
            {
                "position": {
                    "x": pose.position.x_val,
                    "y": pose.position.y_val,
                    "z": -pose.position.z_val,
                },
                "euler_rad": {"pitch": pitch, "roll": roll, "yaw": yaw},
                "velocity": velocity or {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        )


class ComputerVisionControlStrategy(_AirSimStrategyBase):
    """Control AirSim ComputerVision by changing camera/vehicle pose."""

    def connect(self) -> Any:
        client = self._airsim().VehicleClient(
            ip=self.host, port=self.port, timeout_value=10
        )
        client.confirmConnection()
        return client

    def reset(self, client: Any) -> PrimitiveResult:
        client.reset()
        return PrimitiveResult(True, "AirSim ComputerVision reset")

    def takeoff(self, client: Any) -> PrimitiveResult:
        return PrimitiveResult(False, "ComputerVision does not support takeoff")

    def hover(self, client: Any) -> PrimitiveResult:
        return PrimitiveResult(
            True,
            "AirSim ComputerVision pose held",
            {"pose": self.get_pose(client).data or {}},
        )

    def land(self, client: Any) -> PrimitiveResult:
        return PrimitiveResult(False, "ComputerVision does not support land")

    def move_body(
        self, client: Any, vx: float, vy: float, vz: float, duration: float
    ) -> PrimitiveResult:
        pose = client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        pitch, roll, yaw = self._airsim().to_eularian_angles(pose.orientation)
        delta = body_velocity_to_world_delta(
            vx, vy, vz, duration, math.degrees(yaw)
        )
        pose.position.x_val += delta["dx"]
        pose.position.y_val += delta["dy"]
        pose.position.z_val -= delta["dz"]
        client.simSetVehiclePose(
            pose, ignore_collision=True, vehicle_name=self.vehicle_name
        )
        result = self._canonical_pose_result(
            pose, "AirSim ComputerVision move_body"
        )
        return PrimitiveResult(True, result.message, {"pose": result.data or {}})

    def turn_relative(self, client: Any, yaw_delta_deg: float) -> PrimitiveResult:
        pose = client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        pitch, roll, yaw = self._airsim().to_eularian_angles(pose.orientation)
        pose.orientation = self._airsim().to_quaternion(
            pitch, roll, yaw + math.radians(yaw_delta_deg)
        )
        client.simSetVehiclePose(
            pose, ignore_collision=True, vehicle_name=self.vehicle_name
        )
        result = self._canonical_pose_result(
            pose, "AirSim ComputerVision turn_relative"
        )
        return PrimitiveResult(True, result.message, {"pose": result.data or {}})

    def get_pose(self, client: Any) -> PrimitiveResult:
        pose = client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        return self._canonical_pose_result(pose, "AirSim ComputerVision pose")


class MultirotorControlStrategy(_AirSimStrategyBase):
    """Use native AirSim flight APIs for a physical multirotor."""

    def connect(self) -> Any:
        client = self._airsim().MultirotorClient(
            ip=self.host, port=self.port, timeout_value=10
        )
        client.confirmConnection()
        client.enableApiControl(True, self.vehicle_name)
        client.armDisarm(True, self.vehicle_name)
        return client

    def reset(self, client: Any) -> PrimitiveResult:
        client.reset()
        return PrimitiveResult(True, "AirSim Multirotor reset")

    def takeoff(self, client: Any) -> PrimitiveResult:
        client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        return PrimitiveResult(True, "AirSim UAV takeoff")

    def hover(self, client: Any) -> PrimitiveResult:
        client.hoverAsync(vehicle_name=self.vehicle_name).join()
        return PrimitiveResult(True, "AirSim UAV hover")

    def land(self, client: Any) -> PrimitiveResult:
        client.landAsync(vehicle_name=self.vehicle_name).join()
        return PrimitiveResult(True, "AirSim UAV land")

    def move_body(
        self, client: Any, vx: float, vy: float, vz: float, duration: float
    ) -> PrimitiveResult:
        client.moveByVelocityBodyFrameAsync(
            vx, vy, -vz, duration, vehicle_name=self.vehicle_name
        ).join()
        return PrimitiveResult(True, "AirSim UAV move_body")

    def turn_relative(self, client: Any, yaw_delta_deg: float) -> PrimitiveResult:
        pose = self.get_pose(client)
        current_yaw = yaw_deg(pose.data or {})
        client.rotateToYawAsync(
            current_yaw + yaw_delta_deg, vehicle_name=self.vehicle_name
        ).join()
        return PrimitiveResult(True, "AirSim UAV turn_relative")

    def get_pose(self, client: Any) -> PrimitiveResult:
        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        return self._canonical_pose_result(
            self._airsim().Pose(kin.position, kin.orientation),
            "AirSim UAV pose",
            {
                "x": kin.linear_velocity.x_val,
                "y": kin.linear_velocity.y_val,
                "z": -kin.linear_velocity.z_val,
            },
        )


@dataclass(frozen=True)
class _AirSimCaptureSpec:
    """Translate one generic capture source into stable AirSim request flags."""

    image_type_attribute: str
    pixels_as_float: bool
    compress: bool
    payload_kind: str


_AIRSIM_CAPTURE_SPECS = {
    # Request formats are intrinsic to the capture source.  Saving artifacts
    # must never change the simulator RPC contract.
    "scene": _AirSimCaptureSpec("Scene", False, False, "raw_color"),
    "depth_vis": _AirSimCaptureSpec("DepthVis", False, True, "depth_vis"),
    "depth_perspective": _AirSimCaptureSpec(
        "DepthPerspective", True, False, "float_depth"
    ),
    "segmentation": _AirSimCaptureSpec(
        "Segmentation", False, False, "raw_color"
    ),
}

_DEFAULT_CAPTURE_SOURCE = {
    "rgb": "scene",
    "scene": "scene",
    "color": "scene",
    "depth": "depth_perspective",
    "depth_perspective": "depth_perspective",
    "depth_vis": "depth_vis",
    "segmentation": "segmentation",
    "mask": "segmentation",
}


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
        strategy_type = (
            ComputerVisionControlStrategy
            if self._is_computer_vision()
            else MultirotorControlStrategy
        )
        self._control: AirSimControlStrategy = strategy_type(
            self.host,
            self.port,
            self.vehicle_name,
            lambda: self._airsim(),
        )

    def connect(self) -> PrimitiveResult:
        # Open the mode-specific AirSim client.
        self._client = self._control.connect()
        return PrimitiveResult(
            True,
            "AirSim ComputerVision client connected"
            if self._is_computer_vision()
            else "AirSim UAV connected",
        )

    def close(self) -> None:
        """Close the local RPC client without affecting the remote UE process."""

        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self._client = None

    def reset(self) -> PrimitiveResult:
        """Reset AirSim state and reconnect the configured control mode."""

        client = self._require_client()
        result = self._control.reset(client)
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

    def get_frames(
        self,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> PrimitiveResult:
        """Capture canonical frames using caller-selected generic sources.

        ``capture_sources`` is supplied by the run-level ObservationProfile.
        No map ID, scene ID, or benchmark name participates in routing.
        """

        airsim = self._airsim()
        requests = []
        resolved: list[tuple[str, str, _AirSimCaptureSpec]] = []
        for modality in modalities:
            source = self._capture_source(modality, capture_sources)
            spec = self._capture_spec(source)
            image_type = getattr(airsim.ImageType, spec.image_type_attribute)
            requests.append(
                airsim.ImageRequest(
                    self.camera_name,
                    image_type,
                    spec.pixels_as_float,
                    spec.compress,
                )
            )
            resolved.append((modality, source, spec))
        responses = self._require_client().simGetImages(
            requests, vehicle_name=self.vehicle_name
        )
        if len(responses) != len(resolved):
            raise RuntimeError(
                "AirSim returned an unexpected response count: "
                f"requested={len(resolved)}, received={len(responses)}"
            )
        frames = {
            modality: self._frame_metadata(modality, source, spec, response)
            for (modality, source, spec), response in zip(resolved, responses)
        }
        return PrimitiveResult(True, "AirSim UAV frames", {"frames": frames})

    def _frame_metadata(
        self,
        modality: str,
        capture_source: str,
        spec: _AirSimCaptureSpec,
        response: Any,
    ) -> dict[str, Any]:
        """Build transport metadata and optional correctly encoded artifacts."""

        metadata = {
            "width": response.width,
            "height": response.height,
            "capture_source": capture_source,
            "image_type": spec.image_type_attribute,
            "pixels_as_float": spec.pixels_as_float,
            "compressed": spec.compress,
            "bytes": len(response.image_data_uint8 or b""),
            "float_count": len(response.image_data_float or []),
        }
        if self.save_artifacts and self.artifact_dir is not None:
            artifact_path = self._save_frame_artifact(
                modality, response, capture_source
            )
            if artifact_path is not None:
                metadata["artifact_path"] = str(artifact_path)
                if capture_source == "depth_vis":
                    metadata["preview_path"] = str(
                        self._depth_preview_path(artifact_path)
                    )
        return metadata

    def _save_frame_artifact(
        self,
        modality: str,
        response: Any,
        capture_source: str | None = None,
    ) -> Path | None:
        """Persist one response according to its capture source.

        AirSim raw RGB bytes are pixels, not an encoded PNG.  Conversely,
        DepthVis returns encoded image bytes.  Treating both payloads as the
        same format produced valid-looking files containing unusable images.
        """

        import numpy as np
        from PIL import Image

        source = capture_source or self._capture_source(modality, None)
        spec = self._capture_spec(source)
        width = int(response.width)
        height = int(response.height)

        if spec.payload_kind == "float_depth":
            values = np.asarray(response.image_data_float or [], dtype=np.float32)
            self._validate_pixel_count(source, values.size, width * height)
            values = values.reshape(height, width)
            path = self._next_artifact_path("depth", "npy")
            path.parent.mkdir(parents=True, exist_ok=True)
            buffer = BytesIO()
            np.save(buffer, values)
            path.write_bytes(buffer.getvalue())
            return path

        if spec.payload_kind == "depth_vis":
            float_values = np.asarray(
                response.image_data_float or [], dtype=np.float32
            )
            if float_values.size:
                self._validate_pixel_count(
                    source, float_values.size, width * height
                )
                normalized = float_values.reshape(height, width)
                path = self._next_artifact_path("depth", "npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                buffer = BytesIO()
                np.save(buffer, normalized)
                path.write_bytes(buffer.getvalue())
                preview_path = self._depth_preview_path(path)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                preview = np.clip(normalized, 0.0, 1.0)
                Image.fromarray(
                    np.round(preview * 255.0).astype(np.uint8), mode="L"
                ).save(preview_path, format="PNG")
                return path

            payload = bytes(response.image_data_uint8 or b"")
            if not payload:
                return None
            with Image.open(BytesIO(payload)) as encoded:
                color = np.asarray(encoded.convert("RGB"), dtype=np.uint8)
            if color.shape[:2] != (height, width):
                raise ValueError(
                    f"{source} dimensions disagree with payload: "
                    f"metadata={width}x{height}, decoded={color.shape[1]}x{color.shape[0]}"
                )
            # AirSim DepthVis encodes the normalized visualization in the
            # green channel.  Preserve the numeric array used by downstream
            # depth consumers and also emit a human-viewable preview.
            green = color[:, :, 1]
            normalized = green.astype(np.float32) / 255.0
            path = self._next_artifact_path("depth", "npy")
            path.parent.mkdir(parents=True, exist_ok=True)
            buffer = BytesIO()
            np.save(buffer, normalized)
            path.write_bytes(buffer.getvalue())
            preview_path = self._depth_preview_path(path)
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(green, mode="L").save(preview_path, format="PNG")
            return path

        payload = bytes(response.image_data_uint8 or b"")
        if not payload:
            return None

        if spec.payload_kind == "raw_color":
            pixel_count = width * height
            if pixel_count <= 0:
                raise ValueError(f"{source} returned invalid dimensions {width}x{height}")
            channels, remainder = divmod(len(payload), pixel_count)
            if remainder or channels not in {3, 4}:
                raise ValueError(
                    f"{source} raw payload has {len(payload)} bytes for "
                    f"{width}x{height}; expected 3 or 4 channels"
                )
            array = np.frombuffer(payload, dtype=np.uint8).reshape(
                height, width, channels
            )
            mode = "RGBA" if channels == 4 else "RGB"
            folder = "rgb" if modality.lower() in {"rgb", "scene", "color"} else modality.lower()
            path = self._next_artifact_path(folder, "png")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array, mode=mode).save(path, format="PNG")
            return path

        raise ValueError(f"Unsupported AirSim payload kind: {spec.payload_kind}")

    def _depth_preview_path(self, depth_path: Path) -> Path:
        if self.artifact_dir is None:
            raise RuntimeError("artifact_dir is required when saving artifacts")
        return self.artifact_dir / "depth_preview" / f"{depth_path.stem}.png"

    @staticmethod
    def _validate_pixel_count(
        capture_source: str, actual: int, expected: int
    ) -> None:
        if expected <= 0 or actual != expected:
            raise ValueError(
                f"{capture_source} returned {actual} values; expected {expected}"
            )

    def _next_artifact_path(self, folder: str, suffix: str) -> Path:
        # Adapter instances can be recreated inside one run, so never assume an
        # in-memory frame counter is globally unique for the run directory.
        if self.artifact_dir is None:
            raise RuntimeError("artifact_dir is required when saving artifacts")
        while True:
            self._frame_index += 1
            path = self.artifact_dir / folder / f"frame_{self._frame_index:06d}.{suffix}"
            if not path.exists():
                return path

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
        # Import AirSim lazily so dry-run tests do not require the package.
        import airsim

        return airsim

    @staticmethod
    def _capture_source(
        modality: str,
        capture_sources: dict[str, str] | None,
    ) -> str:
        key = modality.strip().lower()
        if capture_sources and key in capture_sources:
            return str(capture_sources[key]).strip().lower()
        try:
            return _DEFAULT_CAPTURE_SOURCE[key]
        except KeyError as exc:
            raise ValueError(
                f"AirSim has no default capture source for modality {modality!r}"
            ) from exc

    @staticmethod
    def _capture_spec(capture_source: str) -> _AirSimCaptureSpec:
        try:
            return _AIRSIM_CAPTURE_SPECS[capture_source]
        except KeyError as exc:
            available = ", ".join(sorted(_AIRSIM_CAPTURE_SPECS))
            raise ValueError(
                f"Unsupported AirSim capture source {capture_source!r}; "
                f"available: {available}"
            ) from exc

    def _is_computer_vision(self) -> bool:
        # Detect AirSim ComputerVision mode, which has observations but no UAV flight APIs.
        return self.mode.lower() in {"computervision", "computer_vision", "cv"}
