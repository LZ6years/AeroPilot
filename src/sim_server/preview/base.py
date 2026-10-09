"""Preview contracts; these frames must never become model evidence."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PreviewConfig:
    """UI camera capture settings. The existing model camera is untouched."""

    camera_name: str = "ui_preview"
    width: int = 960
    height: int = 540
    fps: float = 20.0
    jpeg_quality: int = 80
    timeout_sec: float = 3.0

    def __post_init__(self) -> None:
        if not self.camera_name.strip():
            raise ValueError("preview camera_name must not be empty")
        if not (64 <= self.width <= 3840 and 64 <= self.height <= 2160):
            raise ValueError("preview dimensions exceed supported limits")
        if not (1 <= self.fps <= 60 and 30 <= self.jpeg_quality <= 95):
            raise ValueError("invalid preview fps or JPEG quality")
        if not (0 < self.timeout_sec <= 10):
            raise ValueError("invalid preview RPC timeout")


@dataclass(frozen=True)
class PreviewFrame:
    """One JPEG plus measured timestamps, without observation/checkpoint IDs."""

    frame_id: int
    captured_at: str
    captured_monotonic: float
    capture_ms: float
    width: int
    height: int
    jpeg: bytes


class PreviewSource(Protocol):
    def get_frame(self) -> PreviewFrame: ...
    def close(self) -> None: ...


def add_preview_camera(
    settings: dict[str, Any], vehicle_name: str, model_camera: str, config: PreviewConfig
) -> None:
    """Extend only a scene copy, preserving every original camera setting."""

    if config.camera_name == model_camera:
        raise ValueError("UI preview camera must differ from the model camera")
    vehicle = settings["Vehicles"][vehicle_name]
    cameras = vehicle.setdefault("Cameras", {})
    if config.camera_name in cameras:
        raise ValueError(f"UI preview camera already exists: {config.camera_name}")
    original = cameras.get(model_camera, settings.get("CameraDefaults", {}))
    camera = copy.deepcopy(original)
    # AirSim's custom camera spawner uses these values directly. Unlike an
    # existing pawn camera, unspecified NaN extrinsics can crash older UE builds.
    defaults = {"X": 0.5, "Y": 0.0, "Z": 0.0, "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0}
    for key, default in defaults.items():
        value = camera.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            camera[key] = default
    scene_capture = next((item for item in camera.get("CaptureSettings", []) if item.get("ImageType") == 0), {})
    camera["CaptureSettings"] = [{
        **scene_capture, "ImageType": 0, "Width": config.width,
        "Height": config.height, "FOV_Degrees": scene_capture.get("FOV_Degrees", 90),
    }]
    cameras[config.camera_name] = camera
