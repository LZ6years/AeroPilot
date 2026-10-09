"""Capture human preview using a separate, thread-owned AirSim RPC client."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import cv2
import numpy as np

from src.adapters.airsim.client import close_airsim_client
from .base import PreviewConfig, PreviewFrame


class AirSimPreviewSource:
    """Never shares a client, camera, frame counter or memory writer with Agent."""

    def __init__(self, host: str, port: int, vehicle_name: str, config: PreviewConfig):
        self.host, self.port, self.vehicle_name, self.config = host, port, vehicle_name, config
        self._client = None
        self._frame_id = 0

    def get_frame(self) -> PreviewFrame:
        import airsim

        if self._client is None:
            self._client = airsim.VehicleClient(
                ip=self.host, port=self.port, timeout_value=self.config.timeout_sec
            )
            if not self._client.ping():
                raise ConnectionError("AirSim preview RPC ping failed")
        started = time.monotonic()
        replies = self._client.simGetImages([
            airsim.ImageRequest(self.config.camera_name, airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.vehicle_name)
        if len(replies) != 1:
            raise RuntimeError("AirSim preview did not return one frame")
        reply = replies[0]
        width, height = int(reply.width), int(reply.height)
        if (width, height) != (self.config.width, self.config.height):
            raise RuntimeError(
                f"UI camera {self.config.camera_name!r} returned {width}x{height}; "
                f"expected {self.config.width}x{self.config.height}. Configure the UI camera before attaching."
            )
        raw = bytes(reply.image_data_uint8 or b"")
        channels, remainder = divmod(len(raw), width * height)
        if remainder or channels not in {3, 4}:
            raise ValueError("Invalid AirSim preview pixel payload")
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)
        # The existing RGB capture contract treats AirSim raw pixels as RGB(A).
        color = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_RGBA2BGR)
        ok, jpeg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        if not ok:
            raise RuntimeError("Preview JPEG encoding failed")
        self._frame_id += 1
        return PreviewFrame(
            self._frame_id, datetime.now(timezone.utc).isoformat(),
            time.monotonic(), (time.monotonic() - started) * 1000,
            width, height, jpeg.tobytes(),
        )

    def close(self) -> None:
        close_airsim_client(self._client)
        self._client = None
