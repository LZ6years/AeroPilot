"""Test AirSimUavAdapter behavior without launching a real AirSim scene."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.adapters.airsim_uav import AirSimUavAdapter


class FakeAirSim:
    # Provide the AirSim API surface used by pose and frame tests.
    class ImageType:
        Scene = 0
        DepthVis = 3
        DepthPerspective = 2
        Segmentation = 5

    class ImageRequest:
        def __init__(
            self,
            camera_name: str,
            image_type: int,
            pixels_as_float: bool,
            compress: bool,
        ) -> None:
            self.camera_name = camera_name
            self.image_type = image_type
            self.pixels_as_float = pixels_as_float
            self.compress = compress

    class Vector3r:
        def __init__(self, x_val: float, y_val: float, z_val: float) -> None:
            # Store vector coordinates with AirSim-compatible attribute names.
            self.x_val = x_val
            self.y_val = y_val
            self.z_val = z_val

    class Pose:
        def __init__(self, position: "FakeAirSim.Vector3r", orientation: tuple[float, float, float]) -> None:
            # Store position and orientation exactly like a lightweight AirSim Pose.
            self.position = position
            self.orientation = orientation

    @staticmethod
    def to_quaternion(pitch: float, roll: float, yaw: float) -> tuple[float, float, float]:
        # Return radians directly so the test can inspect degree-to-radian conversion.
        return (pitch, roll, yaw)


class FakeAirSimClient:
    def __init__(self) -> None:
        # Capture simSetVehiclePose calls for assertions.
        self.set_pose_calls = []
        self.move_body_calls = []
        self.image_requests = []
        self.image_responses = []

    def simSetVehiclePose(self, pose, ignore_collision: bool, vehicle_name: str) -> None:
        # Record the pose placement request that would have gone to AirSim.
        self.set_pose_calls.append(
            {
                "pose": pose,
                "ignore_collision": ignore_collision,
                "vehicle_name": vehicle_name,
            }
        )

    def moveByVelocityBodyFrameAsync(self, vx: float, vy: float, vz: float, duration: float, vehicle_name: str):
        # Record body-frame movement calls and return a joinable future.
        self.move_body_calls.append(
            {
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "duration": duration,
                "vehicle_name": vehicle_name,
            }
        )
        return self

    def join(self) -> None:
        # Match the AirSim async future API used by the adapter.
        return None

    def simGetImages(self, requests, vehicle_name: str):
        self.image_requests = list(requests)
        return list(self.image_responses)


class FakeImageResponse:
    def __init__(
        self,
        image_data_uint8: bytes = b"",
        image_data_float: list[float] | None = None,
        *,
        width: int = 2,
        height: int = 2,
    ) -> None:
        self.width = width
        self.height = height
        self.image_data_uint8 = image_data_uint8
        self.image_data_float = image_data_float or []


def test_set_pose_calls_airsim_sim_set_vehicle_pose(monkeypatch):
    # Verify set_pose translates FlyAgent x/y/z/yaw inputs into an AirSim pose call.
    adapter = AirSimUavAdapter(vehicle_name="Drone1")
    client = FakeAirSimClient()
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.set_pose(1.0, 2.0, -3.0, yaw_deg=90.0, ignore_collision=False)

    assert result.ok
    assert result.data["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": -3.0}
    assert result.data["pose"]["euler_deg"]["yaw"] == 90.0
    assert len(client.set_pose_calls) == 1
    call = client.set_pose_calls[0]
    assert call["vehicle_name"] == "Drone1"
    assert call["ignore_collision"] is False
    assert call["pose"].position.x_val == 1.0
    assert call["pose"].position.y_val == 2.0
    assert call["pose"].position.z_val == 3.0
    assert round(call["pose"].orientation[2], 6) == 1.570796


def test_ascend_and_descend_use_speed_and_duration():
    # Verify vertical primitives call AirSim body velocity with correct NED z signs.
    adapter = AirSimUavAdapter(vehicle_name="Drone1")
    client = FakeAirSimClient()
    adapter._client = client

    ascend = adapter.ascend(0.5, 4.0)
    descend = adapter.descend(0.25, 2.0)

    assert ascend.ok
    assert descend.ok
    assert client.move_body_calls[0] == {
        "vx": 0.0,
        "vy": 0.0,
        "vz": -0.5,
        "duration": 4.0,
        "vehicle_name": "Drone1",
    }
    assert client.move_body_calls[1] == {
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.25,
        "duration": 2.0,
        "vehicle_name": "Drone1",
    }


def test_frame_artifacts_do_not_overwrite_existing_files(tmp_path):
    # Separate-scene real runs recreate adapters, so artifact naming must scan
    # the run directory instead of trusting the in-memory frame counter.
    adapter = AirSimUavAdapter(artifact_dir=tmp_path / "artifacts", save_artifacts=True)
    existing_rgb = tmp_path / "artifacts" / "rgb" / "frame_000001.png"
    existing_depth = tmp_path / "artifacts" / "depth" / "frame_000001.npy"
    existing_rgb.parent.mkdir(parents=True)
    existing_depth.parent.mkdir(parents=True)
    existing_rgb.write_bytes(b"old-rgb")
    existing_depth.write_bytes(b"old-depth")

    rgb_payload = bytes(range(12))
    rgb_path = adapter._save_frame_artifact(
        "rgb", FakeImageResponse(image_data_uint8=rgb_payload)
    )
    adapter._frame_index = 0
    depth_path = adapter._save_frame_artifact("depth", FakeImageResponse(image_data_float=[1.0, 2.0, 3.0, 4.0]))

    assert rgb_path == tmp_path / "artifacts" / "rgb" / "frame_000002.png"
    assert depth_path == tmp_path / "artifacts" / "depth" / "frame_000002.npy"
    assert existing_rgb.read_bytes() == b"old-rgb"
    assert existing_depth.read_bytes() == b"old-depth"
    with Image.open(rgb_path) as image:
        assert image.size == (2, 2)
        assert image.mode == "RGB"
    assert np.load(depth_path).shape == (2, 2)


def test_observation_profile_routes_depth_vis_without_map_special_case(
    monkeypatch,
):
    adapter = AirSimUavAdapter(camera_name="front")
    client = FakeAirSimClient()
    client.image_responses = [
        FakeImageResponse(image_data_uint8=bytes(range(12))),
        _depth_vis_response(),
    ]
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.get_frames(
        ["rgb", "depth"],
        {"rgb": "scene", "depth": "depth_vis"},
    )

    assert result.ok
    rgb_request, depth_request = client.image_requests
    assert (
        rgb_request.image_type,
        rgb_request.pixels_as_float,
        rgb_request.compress,
    ) == (FakeAirSim.ImageType.Scene, False, False)
    assert (
        depth_request.image_type,
        depth_request.pixels_as_float,
        depth_request.compress,
    ) == (FakeAirSim.ImageType.DepthVis, False, True)
    assert result.data["frames"]["depth"]["capture_source"] == "depth_vis"


def test_artifact_setting_does_not_change_airsim_request(monkeypatch, tmp_path):
    requests = []
    for save_artifacts in (False, True):
        adapter = AirSimUavAdapter(
            artifact_dir=tmp_path / str(save_artifacts),
            save_artifacts=save_artifacts,
        )
        client = FakeAirSimClient()
        client.image_responses = [
            FakeImageResponse(image_data_uint8=bytes(range(12))),
            FakeImageResponse(image_data_float=[1.0, 2.0, 3.0, 4.0]),
        ]
        adapter._client = client
        monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

        adapter.get_frames(["rgb", "depth"])
        requests.append(
            [
                (item.image_type, item.pixels_as_float, item.compress)
                for item in client.image_requests
            ]
        )

    assert requests[0] == requests[1] == [
        (FakeAirSim.ImageType.Scene, False, False),
        (FakeAirSim.ImageType.DepthPerspective, True, False),
    ]


def test_depth_vis_saves_normalized_array_and_preview(tmp_path):
    adapter = AirSimUavAdapter(
        artifact_dir=tmp_path / "artifacts",
        save_artifacts=True,
    )

    path = adapter._save_frame_artifact(
        "depth",
        _depth_vis_response(),
        "depth_vis",
    )

    values = np.load(path)
    assert values.shape == (2, 2)
    assert values.min() == 0.0
    assert values.max() == 1.0
    assert adapter._depth_preview_path(path).is_file()


def test_depth_vis_accepts_float_payload_from_compatible_servers(tmp_path):
    adapter = AirSimUavAdapter(
        artifact_dir=tmp_path / "artifacts",
        save_artifacts=True,
    )

    path = adapter._save_frame_artifact(
        "depth",
        FakeImageResponse(image_data_float=[0.0, 0.25, 0.5, 1.0]),
        "depth_vis",
    )

    assert np.load(path).shape == (2, 2)
    assert adapter._depth_preview_path(path).is_file()


def test_unknown_capture_source_fails_before_rpc(monkeypatch):
    adapter = AirSimUavAdapter()
    adapter._client = FakeAirSimClient()
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    with pytest.raises(ValueError, match="Unsupported AirSim capture source"):
        adapter.get_frames(["depth"], {"depth": "benchmark_private_depth"})


def _depth_vis_response() -> FakeImageResponse:
    """Return a small compressed DepthVis-like PNG with varying green values."""

    pixels = np.array(
        [
            [[0, 0, 0], [0, 85, 0]],
            [[0, 170, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )
    buffer = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return FakeImageResponse(image_data_uint8=buffer.getvalue())
