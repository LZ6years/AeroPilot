"""Lock generic UnrealCV camera/actor control and capture contracts.

These tests use a protocol-level fake rather than launching UE.  The separate
manual probe validates the same behavior against DowntownWest.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.env_client.schemas import MapPackage
from src.sim_server.backends.unrealcv_backend import UnrealCvBackend
from src.sim_server.backends.unrealcv_capture import UnrealCvCapture
from src.sim_server.backends.unrealcv_control import (
    ActorControlConfig,
    ActorPoseControl,
    CameraPoseControl,
)


class FakeUnrealCvClient:
    """Emulate the small UnrealCV command surface owned by the new modules."""

    def __init__(self) -> None:
        self.actor_location = [100.0, 200.0, 300.0]
        self.actor_rotation = [1.0, -170.0, 2.0]  # roll, yaw, pitch
        self.camera_location = [10.0, 20.0, 30.0]
        self.camera_rotation = [3.0, 4.0, 5.0]  # pitch, yaw, roll
        self.commands: list[str] = []
        self.batches: list[list[str]] = []

    def request_batch(self, commands: list[str]):
        self.batches.append(list(commands))
        return [self.request(command) for command in commands]

    def request(self, command):
        if isinstance(command, list):
            return self.request_batch(command)
        command = str(command)
        self.commands.append(command)
        if command.endswith("/location") and "/object/" in command:
            return " ".join(str(value) for value in self.actor_location).encode()
        if command.endswith("/rotation") and "/object/" in command:
            return " ".join(str(value) for value in self.actor_rotation).encode()
        if command.endswith("/location") and "/camera/" in command:
            return " ".join(str(value) for value in self.camera_location).encode()
        if command.endswith("/rotation") and "/camera/" in command:
            return " ".join(str(value) for value in self.camera_rotation).encode()
        if command.startswith("vset /object/") and "/location " in command:
            self.actor_location = [float(value) for value in command.split()[-3:]]
            return b"ok"
        if command.startswith("vset /object/") and "/rotation " in command:
            pitch, yaw, roll = [float(value) for value in command.split()[-3:]]
            self.actor_rotation = [roll, yaw, pitch]
            return b"ok"
        if command.startswith("vbp BP_Drone set_rotation "):
            # DowntownWest's Blueprint accepts canonical yaw and applies the
            # native -180 degree actor mesh offset internally.
            self.actor_rotation[1] = float(command.split()[-1]) - 180.0
            return b"ok"
        if command.startswith("vset /camera/") and "/location " in command:
            self.camera_location = [float(value) for value in command.split()[-3:]]
            return b"ok"
        if command.startswith("vset /camera/") and "/rotation " in command:
            self.camera_rotation = [float(value) for value in command.split()[-3:]]
            return b"ok"
        if command.startswith("vbp "):
            return b"ok"
        if command.endswith("/lit png"):
            return _png_bytes((20, 80, 160))
        if command.endswith("/object_mask png"):
            return _png_bytes((255, 0, 0))
        if command.endswith("/depth npy"):
            buffer = BytesIO()
            np.save(buffer, np.asarray([[100.0, 200.0], [300.0, 500.0]], dtype=np.float32))
            return buffer.getvalue()
        raise AssertionError(f"Unexpected fake UnrealCV command: {command}")


class StuckYawClient(FakeUnrealCvClient):
    """Accept a Blueprint yaw command without ever reaching its target."""

    def request(self, command):
        if isinstance(command, str) and command.startswith(
            "vbp BP_Drone set_rotation "
        ):
            self.commands.append(command)
            return b"ok"
        return super().request(command)


def _actor_config(**overrides) -> ActorControlConfig:
    values = {
        "name": "BP_Drone",
        "camera_id": "0",
        "units_per_meter": 100.0,
        "yaw_offset_deg": -180.0,
        "rotation_method": "blueprint_yaw",
        "camera_follow": "blueprint_pose",
        "disable_physics": True,
        "yaw_settle_timeout_sec": 0.1,
        "yaw_tolerance_deg": 0.01,
        "yaw_poll_interval_sec": 0.001,
    }
    values.update(overrides)
    return ActorControlConfig(**values)


def _package(control_target: str = "actor") -> MapPackage:
    unrealcv = {
        "host": "127.0.0.1",
        "port": 9068,
        "camera_id": 0,
        "control_target": control_target,
    }
    if control_target == "actor":
        unrealcv["actor"] = {
            "name": "BP_Drone",
            "units_per_meter": 100.0,
            "yaw_offset_deg": -180.0,
            "rotation_method": "blueprint_yaw",
            "camera_follow": "blueprint_pose",
            "disable_physics": True,
            "yaw_settle_timeout_sec": 5.0,
            "yaw_tolerance_deg": 1.0,
            "yaw_poll_interval_sec": 0.05,
        }
    return MapPackage.from_dict(
        {
            "map_id": f"test_{control_target}",
            "provider": "packaged_unreal",
            "scene_id": "test",
            "root": "C:/test",
            "connection": {"mode": "local"},
            "launch": {"windows": "test.exe"},
            "control": {"protocol": "unrealcv"},
            "unrealcv": unrealcv,
        }
    )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_actor_pose_converts_units_yaw_and_syncs_camera() -> None:
    client = FakeUnrealCvClient()
    control = ActorPoseControl(client, _actor_config())

    control.prepare()
    pose = control.get_pose()

    assert pose == pytest.approx([1.0, 2.0, 3.0, 2.0, 10.0, 1.0])
    assert "vbp BP_Drone set_phy 0" in client.commands
    assert any(command.startswith("vbp 0 set_cam ") for command in client.commands)

    control.set_pose([1.5, 2.5, 3.5, 2.0, 20.0, 1.0])

    assert client.actor_location == pytest.approx([150.0, 250.0, 350.0])
    assert client.actor_rotation[1] == pytest.approx(-160.0)
    assert control.get_pose() == pytest.approx([1.5, 2.5, 3.5, 2.0, 20.0, 1.0])


def test_blueprint_yaw_rejects_pitch_or_roll_changes() -> None:
    client = FakeUnrealCvClient()
    control = ActorPoseControl(client, _actor_config())

    with pytest.raises(ValueError, match="cannot change pitch_deg or roll_deg"):
        control.set_pose([1.0, 2.0, 3.0, 99.0, 10.0, 1.0])


def test_blueprint_yaw_reports_settle_timeout() -> None:
    control = ActorPoseControl(
        StuckYawClient(),
        _actor_config(
            yaw_settle_timeout_sec=0.002,
            yaw_poll_interval_sec=0.001,
        ),
    )

    with pytest.raises(RuntimeError, match="yaw did not settle"):
        control.set_pose([1.0, 2.0, 3.0, 2.0, 20.0, 1.0])


def test_standard_actor_rotation_sets_all_euler_components() -> None:
    client = FakeUnrealCvClient()
    control = ActorPoseControl(
        client,
        _actor_config(
            rotation_method="standard_pose",
            camera_follow="none",
            disable_physics=False,
        ),
    )

    control.set_pose([1.0, 2.0, 3.0, 8.0, 30.0, 7.0])

    assert client.actor_rotation == pytest.approx([7.0, -150.0, 8.0])


def test_actor_can_sync_camera_with_standard_unrealcv_pose() -> None:
    client = FakeUnrealCvClient()
    control = ActorPoseControl(
        client,
        _actor_config(
            camera_follow="standard_pose",
            disable_physics=False,
        ),
    )

    control.sync_camera()

    assert client.camera_location == pytest.approx(client.actor_location)
    assert client.camera_rotation == pytest.approx([2.0, -170.0, 1.0])
    assert not any(command.startswith("vbp 0 set_cam") for command in client.commands)


def test_camera_pose_strategy_preserves_legacy_native_coordinates() -> None:
    client = FakeUnrealCvClient()
    control = CameraPoseControl(client, "0")

    assert control.get_pose() == pytest.approx([10.0, 20.0, 30.0, 3.0, 4.0, 5.0])
    control.set_pose([11.0, 22.0, 33.0, 6.0, 7.0, 8.0])
    assert client.camera_location == pytest.approx([11.0, 22.0, 33.0])
    assert client.camera_rotation == pytest.approx([6.0, 7.0, 8.0])


def test_capture_batches_modalities_converts_depth_and_saves_artifacts(
    tmp_path: Path,
) -> None:
    client = FakeUnrealCvClient()
    sync_calls: list[bool] = []
    capture = UnrealCvCapture(
        client,
        camera_id="0",
        units_per_meter=100.0,
        artifact_dir=tmp_path,
        save_artifacts=True,
        sync_camera=lambda: sync_calls.append(True),
    )

    frames = capture.capture(
        ["rgb", "depth", "segmentation"],
        {
            "rgb": "scene",
            "depth": "depth_perspective",
            "segmentation": "segmentation",
        },
    )

    assert sync_calls == [True]
    assert len(client.batches) == 1
    assert client.batches[0] == [
        "vget /camera/0/lit png",
        "vget /camera/0/depth npy",
        "vget /camera/0/object_mask png",
    ]
    assert frames["rgb"]["capture_group"] == frames["depth"]["capture_group"]
    assert Path(frames["rgb"]["artifact_path"]).is_file()
    assert Path(frames["segmentation"]["artifact_path"]).is_file()
    depth_path = Path(frames["depth"]["artifact_path"])
    assert depth_path.is_file()
    assert Path(frames["depth"]["preview_path"]).is_file()
    assert np.load(depth_path) == pytest.approx(
        np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    )


def test_capture_rejects_unknown_modality_and_source(tmp_path: Path) -> None:
    capture = UnrealCvCapture(
        FakeUnrealCvClient(),
        camera_id="0",
        units_per_meter=100.0,
        artifact_dir=tmp_path,
        save_artifacts=False,
        sync_camera=lambda: None,
    )

    with pytest.raises(ValueError, match="observation modality"):
        capture.capture(["normal"], None)
    with pytest.raises(ValueError, match="capture source"):
        capture.capture(["rgb"], {"rgb": "benchmark_special"})


def test_map_package_validates_actor_control_and_capabilities() -> None:
    actor_package = _package("actor")
    camera_package = _package("camera")
    backend = UnrealCvBackend()

    assert actor_package.unrealcv["actor"]["units_per_meter"] == 100.0
    assert backend.describe_capabilities(actor_package).control_mode == "actor_pose"
    assert backend.describe_capabilities(camera_package).control_mode == "camera_pose"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"control_target": "vehicle"}, "control_target"),
        ({"control_target": "actor"}, "unrealcv.actor"),
        (
            {"control_target": "actor", "actor": {"name": "drone", "units_per_meter": 0}},
            "units_per_meter",
        ),
        (
            {
                "control_target": "actor",
                "actor": {"name": "drone", "rotation_method": "magic"},
            },
            "rotation_method",
        ),
        (
            {
                "control_target": "actor",
                "actor": {"name": "drone", "yaw_settle_timeout_sec": 0},
            },
            "yaw_settle_timeout_sec",
        ),
    ],
)
def test_map_package_rejects_invalid_actor_configuration(
    change: dict,
    message: str,
) -> None:
    payload = _package("camera").to_dict()
    payload["unrealcv"].update(change)

    with pytest.raises(ValueError, match=message):
        MapPackage.from_dict(payload)
