"""Lock generic UnrealCV camera/actor control and capture contracts.

These tests use a protocol-level fake rather than launching UE. Production
UAV-Flow benchmark runs validate the same behavior against DowntownWest.
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
        self.actor_rotation = [1.0, 10.0, 2.0]  # roll, yaw, pitch
        self.actor_scale = [0.2, 0.2, 0.2]
        self.objects = {
            "BP_Drone",
            "BP_Character",
            "BP_Animal",
            "BP_Car",
        }
        self.hidden_objects: set[str] = set()
        self.camera_location = [10.0, 20.0, 30.0]
        self.camera_rotation = [3.0, 4.0, 5.0]  # pitch, yaw, roll
        self.commands: list[str] = []
        self.batches: list[list[str]] = []
        self.requests: list[tuple[str, int]] = []

    def request_batch(self, commands: list[str]):
        self.batches.append(list(commands))
        return [self.request(command) for command in commands]

    def request(self, command, timeout=5):
        if isinstance(command, list):
            return self.request_batch(command)
        command = str(command)
        self.commands.append(command)
        self.requests.append((command, int(timeout)))
        if command.endswith("/location") and "/object/" in command:
            return " ".join(str(value) for value in self.actor_location).encode()
        if command.endswith("/rotation") and "/object/" in command:
            return " ".join(str(value) for value in self.actor_rotation).encode()
        if command.endswith("/location") and "/camera/" in command:
            return " ".join(str(value) for value in self.camera_location).encode()
        if command.endswith("/rotation") and "/camera/" in command:
            return " ".join(str(value) for value in self.camera_rotation).encode()
        if command == "vget /objects":
            return " ".join(sorted(self.objects)).encode()
        if command.startswith("vset /object/") and command.endswith("/hide"):
            self.hidden_objects.add(command.split("/")[2])
            return b"ok"
        if command.startswith("vset /object/") and "/location " in command:
            self.actor_location = [float(value) for value in command.split()[-3:]]
            return b"ok"
        if command.startswith("vset /object/") and "/rotation " in command:
            pitch, yaw, roll = [float(value) for value in command.split()[-3:]]
            self.actor_rotation = [roll, yaw, pitch]
            return b"ok"
        if command.startswith("vset /object/") and "/scale " in command:
            self.actor_scale = [
                float(value) for value in command.split()[-3:]
            ]
            return b"ok"
        if command.startswith("vbp BP_Drone set_rotation "):
            # DowntownWest applies another -180 degree mesh-facing offset
            # inside the Blueprint. A caller-provided native -180 offset thus
            # resolves to the physical camera/world yaw.
            result = float(command.split()[-1]) - 180.0
            self.actor_rotation[1] = (result + 180.0) % 360.0 - 180.0
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

    def request(self, command, timeout=5):
        if isinstance(command, str) and command.startswith(
            "vbp BP_Drone set_rotation "
        ):
            self.commands.append(command)
            self.requests.append((command, int(timeout)))
            return b"ok"
        return super().request(command, timeout)


class HideFailureClient(FakeUnrealCvClient):
    """Return an explicit UnrealCV error for one declared startup hide."""

    def request(self, command, timeout=5):
        if command == "vset /object/BP_Character/hide":
            self.commands.append(command)
            self.requests.append((command, int(timeout)))
            return b"error hide failed"
        return super().request(command, timeout)


class StuckStandardCameraClient(FakeUnrealCvClient):
    """Acknowledge camera pose writes while leaving its pose unchanged."""

    def request(self, command, timeout=5):
        if isinstance(command, str) and command.startswith("vset /camera/"):
            self.commands.append(command)
            self.requests.append((command, int(timeout)))
            return b"ok"
        return super().request(command, timeout)


class WrappedYawCameraClient(FakeUnrealCvClient):
    """Expose an equivalent camera yaw on the opposite wrap boundary."""

    def request(self, command, timeout=5):
        result = super().request(command, timeout)
        if (
            isinstance(command, str)
            and command.startswith("vset /camera/")
            and "/rotation " in command
        ):
            self.camera_rotation[1] -= 360.0
        return result


class MountedCameraClient(FakeUnrealCvClient):
    """Apply the packaged map's fixed body-frame camera mount offset."""

    def request(self, command, timeout=5):
        result = super().request(command, timeout)
        if (
            isinstance(command, str)
            and command.startswith("vset /camera/")
            and "/location " in command
        ):
            yaw_rad = np.deg2rad(self.actor_rotation[1])
            forward_m, right_m, up_m = (0.2, 0.0, 0.6)
            self.camera_location = [
                self.actor_location[0]
                + 100.0
                * (
                    forward_m * np.cos(yaw_rad)
                    - right_m * np.sin(yaw_rad)
                ),
                self.actor_location[1]
                + 100.0
                * (
                    forward_m * np.sin(yaw_rad)
                    + right_m * np.cos(yaw_rad)
                ),
                self.actor_location[2] + 100.0 * up_m,
            ]
        return result


class StuckActorLocationClient(FakeUnrealCvClient):
    """Acknowledge Actor location writes without applying them."""

    def request(self, command, timeout=5):
        if (
            isinstance(command, str)
            and command.startswith("vset /object/")
            and "/location " in command
        ):
            self.commands.append(command)
            return "ok"
        return super().request(command, timeout)


def _actor_config(**overrides) -> ActorControlConfig:
    values = {
        "name": "BP_Drone",
        "camera_id": "0",
        "startup_hide_objects": (),
        "scale": None,
        "units_per_meter": 100.0,
        "yaw_offset_deg": -180.0,
        "rotation_method": "blueprint_yaw",
        "camera_follow": "blueprint_pose",
        "camera_validation_enabled": False,
        "camera_position_mode": "expected_body_offset",
        "camera_expected_body_offset_m": (0.0, 0.0, 0.0),
        "camera_position_tolerance_m": 0.05,
        "camera_max_distance_m": 1.5,
        "camera_yaw_tolerance_deg": 1.0,
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
            "scale": [0.1, 0.1, 0.1],
            "units_per_meter": 100.0,
            "yaw_offset_deg": -180.0,
            "rotation_method": "blueprint_yaw",
            "camera_follow": "blueprint_pose",
            "camera_validation": {
                "enabled": False,
                "position_mode": "expected_body_offset",
                "expected_body_offset_m": [0.0, 0.0, 0.0],
                "position_tolerance_m": 0.05,
                "max_distance_m": 1.5,
                "yaw_tolerance_deg": 1.0,
            },
            "disable_physics": True,
            "yaw_settle_timeout_sec": 5.0,
            "yaw_tolerance_deg": 1.0,
            "yaw_poll_interval_sec": 0.05,
        }
        unrealcv["startup"] = {
            "hide_objects": [
                "BP_Character",
                "BP_Animal",
                "BP_Car",
                "BP_Drone",
            ]
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
    assert "vbp BP_Drone set_cam 100.0 200.0 300.0 1.0 2.0 10.0" in client.commands
    assert "vbp BP_Drone set_viewport" in client.commands
    assert not any(command.startswith("vbp 0 set_cam ") for command in client.commands)

    control.set_pose([1.5, 2.5, 3.5, 2.0, 20.0, 1.0])

    assert client.actor_location == pytest.approx([150.0, 250.0, 350.0])
    assert "vbp BP_Drone set_rotation -160.0" in client.commands
    assert client.actor_rotation[1] == pytest.approx(20.0)
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
            camera_validation_enabled=True,
            disable_physics=False,
        ),
    )

    control.sync_camera()

    assert client.camera_location == pytest.approx(client.actor_location)
    assert client.camera_rotation == pytest.approx([2.0, 10.0, 1.0])
    assert not any(command.startswith("vbp 0 set_cam") for command in client.commands)


def test_standard_camera_validation_rejects_silent_noop() -> None:
    control = ActorPoseControl(
        StuckStandardCameraClient(),
        _actor_config(
            camera_follow="standard_pose",
            camera_validation_enabled=True,
            disable_physics=False,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="camera follow validation failed",
    ):
        control.set_pose([1.0, 2.0, 3.0, 2.0, 10.0, 1.0])
        control.sync_camera()


def test_max_distance_camera_validation_accepts_collision_offset() -> None:
    client = MountedCameraClient()
    control = ActorPoseControl(
        client,
        _actor_config(
            camera_follow="standard_pose",
            camera_validation_enabled=True,
            camera_position_mode="max_distance",
            camera_max_distance_m=1.5,
            disable_physics=False,
        ),
    )

    control.set_pose([1.0, 2.0, 3.0, 2.0, 10.0, 1.0])
    control.sync_camera()


def test_actor_pose_write_defers_camera_validation_until_capture() -> None:
    control = ActorPoseControl(
        StuckStandardCameraClient(),
        _actor_config(
            camera_follow="standard_pose",
            camera_validation_enabled=True,
            disable_physics=False,
        ),
    )

    # Scene preflight takeoff uses the same pose path before episode placement.
    control.set_pose([1.0, 2.0, 3.0, 2.0, 10.0, 1.0])


def test_actor_pose_validation_rejects_silent_location_noop() -> None:
    control = ActorPoseControl(
        StuckActorLocationClient(),
        _actor_config(
            camera_follow="standard_pose",
            camera_validation_enabled=True,
            disable_physics=False,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="actor pose validation failed",
    ):
        control.set_pose([1.5, 2.5, 3.5, 2.0, 10.0, 1.0])


def test_actor_pose_validation_accepts_wrapped_yaw() -> None:
    client = FakeUnrealCvClient()
    client.actor_rotation[1] = 179.0
    control = ActorPoseControl(
        client,
        _actor_config(
            camera_follow="standard_pose",
            camera_validation_enabled=True,
            disable_physics=False,
        ),
    )

    control.set_pose([1.0, 2.0, 3.0, 2.0, -181.0, 1.0])


def test_blueprint_camera_validation_rejects_silent_noop() -> None:
    control = ActorPoseControl(
        FakeUnrealCvClient(),
        _actor_config(camera_validation_enabled=True),
    )

    with pytest.raises(
        RuntimeError,
        match="strategy=blueprint_pose",
    ):
        control.set_pose([1.0, 2.0, 3.0, 2.0, 10.0, 1.0])
        control.sync_camera()


def test_camera_validation_accepts_equivalent_wrapped_yaw() -> None:
    client = WrappedYawCameraClient()
    client.actor_rotation[1] = 179.0
    control = ActorPoseControl(
        client,
        _actor_config(
            camera_follow="standard_pose",
            camera_validation_enabled=True,
            disable_physics=False,
        ),
    )

    control.set_pose([1.0, 2.0, 3.0, 2.0, 179.0, 1.0])
    control.sync_camera()

    assert client.camera_rotation[1] == pytest.approx(-181.0)


def test_camera_validation_accepts_declared_body_mount_offset() -> None:
    control = ActorPoseControl(
        MountedCameraClient(),
        _actor_config(
            camera_follow="standard_pose",
            camera_validation_enabled=True,
            camera_expected_body_offset_m=(0.2, 0.0, 0.6),
            disable_physics=False,
        ),
    )

    control.set_pose([1.0, 2.0, 3.0, 2.0, 10.0, 1.0])
    control.sync_camera()


def test_actor_prepare_normalizes_scene_before_camera_sync() -> None:
    client = FakeUnrealCvClient()
    control = ActorPoseControl(
        client,
        _actor_config(
            startup_hide_objects=(
                "BP_Character",
                "BP_Animal",
                "BP_Car",
                "BP_Drone",
            ),
            scale=(0.1, 0.1, 0.1),
            camera_follow="standard_pose",
        ),
    )

    control.prepare()

    scale_command = "vset /object/BP_Drone/scale 0.1 0.1 0.1"
    hide_commands = [
        "vset /object/BP_Character/hide",
        "vset /object/BP_Animal/hide",
        "vset /object/BP_Car/hide",
        "vset /object/BP_Drone/hide",
    ]
    assert client.hidden_objects == {
        "BP_Character",
        "BP_Animal",
        "BP_Car",
        "BP_Drone",
    }
    assert all((command, 5) in client.requests for command in hide_commands)
    assert client.actor_scale == pytest.approx([0.1, 0.1, 0.1])
    assert client.commands.count(scale_command) == 1
    assert max(client.commands.index(command) for command in hide_commands) < (
        client.commands.index(scale_command)
    )
    assert client.commands.index(scale_command) < next(
        index
        for index, command in enumerate(client.commands)
        if command.startswith("vset /camera/0/location ")
    )


def test_hidden_controlled_actor_remains_pose_controllable() -> None:
    """Hiding the UAV mesh must not turn Actor control into camera-only motion."""

    client = FakeUnrealCvClient()
    control = ActorPoseControl(
        client,
        _actor_config(
            startup_hide_objects=("BP_Drone",),
            camera_follow="standard_pose",
            camera_validation_enabled=True,
            disable_physics=False,
        ),
    )

    control.prepare()
    control.set_pose([1.5, 2.5, 3.5, 2.0, 20.0, 1.0])
    control.sync_camera()

    assert client.hidden_objects == {"BP_Drone"}
    assert control.get_pose() == pytest.approx(
        [1.5, 2.5, 3.5, 2.0, 20.0, 1.0]
    )
    assert client.actor_location == pytest.approx([150.0, 250.0, 350.0])
    assert client.camera_location == pytest.approx(client.actor_location)


def test_actor_prepare_ignores_missing_declared_hide_object() -> None:
    client = FakeUnrealCvClient()
    control = ActorPoseControl(
        client,
        _actor_config(startup_hide_objects=("AlreadyMissing",)),
    )

    control.prepare()

    assert not any(
        command.endswith("/hide") for command in client.commands
    )


def test_actor_prepare_reports_startup_hide_failure() -> None:
    control = ActorPoseControl(
        HideFailureClient(),
        _actor_config(startup_hide_objects=("BP_Character",)),
    )

    with pytest.raises(RuntimeError, match="hide failed"):
        control.prepare()


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
    assert actor_package.unrealcv["actor"]["scale"] == [0.1, 0.1, 0.1]
    assert actor_package.unrealcv["startup"]["hide_objects"] == [
        "BP_Character",
        "BP_Animal",
        "BP_Car",
        "BP_Drone",
    ]
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
                "actor": {"name": "drone", "camera_follow": "magic"},
            },
            "camera_follow",
        ),
        (
            {
                "control_target": "actor",
                "actor": {"name": "drone", "scale": [0.1, 0.1]},
            },
            "scale",
        ),
        (
            {
                "control_target": "actor",
                "actor": {"name": "drone", "scale": [0.1, 0.0, 0.1]},
            },
            "scale",
        ),
        (
            {
                "control_target": "actor",
                "actor": {"name": "drone"},
                "startup": {"hide_objects": ["person", "person"]},
            },
            "unique",
        ),
        (
            {
                "control_target": "actor",
                "actor": {
                    "name": "drone",
                    "camera_validation": {"enabled": "yes"},
                },
            },
            "camera_validation.enabled",
        ),
        (
            {
                "control_target": "actor",
                "actor": {
                    "name": "drone",
                    "camera_validation": {
                        "position_tolerance_m": 0
                    },
                },
            },
            "position_tolerance_m",
        ),
        (
            {
                "control_target": "actor",
                "actor": {
                    "name": "drone",
                    "camera_validation": {
                        "expected_body_offset_m": [0.0, 0.0]
                    },
                },
            },
            "expected_body_offset_m",
        ),
        (
            {
                "control_target": "actor",
                "actor": {
                    "name": "drone",
                    "camera_validation": {
                        "position_mode": "anywhere"
                    },
                },
            },
            "position_mode",
        ),
        (
            {
                "control_target": "actor",
                "actor": {
                    "name": "drone",
                    "camera_validation": {"max_distance_m": 0},
                },
            },
            "max_distance_m",
        ),
        (
            {
                "control_target": "actor",
                "actor": {"name": "drone"},
                "startup": {"hide_objects": "person"},
            },
            "hide_objects",
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
