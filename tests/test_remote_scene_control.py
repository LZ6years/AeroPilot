"""Test unified SSH deployment without requiring a Linux host or AirSim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.env_client.schemas import (
    ActionResult,
    MapPackage,
    Observation,
    SceneCapabilities,
)
from src.env_client.simulator_client import SimulatorEnvClient
from src.sim_server.airsim_settings import AirSimSettingsRenderer
from src.sim_server.backends.base import BackendProbe
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.remote.control import RemoteSceneControl
from src.sim_server.server import SimulationServer


def make_template(path: Path) -> dict:
    settings = {
        "SettingsVersion": 1.2,
        "SimMode": "ComputerVision",
        "ApiServerPort": 9999,
        "Vehicles": {
            "Drone_1": {
                "VehicleType": "ComputerVision",
                "Cameras": {"front_0": {"CaptureSettings": []}},
            }
        },
    }
    path.write_text(json.dumps(settings), encoding="utf-8")
    return settings


def make_remote_package(template: Path) -> MapPackage:
    return MapPackage.from_dict(
        {
            "map_id": "remote_env_1",
            "provider": "packaged_unreal",
            "scene_id": "1",
            "root": None,
            "connection": {
                "mode": "remote",
                "data_transport": "ssh_tunnel",
                "machine_id": "linux_gpu_1",
            },
            "ssh": {
                "host": "192.168.1.20",
                "user": "uav",
                "key_path": str(template),
            },
            "launch": {
                "method": "linux_bash",
                "linux_executable": "/data/env_1/LinuxNoEditor/AirVLN.sh",
                "remote_run_root": ".flyagent/scenes",
            },
            "control": {"protocol": "airsim"},
            "airsim": {
                "remote_host": "127.0.0.1",
                "port": 41451,
                "mode": "ComputerVision",
                "settings_template_source": "local",
                "settings_template": str(template),
                "vehicle_name": "Drone_1",
                "camera_name": "front_0",
            },
        }
    )


def test_settings_renderer_preserves_template_and_runtime_fields(tmp_path):
    template = tmp_path / "template.json"
    original = make_template(template)
    package = make_remote_package(template)

    rendered = AirSimSettingsRenderer().render(
        package, tmp_path / "scene" / "settings.json", 41451
    )

    assert json.loads(template.read_text(encoding="utf-8")) == original
    output = json.loads(rendered.path.read_text(encoding="utf-8"))
    assert output["ApiServerPort"] == 41451
    assert output["Vehicles"] == original["Vehicles"]
    assert rendered.sha256 == hashlib.sha256(rendered.path.read_bytes()).hexdigest()


def test_settings_renderer_rejects_vehicle_mismatch(tmp_path):
    template = tmp_path / "template.json"
    settings = make_template(template)
    settings["Vehicles"] = {"OtherDrone": {}}
    template.write_text(json.dumps(settings), encoding="utf-8")

    with pytest.raises(ValueError, match="missing vehicle"):
        AirSimSettingsRenderer().render(
            make_remote_package(template),
            tmp_path / "output.json",
            41451,
        )


class FakeTunnel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def check_alive(self) -> None:
        assert not self.closed


class FakeRemoteControl:
    def __init__(self) -> None:
        self.starts: list[str] = []
        self.stops: list[str] = []
        self.tunnel = FakeTunnel()

    def preflight(self, package):
        return {"python": "3.10"}

    def open_tunnel(self, package, local_port, remote_host, remote_port, log_path=None):
        return self.tunnel

    def remote_port_status(
        self, package, scene_instance_id, host, port, local_scene_dir
    ):
        return {"ok": True, "status": "unreachable"}

    def start(self, package, scene_instance_id, settings_path, local_scene_dir):
        self.starts.append(scene_instance_id)
        assert settings_path and settings_path.is_file()
        return {
            "ok": True,
            "pid": 1234,
            "remote_settings_path": (
                f".flyagent/scenes/{scene_instance_id}/settings.json"
            ),
        }

    def stop(self, package, scene_instance_id):
        self.stops.append(scene_instance_id)
        return {"ok": True, "status": "stopped"}

    def probe_prepared_port(self, package, scene_instance_id, host, port):
        return {"status": "reachable" if self.starts else "unreachable"}


class ManagedRemoteBackend:
    """Model protocol readiness without importing or contacting AirSim."""

    def __init__(self, remote: FakeRemoteControl) -> None:
        self.remote = remote

    def probe(self, package, host, port):
        return BackendProbe(bool(self.remote.starts), "fake protocol state")

    def describe_capabilities(self, package):
        return SceneCapabilities(
            protocol="airsim",
            control_mode="ComputerVision",
            supported_actions=("connect", "get_pose", "get_frames"),
            observation_modalities=("rgb", "depth", "pose"),
            requires_takeoff=False,
            supports_soft_reset=True,
        )

    def start(self, package, scene):
        scene.status = "running"
        return scene

    def stop(self, scene):
        scene.status = "stopped"

    def reset(self, scene):
        return scene

    def execute_action(self, scene, agent_id, action, args):
        return ActionResult(True, action)

    def get_observation(self, scene, agent_id, modalities):
        return Observation(scene.scene_instance_id, agent_id, modalities, {})


def test_unified_client_starts_and_stops_owned_remote_scene(tmp_path):
    template = tmp_path / "template.json"
    make_template(template)
    remote = FakeRemoteControl()
    registry = BackendRegistry()
    registry.register(
        "airsim", lambda context: ManagedRemoteBackend(remote)
    )
    service = SimulationServer(
        [make_remote_package(template)],
        run_dir=tmp_path / "run",
        execution_mode=SimulationExecutionMode.MANAGED,
        remote_scene_control=remote,  # type: ignore[arg-type]
        backend_registry=registry,
    )
    client = SimulatorEnvClient(service)

    scene = client.start_scene("remote_env_1")

    assert scene.control_host == "127.0.0.1"
    assert scene.process_ownership == "remote_managed"
    assert scene.remote_process_id == 1234
    assert scene.settings_sha256
    assert remote.starts == [scene.scene_instance_id]

    client.stop_scene(scene.scene_instance_id)

    assert remote.stops == [scene.scene_instance_id]
    assert remote.tunnel.closed


def test_remote_config_requires_explicit_ssh_fields():
    with pytest.raises(ValueError, match="ssh.host"):
        MapPackage.from_dict(
            {
                "map_id": "bad",
                "provider": "packaged_unreal",
                "scene_id": "1",
                "connection": {"mode": "remote"},
                "ssh": {},
                "launch": {"manual": True},
                "control": {"protocol": "airsim"},
                "airsim": {
                    "vehicle_name": "Drone_1",
                    "camera_name": "front_0",
                },
            }
        )


class RecordingSshRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.uploads: list[tuple[Path, str]] = []
        self.downloads: list[tuple[str, Path]] = []

    def run(self, target, remote_args, timeout_sec=30):
        self.commands.append(list(remote_args))
        if "start" in remote_args:
            return '{"ok": true, "pid": 77}\n'
        return ""

    def upload(self, target, local_path, remote_path, timeout_sec=30):
        self.uploads.append((Path(local_path), remote_path))

    def download(self, target, remote_path, local_path, timeout_sec=30):
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}", encoding="utf-8")
        self.downloads.append((remote_path, destination))


def test_remote_scene_control_uploads_worker_settings_and_spec(tmp_path):
    template = tmp_path / "template.json"
    make_template(template)
    package = make_remote_package(template)
    settings = tmp_path / "scene" / "settings.json"
    settings.parent.mkdir()
    settings.write_text("{}", encoding="utf-8")
    runner = RecordingSshRunner()

    result = RemoteSceneControl(runner).start(  # type: ignore[arg-type]
        package,
        "scene_123",
        settings,
        settings.parent,
    )

    uploaded_paths = [remote for _, remote in runner.uploads]
    assert ".flyagent/scenes/scene_123/flyagent_remote_worker.py" in uploaded_paths
    assert ".flyagent/scenes/scene_123/settings.json" in uploaded_paths
    assert ".flyagent/scenes/scene_123/scene-spec.json" in uploaded_paths
    assert runner.commands[-1][-3:] == [
        "start",
        "--spec",
        ".flyagent/scenes/scene_123/scene-spec.json",
    ]
    assert result["pid"] == 77


def test_remote_settings_template_is_downloaded_read_only(tmp_path):
    template = tmp_path / "template.json"
    make_template(template)
    package = make_remote_package(template)
    package.airsim["settings_template_source"] = "remote"
    package.airsim["settings_template"] = "/data/settings/1/settings.json"
    runner = RecordingSshRunner()
    destination = tmp_path / "scene" / "source.settings.json"

    result = RemoteSceneControl(runner).download_settings_template(
        package, destination
    )

    assert result == destination
    assert runner.downloads == [
        ("/data/settings/1/settings.json", destination)
    ]
