"""Lock the two-axis simulator routing and capability contracts."""

from __future__ import annotations

import json
import math

import pytest

from src.adapters.airsim.uav import AirSimUavAdapter
from src.env_client.schemas import (
    ActionResult,
    MapPackage,
    Observation,
    PortPolicy,
    SceneCapabilities,
)
from src.env_client.simulator_client import SimulatorEnvClient
from src.sim_server.backends.base import BackendProbe
from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.remote.ssh import SshTarget
from src.sim_server.remote.tunnel import SshTunnel
from src.sim_server.server import SimulationServer
from tests.support.simulator_fakes import in_memory_backend_registry


def _template(tmp_path):
    path = tmp_path / "cv.settings.json"
    path.write_text(
        json.dumps(
            {
                "SettingsVersion": 1.2,
                "SimMode": "ComputerVision",
                "Vehicles": {
                    "Drone_1": {
                        "VehicleType": "ComputerVision",
                        "Cameras": {"front_0": {"CaptureSettings": []}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _package(
    map_id: str,
    connection_mode: str,
    protocol: str,
    template,
) -> MapPackage:
    data = {
        "map_id": map_id,
        "provider": "packaged_unreal",
        "scene_id": map_id,
        "root": None,
        "connection": {"mode": connection_mode},
        "launch": {"manual": True},
        "control": {"protocol": protocol},
    }
    if connection_mode == "remote":
        data["connection"]["data_transport"] = "ssh_tunnel"
        data["ssh"] = {
            "host": "192.0.2.10",
            "user": "uav",
            "key_path": "unused-test-key",
        }
    if protocol == "airsim":
        data["airsim"] = {
            "host": "127.0.0.1",
            "remote_host": "127.0.0.1",
            "port": 41451,
            "mode": "ComputerVision",
            "settings_template_source": "local",
            "settings_template": str(template),
            "vehicle_name": "Drone_1",
            "camera_name": "front_0",
        }
    else:
        data["unrealcv"] = {
            "host": "127.0.0.1",
            "remote_host": "127.0.0.1",
            "port": 9010,
            "camera_id": 0,
        }
    return MapPackage.from_dict(data)


def test_one_env_client_routes_all_four_deployment_protocol_pairs(tmp_path):
    template = _template(tmp_path)
    packages = [
        _package("local_air", "local", "airsim", template),
        _package("remote_air", "remote", "airsim", template),
        _package("local_ucv", "local", "unrealcv", template),
        _package("remote_ucv", "remote", "unrealcv", template),
    ]
    service = SimulationServer(
        packages,
        PortPolicy(control_port_start=35001, max_scene_num=8),
        run_dir=tmp_path / "run",
        execution_mode=SimulationExecutionMode.ATTACH,
        remote_scene_control=AttachRemoteControl(),  # type: ignore[arg-type]
        backend_registry=in_memory_backend_registry(),
    )
    client = SimulatorEnvClient(service)

    scenes = [client.start_scene(package.map_id) for package in packages]

    assert [scene.capabilities.protocol for scene in scenes] == [
        "airsim",
        "airsim",
        "unrealcv",
        "unrealcv",
    ]
    assert [scene.machine_id for scene in scenes] == [
        "local",
        "remote",
        "local",
        "remote",
    ]
    assert len({id(value) for value in service.backends.values()}) == 4
    for scene in scenes:
        client.stop_scene(scene.scene_instance_id)


def test_backend_registry_rejects_unknown_protocol():
    registry = BackendRegistry.defaults()

    with pytest.raises(ValueError, match="available: airsim, unrealcv"):
        registry.create("gazebo", None)  # type: ignore[arg-type]


class FakePose:
    def __init__(self) -> None:
        self.position = FakeAirSim.Vector3r(0.0, 0.0, 0.0)
        self.orientation = (0.0, 0.0, 0.0)


class FakeAirSim:
    class Vector3r:
        def __init__(self, x_val, y_val, z_val):
            self.x_val = x_val
            self.y_val = y_val
            self.z_val = z_val

    class Pose:
        def __init__(self, position, orientation):
            self.position = position
            self.orientation = orientation

    @staticmethod
    def to_quaternion(pitch, roll, yaw):
        return (pitch, roll, yaw)

    @staticmethod
    def to_eularian_angles(orientation):
        return orientation


class FakeCvClient:
    def __init__(self) -> None:
        self.pose = FakePose()

    def simGetVehiclePose(self, vehicle_name):
        return self.pose

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name):
        self.pose = pose


def test_computer_vision_strategy_moves_turns_hovers_without_takeoff(monkeypatch):
    adapter = AirSimUavAdapter(
        vehicle_name="Drone_1",
        mode="ComputerVision",
    )
    adapter._client = FakeCvClient()
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    moved = adapter.move_body(0.5, 0.0, 0.0, 1.0)
    turned = adapter.turn_relative(10.0)
    held = adapter.hover()

    assert moved.ok and turned.ok and held.ok
    assert adapter.get_pose().data["position"]["x"] == pytest.approx(0.5)
    assert math.degrees(adapter.get_pose().data["euler_rad"]["yaw"]) == pytest.approx(10.0)
    assert not adapter.takeoff().ok
    assert not adapter.land().ok


class ReadyBackend:
    def probe(self, package, host, port):
        return BackendProbe(True, "ready")

    def describe_capabilities(self, package):
        return SceneCapabilities(
            protocol="airsim",
            control_mode="ComputerVision",
            supported_actions=("connect", "hover", "get_pose", "get_frames"),
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


class FakeTunnel:
    def __init__(self) -> None:
        self.closed = False

    def close(self):
        self.closed = True


class AttachRemoteControl:
    def __init__(self) -> None:
        self.tunnel = FakeTunnel()
        self.starts = 0
        self.stops = 0

    def preflight(self, package):
        return {"python": "3.10"}

    def open_tunnel(self, package, local_port, remote_host, remote_port, log_path=None):
        return self.tunnel

    def start(self, *args):
        self.starts += 1
        raise AssertionError("healthy attach must not start UE")

    def stop(self, *args):
        self.stops += 1
        raise AssertionError("attached UE must not be stopped")


@pytest.mark.parametrize(
    "execution_mode",
    [
        SimulationExecutionMode.ATTACH,
        SimulationExecutionMode.MANAGED,
    ],
)
def test_ready_remote_scene_attaches_without_owning_external_ue(
    tmp_path, execution_mode, monkeypatch
):
    template = _template(tmp_path)
    package = _package("remote_air", "remote", "airsim", template)
    registry = BackendRegistry()
    registry.register("airsim", lambda context: ReadyBackend())
    remote = AttachRemoteControl()
    service = SimulationServer(
        [package],
        PortPolicy(control_port_start=35001, max_scene_num=2),
        run_dir=tmp_path / "run",
        execution_mode=execution_mode,
        remote_scene_control=remote,  # type: ignore[arg-type]
        backend_registry=registry,
    )
    client = SimulatorEnvClient(service)

    # The fake tunnel never binds a socket. Keep this ownership test independent
    # of real evaluation tunnels using the same fixture port on this host.
    monkeypatch.setattr(service.port_manager, "_is_available", lambda port: True)

    scene = client.start_scene("remote_air")
    assert scene.process_ownership == "external"
    client.stop_scene(scene.scene_instance_id)

    assert remote.starts == 0
    assert remote.stops == 0
    assert remote.tunnel.closed


def test_ready_remote_scene_rejects_external_process_when_fresh_required(
    tmp_path, monkeypatch,
):
    template = _template(tmp_path)
    package = _package("remote_air", "remote", "airsim", template)
    package.launch["existing_process_policy"] = "reject"
    registry = BackendRegistry()
    registry.register("airsim", lambda context: ReadyBackend())
    remote = AttachRemoteControl()
    service = SimulationServer(
        [package],
        PortPolicy(control_port_start=35001, max_scene_num=1),
        run_dir=tmp_path / "run",
        execution_mode=SimulationExecutionMode.MANAGED,
        remote_scene_control=remote,  # type: ignore[arg-type]
        backend_registry=registry,
    )

    # This fake tunnel never binds a socket; do not depend on a real engine's
    # unrelated local tunnel occupying the fixed fixture port.
    monkeypatch.setattr(service.port_manager, "_is_available", lambda port: True)
    with pytest.raises(ConnectionError, match="fresh managed process"):
        SimulatorEnvClient(service).start_scene("remote_air")

    assert remote.starts == 0
    assert remote.stops == 0
    assert remote.tunnel.closed


def test_map_package_rejects_unknown_existing_process_policy(tmp_path):
    template = _template(tmp_path)
    package = _package("remote_air", "remote", "airsim", template)
    payload = package.to_dict()
    payload["launch"]["existing_process_policy"] = "replace"

    with pytest.raises(ValueError, match="existing_process_policy"):
        MapPackage.from_dict(payload)


@pytest.mark.parametrize(
    ("mode", "expected_strategy"),
    [
        ("ComputerVision", "pose_kinematic"),
        ("Multirotor", "paused_multirotor"),
    ],
)
def test_map_package_infers_two_strategy_airsim_contract(
    tmp_path,
    mode,
    expected_strategy,
):
    package = _package("strategy", "local", "airsim", _template(tmp_path))
    payload = package.to_dict()
    payload["airsim"]["mode"] = mode
    payload["airsim"].pop("control_strategy", None)

    parsed = MapPackage.from_dict(payload)

    assert parsed.airsim["mode"] == mode
    assert parsed.airsim["control_strategy"] == expected_strategy
    capabilities = AirSimBackend().describe_capabilities(parsed)
    assert capabilities.control_mode == expected_strategy
    assert capabilities.requires_takeoff is False
    assert "takeoff" not in capabilities.supported_actions
    assert "land" not in capabilities.supported_actions


@pytest.mark.parametrize(
    ("mode", "strategy"),
    [
        ("ComputerVision", "paused_multirotor"),
        ("Multirotor", "pose_kinematic"),
        ("ComputerVision", "native_continuous"),
    ],
)
def test_map_package_rejects_invalid_airsim_strategy_pair(
    tmp_path,
    mode,
    strategy,
):
    package = _package("strategy", "local", "airsim", _template(tmp_path))
    payload = package.to_dict()
    payload["airsim"].update(
        {"mode": mode, "control_strategy": strategy}
    )

    with pytest.raises(ValueError, match="control_strategy|requires airsim.mode"):
        MapPackage.from_dict(payload)


class TunnelProcess:
    def __init__(self, command):
        self.command = command
        self.return_code = None
        self.stderr = None

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = 0

    def wait(self, timeout):
        return self.return_code

    def kill(self):
        self.return_code = -9


class TunnelRunner:
    @staticmethod
    def _options(target):
        return ["-p", str(target.port)]


class ConnectedSocket:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_ssh_tunnel_uses_forward_failure_and_keepalive_options(
    tmp_path, monkeypatch
):
    processes = []

    def fake_popen(command, **kwargs):
        process = TunnelProcess(command)
        processes.append(process)
        return process

    monkeypatch.setattr(
        "src.sim_server.remote.tunnel.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "src.sim_server.remote.tunnel.socket.create_connection",
        lambda *args, **kwargs: ConnectedSocket(),
    )
    tunnel = SshTunnel.open(
        TunnelRunner(),  # type: ignore[arg-type]
        SshTarget("example.test", "uav", str(tmp_path / "key"), 2222),
        35001,
        "127.0.0.1",
        30001,
    )

    command = processes[0].command
    assert "ExitOnForwardFailure=yes" in command
    assert "ServerAliveInterval=15" in command
    assert "127.0.0.1:35001:127.0.0.1:30001" in command
    tunnel.close()
    assert processes[0].return_code == 0
