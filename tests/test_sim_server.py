"""Test LocalSimulationServer scene and primitive action behavior."""

from __future__ import annotations

import json

from src.env_client.schemas import MapPackage, PortPolicy, SceneInstance
from src.adapters.airsim_uav import AirSimUavAdapter
from src.sim_server.process_manager import ProcessManager, windows_path_to_wsl
from src.sim_server.server import LocalSimulationServer


def make_package() -> MapPackage:
    # Build a reusable manual-launch MapPackage for tests.
    return MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "packaged_unreal",
            "scene_id": 1,
            "root": None,
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )


def make_launch_package(tmp_path) -> MapPackage:
    # Build a launchable MapPackage that points at a temporary fake script.
    script = tmp_path / "env_1" / "LinuxNoEditor" / "AirVLN.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    return MapPackage.from_dict(
        {
            "map_id": "m_launch",
            "provider": "packaged_unreal",
            "scene_id": 1,
            "root": str(tmp_path / "env_1"),
            "launch": {"method": "wsl_bash", "linux": "LinuxNoEditor/AirVLN.sh"},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )


def make_windows_airsim_package(tmp_path) -> MapPackage:
    # Build a Windows AirSim MapPackage that needs CameraDefaults settings.
    exe = tmp_path / "City" / "Binaries" / "Win64" / "City.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    return MapPackage.from_dict(
        {
            "map_id": "city",
            "provider": "packaged_unreal",
            "scene_id": "city",
            "root": str(tmp_path / "CityRoot"),
            "launch": {"method": "direct", "windows": str(exe)},
            "control": {"protocol": "airsim"},
            "airsim": {
                "vehicle_name": "Drone1",
                "camera_name": "front_center",
                "settings_profile": "camera_defaults",
                "image_width": 224,
                "image_height": 224,
                "depth_width": 256,
                "depth_height": 256,
                "x": 0,
                "y": -5,
                "z": 0,
            },
        }
    )


def test_start_scene_creates_scene_instance_and_settings(tmp_path):
    # Verify scene startup records ports, status, and settings artifacts.
    server = LocalSimulationServer([make_package()], PortPolicy(30000, 30001, 2), tmp_path)

    scene = server.start_scene("m1")

    assert scene.map_id == "m1"
    assert scene.airsim_port == 30001
    assert scene.status == "running"
    assert scene.process_id is None
    assert (tmp_path / "scenes" / scene.scene_instance_id / "settings.json").exists()


def test_camera_defaults_settings_profile_avoids_vehicle_camera_block(tmp_path):
    # Verify Windows AirSim maps can use CameraDefaults instead of per-vehicle Cameras.
    server = LocalSimulationServer([make_windows_airsim_package(tmp_path)], PortPolicy(30000, 30001, 2), tmp_path)

    scene = server.start_scene("city")
    settings = json.loads((tmp_path / "scenes" / scene.scene_instance_id / "settings.json").read_text())

    assert settings["ApiServerPort"] == 30001
    assert "CameraDefaults" in settings
    assert settings["Vehicles"]["Drone1"]["Y"] == -5
    assert "Cameras" not in settings["Vehicles"]["Drone1"]


def test_execute_smoke_actions_and_observation(tmp_path):
    # Verify dry-run primitive actions and raw observations work together.
    server = LocalSimulationServer([make_package()], run_dir=tmp_path)
    scene = server.start_scene("m1")

    assert server.execute_action(scene.scene_instance_id, "uav_1", "connect", {}).ok
    assert server.execute_action(scene.scene_instance_id, "uav_1", "takeoff", {}).ok
    assert server.execute_action(scene.scene_instance_id, "uav_1", "turn_relative", {"yaw_deg": 5}).ok
    observation = server.get_observation(scene.scene_instance_id, "uav_1", ["rgb", "depth", "pose"])

    assert observation.scene_instance_id == scene.scene_instance_id
    assert "pose" in observation.data
    assert "frames" in observation.data


def test_windows_path_to_wsl_converts_drive_paths():
    # Verify Windows drive paths become WSL /mnt paths.
    assert windows_path_to_wsl("G:/airvln-envs/env_1") == "/mnt/g/airvln-envs/env_1"
    assert windows_path_to_wsl(r"C:\Users\HP\run.json") == "/mnt/c/Users/HP/run.json"


def test_wsl_launch_command_includes_settings(tmp_path):
    # Verify WSL launch commands pass the generated settings path.
    package = make_launch_package(tmp_path)
    executable = str(tmp_path / "env_1" / "LinuxNoEditor" / "AirVLN.sh")
    settings = str(tmp_path / "runs" / "scene_1" / "settings.json")

    command = ProcessManager().build_command(
        executable, ["-RenderOffscreen"], method="wsl_bash", settings_path=windows_path_to_wsl(settings)
    )

    assert command[:3] == ["wsl.exe", "bash", "-lc"]
    assert "AirVLN.sh" in command[3]
    assert "--settings" in command[3]
    assert "settings.json" in command[3]
    assert package.map_id == "m_launch"


def test_direct_launch_command_uses_airsim_settings_argument(tmp_path):
    # Verify Windows/direct launches use AirSim's single-dash settings argument.
    exe = tmp_path / "City" / "Binaries" / "Win64" / "City.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    settings = str(tmp_path / "settings.json")

    command = ProcessManager().build_command(str(exe), ["-windowed"], method="direct", settings_path=settings)

    assert command[0] == str(exe)
    assert "-windowed" in command
    assert f"-settings={settings}" in command


def test_launch_backend_uses_real_airsim_adapter(tmp_path):
    # Verify launch-mode servers construct a real AirSim adapter for the backend.
    server = LocalSimulationServer([make_launch_package(tmp_path)], run_dir=tmp_path, launch_processes=True)
    package = server._get_package("m_launch")
    backend = server._backend_for(package)
    scene = SceneInstance("scene_test", "m_launch", "1", "local", "127.0.0.1", 30000, 30001, "settings.json", 123, "created")
    backend.start(package, scene)

    assert backend.use_real_uav
    assert isinstance(backend.uav, AirSimUavAdapter)
