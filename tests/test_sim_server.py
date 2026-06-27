"""Test LocalSimulationServer scene and primitive action behavior."""

from __future__ import annotations

import json

import src.sim_server.server as server_module
from src.env_client.schemas import MapPackage, PortPolicy, SceneInstance
from src.adapters.airsim_uav import AirSimUavAdapter
from src.runtime.configs.action_config import UavActionConfig
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


def test_get_or_start_scene_reuses_running_scene(tmp_path):
    server = LocalSimulationServer([make_package()], PortPolicy(30000, 30001, 2), tmp_path)

    first = server.get_or_start_scene("m1")
    second = server.get_or_start_scene("m1")
    isolated = server.start_scene("m1")

    assert second.scene_instance_id == first.scene_instance_id
    assert isolated.scene_instance_id != first.scene_instance_id


def test_get_or_start_scene_does_not_reuse_stopped_scene(tmp_path):
    server = LocalSimulationServer([make_package()], PortPolicy(30000, 30001, 2), tmp_path)

    first = server.get_or_start_scene("m1")
    server.stop_scene(first.scene_instance_id)
    second = server.get_or_start_scene("m1")

    assert second.scene_instance_id != first.scene_instance_id


def test_camera_defaults_settings_profile_avoids_vehicle_camera_block(tmp_path):
    # Verify Windows AirSim maps can use CameraDefaults instead of per-vehicle Cameras.
    server = LocalSimulationServer([make_windows_airsim_package(tmp_path)], PortPolicy(30000, 30001, 2), tmp_path)

    scene = server.start_scene("city")
    settings = json.loads((tmp_path / "scenes" / scene.scene_instance_id / "settings.json").read_text())

    assert settings["ApiServerPort"] == 30001
    assert "CameraDefaults" in settings
    assert settings["Vehicles"]["Drone1"]["Y"] == -5
    assert "Cameras" not in settings["Vehicles"]["Drone1"]


def test_show_display_overrides_airsim_view_mode(tmp_path):
    # Verify the backend can force visible AirSim FPV mode at runtime.
    server = LocalSimulationServer(
        [make_windows_airsim_package(tmp_path)],
        PortPolicy(30000, 30001, 2),
        tmp_path,
        show_display=True,
    )

    scene = server.start_scene("city")
    settings = json.loads((tmp_path / "scenes" / scene.scene_instance_id / "settings.json").read_text())

    assert settings["ViewMode"] == "Fpv"


def test_hide_display_overrides_airsim_view_mode(tmp_path):
    # Verify the backend can force AirSim NoDisplay mode at runtime.
    package = make_windows_airsim_package(tmp_path)
    package.airsim["view_mode"] = "Fpv"
    server = LocalSimulationServer([package], PortPolicy(30000, 30001, 2), tmp_path, show_display=False)

    scene = server.start_scene("city")
    settings = json.loads((tmp_path / "scenes" / scene.scene_instance_id / "settings.json").read_text())

    assert settings["ViewMode"] == "NoDisplay"


def test_airsim_wait_reports_rpc_timeout(tmp_path):
    # Verify AirSim readiness checks wait for RPC ping instead of only TCP.
    server = LocalSimulationServer([make_package()], run_dir=tmp_path)

    try:
        server._wait_for_airsim_rpc("127.0.0.1", 9, 0.01, "settings.json")
    except TimeoutError as exc:
        assert "AirSim RPC did not become ready" in str(exc)


def test_post_launch_wait_uses_package_stabilization_delay(tmp_path, monkeypatch):
    # Verify launched scenes can wait briefly after RPC readiness for UE stabilization.
    sleeps = []
    monkeypatch.setattr(server_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    package = make_package()
    package.launch["post_launch_wait_sec"] = 2.5
    server = LocalSimulationServer([package], run_dir=tmp_path)

    server._post_launch_wait(package)

    assert sleeps == [2.5]


def test_post_launch_wait_defaults_to_five_seconds(tmp_path, monkeypatch):
    # Verify launch stabilization has a default even when map configs omit it.
    sleeps = []
    monkeypatch.setattr(server_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    package = make_package()
    server = LocalSimulationServer([package], run_dir=tmp_path)

    server._post_launch_wait(package)

    assert sleeps == [5.0]


def test_execute_smoke_actions_and_observation(tmp_path):
    # Verify dry-run primitive actions and raw observations work together.
    server = LocalSimulationServer([make_package()], run_dir=tmp_path)
    scene = server.start_scene("m1")

    assert server.execute_action(scene.scene_instance_id, "uav_1", "connect", {}).ok
    assert server.execute_action(scene.scene_instance_id, "uav_1", "takeoff", {}).ok
    set_pose = server.execute_action(
        scene.scene_instance_id,
        "uav_1",
        "set_pose",
        {"x": 1.0, "y": 2.0, "z": -3.0, "yaw_deg": 30.0},
    )
    assert set_pose.ok
    assert set_pose.data["pose"]["x"] == 1.0
    assert set_pose.data["pose"]["z"] == -3.0
    assert server.execute_action(scene.scene_instance_id, "uav_1", "turn_relative", {"yaw_deg": 5}).ok
    observation = server.get_observation(scene.scene_instance_id, "uav_1", ["rgb", "depth", "pose"])

    assert observation.scene_instance_id == scene.scene_instance_id
    assert "pose" in observation.data
    assert "frames" in observation.data


def test_action_config_provides_default_move_turn_and_vertical_steps(tmp_path):
    # Verify primitive actions can omit physical numbers and use the shared action scale.
    action_config = UavActionConfig(
        linear_velocity_mps=2.0,
        vertical_velocity_mps=0.5,
        move_duration_sec=4.0,
        vertical_duration_sec=2.0,
        vertical_step_m=1.5,
        yaw_step_deg=20.0,
    )
    server = LocalSimulationServer([make_package()], run_dir=tmp_path, action_config=action_config)
    scene = server.start_scene("m1")

    assert server.execute_action(scene.scene_instance_id, "uav_1", "connect", {}).ok
    move = server.execute_action(scene.scene_instance_id, "uav_1", "move_body", {"direction": "forward"})
    turn = server.execute_action(scene.scene_instance_id, "uav_1", "turn_relative", {})
    ascend = server.execute_action(scene.scene_instance_id, "uav_1", "ascend", {})
    descend = server.execute_action(scene.scene_instance_id, "uav_1", "descend", {})

    assert move.data["pose"]["x"] == 8.0
    assert turn.data["pose"]["yaw_deg"] == 20.0
    assert ascend.data["pose"]["z"] == 1.0
    assert descend.data["pose"]["z"] == 0.0


def test_action_config_direction_supports_backward_and_lateral_moves(tmp_path):
    # Verify semantic movement directions resolve to the same configured body-frame scale.
    action_config = UavActionConfig(linear_velocity_mps=1.0, move_duration_sec=2.0)
    server = LocalSimulationServer([make_package()], run_dir=tmp_path, action_config=action_config)
    scene = server.start_scene("m1")

    assert server.execute_action(scene.scene_instance_id, "uav_1", "connect", {}).ok
    assert server.execute_action(scene.scene_instance_id, "uav_1", "move_body", {"direction": "backward"}).ok
    left = server.execute_action(scene.scene_instance_id, "uav_1", "move_body", {"direction": "left"})
    right = server.execute_action(scene.scene_instance_id, "uav_1", "move_body", {"direction": "right"})

    assert left.data["pose"]["x"] == -2.0
    assert left.data["pose"]["y"] == -2.0
    assert right.data["pose"]["y"] == 0.0


def test_vertical_actions_support_duration_speed_and_meters(tmp_path):
    # Verify ascend/descend resolve duration, speed, and meters into speed-duration motion.
    action_config = UavActionConfig(vertical_velocity_mps=1.0, vertical_duration_sec=1.0)
    server = LocalSimulationServer([make_package()], run_dir=tmp_path, action_config=action_config)
    scene = server.start_scene("m1")

    assert server.execute_action(scene.scene_instance_id, "uav_1", "connect", {}).ok
    ascend_duration = server.execute_action(scene.scene_instance_id, "uav_1", "ascend", {"duration": 3})
    descend_speed_duration = server.execute_action(
        scene.scene_instance_id,
        "uav_1",
        "descend",
        {"speed_mps": 0.5, "duration": 4},
    )
    ascend_meters = server.execute_action(scene.scene_instance_id, "uav_1", "ascend", {"meters": 2})

    assert ascend_duration.data["pose"]["z"] == 3.0
    assert descend_speed_duration.data["pose"]["z"] == 1.0
    assert ascend_meters.data["pose"]["z"] == 3.0


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


def test_show_display_removes_render_offscreen_arg(tmp_path):
    # Verify visible launches do not keep the offscreen rendering flag.
    package = make_windows_airsim_package(tmp_path)
    package.launch["args"] = ["-windowed", "-RenderOffscreen", "-NoSound"]
    server = LocalSimulationServer([package], run_dir=tmp_path, show_display=True)

    assert server._launch_args(package) == ["-windowed", "-NoSound"]


def test_launch_backend_uses_real_airsim_adapter(tmp_path):
    # Verify launch-mode servers construct a real AirSim adapter for the backend.
    server = LocalSimulationServer([make_launch_package(tmp_path)], run_dir=tmp_path, launch_processes=True)
    package = server._get_package("m_launch")
    backend = server._backend_for(package)
    scene = SceneInstance("scene_test", "m_launch", "1", "local", "127.0.0.1", 30000, 30001, "settings.json", 123, "created")
    backend.start(package, scene)

    assert backend.use_real_uav
    assert isinstance(backend.uav, AirSimUavAdapter)
