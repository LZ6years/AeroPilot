"""Test LocalSimulationServer scene and primitive action behavior."""

from __future__ import annotations

from src.env_client.schemas import MapPackage, PortPolicy
from src.sim_server.server import LocalSimulationServer


def make_package() -> MapPackage:
    # Build a reusable manual-launch MapPackage for tests.
    return MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "airvln",
            "scene_id": 1,
            "root": None,
            "launch": {"manual": True},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
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
