"""Test LocalEnvClient delegation to LocalSimulationServer."""

from __future__ import annotations

from src.env_client.local_client import LocalEnvClient
from src.env_client.schemas import MapPackage
from src.sim_server.server import LocalSimulationServer


def test_local_env_client_delegates_through_server(tmp_path):
    # Verify local pseudo C/S calls flow through the client boundary.
    package = MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "airvln",
            "scene_id": 1,
            "root": None,
            "launch": {"manual": True},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )
    client = LocalEnvClient(LocalSimulationServer([package], run_dir=tmp_path))

    scene = client.start_scene("m1")
    result = client.execute_action(scene.scene_instance_id, "uav_1", "connect", {})

    assert result.ok
    assert client.health_check()
