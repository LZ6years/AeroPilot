"""Test ToolDispatcher routing through EnvClient."""

from __future__ import annotations

from src.env_client.local_client import LocalEnvClient
from src.env_client.schemas import MapPackage
from src.sim_server.server import LocalSimulationServer
from src.tools.dispatcher import ToolDispatcher


def test_tool_dispatcher_routes_uav_actions_through_env_client(tmp_path):
    # Verify tool names are translated into primitive EnvClient actions.
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
    dispatcher = ToolDispatcher(client)

    result = dispatcher.call(scene.scene_instance_id, "uav_1", "uav.connect", {})

    assert result.ok
    assert "uav.connect" in dispatcher.registry.names()
