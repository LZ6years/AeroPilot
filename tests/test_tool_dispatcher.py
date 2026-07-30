"""Test ToolDispatcher routing through EnvClient."""

from __future__ import annotations

from src.env_client.simulator_client import SimulatorEnvClient
from src.env_client.schemas import MapPackage
from src.sim_server.server import SimulationServer
from src.tools.dispatcher import ToolDispatcher
from tests.simulator_fakes import in_memory_backend_registry


def test_tool_dispatcher_routes_uav_actions_through_env_client(tmp_path):
    # Verify tool names are translated into primitive EnvClient actions.
    package = MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "airvln",
            "scene_id": 1,
            "root": None,
            "connection": {"mode": "local"},
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )
    client = SimulatorEnvClient(
        SimulationServer(
            [package],
            run_dir=tmp_path,
            backend_registry=in_memory_backend_registry(),
        )
    )
    scene = client.start_scene("m1")
    dispatcher = ToolDispatcher(client)

    result = dispatcher.call(scene.scene_instance_id, "uav_1", "uav.connect", {})
    set_pose = dispatcher.call(
        scene.scene_instance_id,
        "uav_1",
        "uav.set_pose",
        {"x": 4.0, "y": 5.0, "z": -6.0, "yaw_deg": 90.0},
    )

    assert result.ok
    assert set_pose.ok
    assert set_pose.data["pose"]["yaw_deg"] == 90.0
    assert set_pose.data["pose"]["z"] == -6.0
    assert "uav.connect" in dispatcher.registry.names()
    assert "uav.set_pose" in dispatcher.registry.names()
