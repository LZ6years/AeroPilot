"""TravelUAV AirSim target-spawn contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.benchmarks.travel_uav.adapter import TravelUavScenarioPayload
from evaluation.benchmarks.travel_uav.scenario import TravelUavScenarioController
from src.env_client import SceneInstance
from tests.travel_uav_test_support import case


class Vector:
    def __init__(self, x, y, z):
        self.x_val, self.y_val, self.z_val = x, y, z


class Quaternion:
    def __init__(self, x, y, z, w):
        self.x_val, self.y_val, self.z_val, self.w_val = x, y, z, w


class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


AIRSIM = SimpleNamespace(Vector3r=Vector, Quaternionr=Quaternion, Pose=Pose)


class FakeClient:
    def __init__(self, offset=0.0):
        self.offset = offset
        self.spawn_calls = []
        self.destroy_calls = []
        self.close_count = 0

    def ping(self):
        return True

    def simSpawnObject(self, *args, **kwargs):
        self.spawn_calls.append((args, kwargs))
        self.pose = args[2]
        return args[0]

    def simGetObjectPose(self, name):
        return Pose(
            Vector(
                self.pose.position.x_val + self.offset,
                self.pose.position.y_val,
                self.pose.position.z_val,
            ),
            self.pose.orientation,
        )

    def simDestroyObject(self, name):
        self.destroy_calls.append(name)
        return True

    def close(self):
        self.close_count += 1


def test_scenario_spawns_official_asset_pose_scale_and_can_destroy():
    client = FakeClient()
    controller = TravelUavScenarioController(
        client_factory=lambda host, port: client,
        airsim_provider=lambda: AIRSIM,
        cleanup_strategy="destroy_object",
    )
    payload = TravelUavScenarioPayload(target=case().target)
    scene = _scene()

    controller.prepare_episode(scene, payload)
    controller.cleanup_episode(scene, payload)

    args, kwargs = client.spawn_calls[0]
    assert args[0] == case().target.object_name
    assert args[1] == "SM_Target"
    assert (args[2].position.x_val, args[2].position.y_val, args[2].position.z_val) == (101, 2, -3)
    assert (args[3].x_val, args[3].y_val, args[3].z_val) == (1, 1, 1)
    assert kwargs == {"physics_enabled": False, "is_blueprint": False}
    assert client.destroy_calls == [case().target.object_name]
    assert client.close_count == 2


def test_scenario_fails_when_spawn_readback_is_wrong():
    controller = TravelUavScenarioController(
        client_factory=lambda host, port: FakeClient(offset=1.0),
        airsim_provider=lambda: AIRSIM,
        position_tolerance_m=0.1,
    )
    with pytest.raises(RuntimeError, match="readback error"):
        controller.prepare_episode(_scene(), TravelUavScenarioPayload(target=case().target))


def _scene():
    return SceneInstance(
        scene_instance_id="scene-1",
        map_id="travel_map",
        scene_id="MapSeen",
        machine_id="local",
        control_host="127.0.0.1",
        control_port=30001,
        settings_path=None,
        process_id=1,
        status="running",
    )
