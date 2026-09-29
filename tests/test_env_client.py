"""Test SimulatorEnvClient delegation to SimulationServer."""

from __future__ import annotations

from src.env_client.observation_profile import ObservationProfile
from src.env_client.schemas import MapPackage, Observation
from src.env_client.simulator_client import SimulatorEnvClient
from src.sim_server.server import SimulationServer
from tests.support.simulator_fakes import in_memory_backend_registry


def test_local_env_client_delegates_through_server(tmp_path):
    # Verify local pseudo C/S calls flow through the client boundary.
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
    reused = client.get_or_start_scene("m1")
    result = client.execute_action(scene.scene_instance_id, "uav_1", "connect", {})

    assert reused.scene_instance_id == scene.scene_instance_id
    assert result.ok
    assert client.health_check()


def test_two_clients_can_select_different_capture_sources_for_one_service():
    """A run profile must not mutate deployment or leak across clients."""

    class RecordingService:
        def __init__(self) -> None:
            self.capture_sources = []

        def get_observation(
            self,
            scene_instance_id,
            agent_id,
            modalities,
            capture_sources=None,
        ):
            self.capture_sources.append(dict(capture_sources or {}))
            return Observation(scene_instance_id, agent_id, modalities, {})

    service = RecordingService()
    default_client = SimulatorEnvClient(service)
    depth_vis_client = SimulatorEnvClient(
        service,
        ObservationProfile.from_dict(
            {"capture_sources": {"depth": "depth_vis"}}
        ),
    )

    default_client.get_observation("scene_1", "uav_1", ["depth"])
    depth_vis_client.get_observation("scene_1", "uav_1", ["depth"])

    assert service.capture_sources[0]["depth"] == "depth_perspective"
    assert service.capture_sources[1]["depth"] == "depth_vis"


def test_client_preserves_optional_agl_inside_existing_pose_observation():
    """Optional pose enrichment must not require another public client call."""

    class GroundAwareService:
        def get_observation(
            self,
            scene_instance_id,
            agent_id,
            modalities,
            capture_sources=None,
        ):
            return Observation(
                scene_instance_id,
                agent_id,
                modalities,
                {"pose": {"z": 2.5, "agl_m": 2.15}},
            )

    observation = SimulatorEnvClient(GroundAwareService()).get_observation(
        "scene_1", "uav_1", ["pose"]
    )

    assert observation.data["pose"] == {"z": 2.5, "agl_m": 2.15}
