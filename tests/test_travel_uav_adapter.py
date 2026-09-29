"""TravelUAV coordinate and hidden-truth isolation tests."""

from evaluation.benchmarks.travel_uav.adapter import TravelUavCaseAdapter
from src.env_client import ObservationProfile
from tests.support.travel_uav import case


def test_adapter_converts_ned_and_xyzw_without_leaking_truth():
    episode = TravelUavCaseAdapter(
        scene_map={"MapSeen": "travel_map"},
        observation_profile=ObservationProfile(),
    ).prepare_case(case())

    initial = episode.execution_options.initial_pose
    assert (initial.x, initial.y, initial.z) == (1.0, 2.0, 3.0)
    assert (initial.yaw_deg, initial.pitch_deg, initial.roll_deg) == (0.0, 0.0, 0.0)
    manifest = episode.planner_manifest()
    serialized = str(manifest)
    assert "target_position" not in serialized
    assert "reference_path" not in serialized
    assert "spawn" not in serialized
    assert episode.evaluation_payload.target_position_ned == (101.0, 2.0, -3.0)
