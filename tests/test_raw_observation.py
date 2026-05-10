"""Test raw observation normalization."""

from __future__ import annotations

from src.env_client.schemas import Observation
from src.perception.raw_observation import RawObservation


def test_raw_observation_from_env_observation_extracts_common_fields():
    observation = Observation(
        scene_instance_id="scene_1",
        agent_id="uav_1",
        modalities=["rgb", "depth", "pose"],
        data={
            "pose": {"pose": {"x": 1.0, "y": 2.0, "z": -3.0}},
            "frames": {"frames": {"rgb": {"artifact": "rgb.png"}}},
        },
    )

    raw = RawObservation.from_env_observation(observation, source="preflight", timestamp="t0")

    assert raw.scene_instance_id == "scene_1"
    assert raw.agent_id == "uav_1"
    assert raw.modalities == ["rgb", "depth", "pose"]
    assert raw.pose == {"pose": {"x": 1.0, "y": 2.0, "z": -3.0}}
    assert raw.frames == {"frames": {"rgb": {"artifact": "rgb.png"}}}
    assert raw.source == "preflight"
    assert raw.timestamp == "t0"
    assert raw.to_dict()["data"] == observation.data


def test_raw_observation_missing_pose_or_frames_keeps_raw_data():
    raw = RawObservation.from_env_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["pose"],
            "data": {"custom": {"ok": True}},
        },
        timestamp="t0",
    )

    assert raw.pose == {}
    assert raw.frames == {}
    assert raw.data == {"custom": {"ok": True}}
