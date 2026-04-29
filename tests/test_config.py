"""Test map configuration loading and validation."""

from __future__ import annotations

import json

import pytest

from src.runtime.config import load_map_config


def test_load_map_config_valid(tmp_path):
    # Verify a minimal valid map config becomes typed runtime objects.
    config = tmp_path / "maps.json"
    config.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "map_id": "m1",
                        "provider": "airvln",
                        "scene_id": 1,
                        "root": None,
                        "launch": {"manual": True},
                        "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
                    }
                ],
                "port_policy": {"server_port": 1, "airsim_port_start": 2, "max_scene_num": 1},
            }
        ),
        encoding="utf-8",
    )

    maps, policy = load_map_config(config)

    assert maps[0].map_id == "m1"
    assert policy.airsim_port_start == 2


def test_load_map_config_requires_vehicle_and_camera(tmp_path):
    # Verify missing AirSim vehicle/camera fields are rejected clearly.
    config = tmp_path / "maps.json"
    config.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "map_id": "m1",
                        "provider": "airvln",
                        "scene_id": 1,
                        "launch": {"manual": True},
                        "airsim": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="airsim.vehicle_name"):
        load_map_config(config)
