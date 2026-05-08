"""Test map configuration loading and validation."""

from __future__ import annotations

import json

import pytest

from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config
from src.runtime.configs.map_config import build_remote_airsim_config, load_map_config
from test_uav_client_demo import load_demo_action_config


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


def test_load_map_config_accepts_unrealcv_package(tmp_path):
    # Verify packaged maps can select UnrealCV as their control protocol.
    config = tmp_path / "maps.json"
    config.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "map_id": "uzoo",
                        "provider": "packaged_unreal",
                        "scene_id": "collection",
                        "root": "G:/UnrealZoo-UE4/Collection_WinNoEditor_0424_25",
                        "launch": {"method": "direct", "windows": "Collection.exe"},
                        "control": {"protocol": "unrealcv"},
                        "unrealcv": {"host": "127.0.0.1", "port": 9010, "camera_id": 0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    maps, _ = load_map_config(config)

    assert maps[0].provider == "packaged_unreal"
    assert maps[0].control_protocol == "unrealcv"
    assert maps[0].unrealcv["port"] == 9010


def test_load_map_config_accepts_existing_remote_process(tmp_path):
    # Verify remote already-running UE processes do not require local root paths.
    config = tmp_path / "maps.json"
    config.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "map_id": "remote_uzoo",
                        "provider": "packaged_unreal",
                        "scene_id": "collection",
                        "root": None,
                        "launch": {"method": "existing_process"},
                        "control": {"protocol": "unrealcv"},
                        "unrealcv": {"host": "192.168.1.20", "port": 9010, "camera_id": 0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    maps, _ = load_map_config(config)

    assert maps[0].root is None
    assert maps[0].unrealcv["host"] == "192.168.1.20"


def test_build_remote_airsim_config_defaults():
    # Verify the temporary remote Linux AirSim config matches the planned smoke test.
    maps, policy = build_remote_airsim_config("192.168.1.30")

    package = maps[0]
    assert package.map_id == "remote_linux_airsim"
    assert package.launch["method"] == "existing_process"
    assert package.control_protocol == "airsim"
    assert package.airsim["host"] == "192.168.1.30"
    assert package.airsim["port"] == 41451
    assert package.airsim["vehicle_name"] == "Drone_1"
    assert package.airsim["camera_name"] == "front_0"
    assert policy.server_port == 30000


def test_build_remote_airsim_config_allows_computer_vision_mode():
    # Verify remote AirVLN observation tests can target ComputerVision scenes.
    maps, _ = build_remote_airsim_config("127.0.0.1", port=30001, mode="ComputerVision")

    assert maps[0].airsim["port"] == 30001
    assert maps[0].airsim["mode"] == "ComputerVision"


def test_load_uav_action_config_defaults_when_file_missing(tmp_path):
    # Verify missing action config files use the built-in fixed action scale.
    config = load_uav_action_config(tmp_path / "missing.json")

    assert config == UavActionConfig()


def test_load_uav_action_config_reads_fixed_scale_values(tmp_path):
    # Verify UAV action scale config controls move speed, duration, vertical step, and yaw step.
    config_path = tmp_path / "uav_action_config.json"
    config_path.write_text(
        json.dumps(
            {
                "linear_velocity_mps": 2.0,
                "vertical_velocity_mps": 0.75,
                "move_duration_sec": 4.0,
                "vertical_duration_sec": 2.0,
                "vertical_step_m": 1.5,
                "yaw_step_deg": 20.0,
            }
        ),
        encoding="utf-8",
    )

    config = load_uav_action_config(config_path)

    assert config.linear_velocity_mps == 2.0
    assert config.vertical_velocity_mps == 0.75
    assert config.move_duration_sec == 4.0
    assert config.vertical_duration_sec == 2.0
    assert config.vertical_step_m == 1.5
    assert config.yaw_step_deg == 20.0


def test_demo_move_duration_override_preserves_vertical_duration(tmp_path):
    # Verify --move-duration only overrides horizontal movement duration.
    config_path = tmp_path / "uav_action_config.json"
    config_path.write_text(
        json.dumps(
            {
                "linear_velocity_mps": 2.0,
                "vertical_velocity_mps": 2.0,
                "move_duration_sec": 2.0,
                "vertical_duration_sec": 8.0,
                "vertical_step_m": 1.0,
                "yaw_step_deg": 30.0,
            }
        ),
        encoding="utf-8",
    )

    args = type("Args", (), {"action_config": str(config_path), "move_duration": 8.0})()
    config = load_demo_action_config(args)

    assert config.move_duration_sec == 8.0
    assert config.vertical_duration_sec == 8.0
