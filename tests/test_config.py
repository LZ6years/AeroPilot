"""Test map configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config
from src.runtime.configs.safety_config import UavSafetyConfig, load_uav_safety_config
from src.runtime.configs.stabilization_config import UavStabilizationConfig, load_uav_stabilization_config
from src.runtime.configs.map_config import load_map_config


REPO_ROOT = Path(__file__).resolve().parents[1]


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
                        "connection": {"mode": "local"},
                        "launch": {"manual": True},
                        "control": {"protocol": "airsim"},
                        "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
                    }
                ],
                "port_policy": {"control_port_start": 2, "max_scene_num": 1},
            }
        ),
        encoding="utf-8",
    )

    maps, policy = load_map_config(config)

    assert maps[0].map_id == "m1"
    assert policy.control_port_start == 2


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
                        "connection": {"mode": "local"},
                        "launch": {"manual": True},
                        "control": {"protocol": "airsim"},
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
                        "connection": {"mode": "local"},
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


def test_unrealcv_actor_example_uses_verified_uav_flow_camera_contract():
    maps, _ = load_map_config(
        REPO_ROOT / "configs" / "maps" / "unrealcv_actor_control.example.json"
    )

    actor = maps[0].unrealcv["actor"]
    assert maps[0].unrealcv["camera_id"] == 0
    assert actor["name"] not in maps[0].unrealcv["startup"]["hide_objects"]
    assert actor["rotation_method"] == "blueprint_yaw"
    assert actor["camera_follow"] == "standard_pose"
    assert actor["camera_validation"] == {
        "enabled": True,
        "position_mode": "max_distance",
        "expected_body_offset_m": [0.0, 0.0, 0.0],
        "position_tolerance_m": 0.05,
        "max_distance_m": 1.5,
        "yaw_tolerance_deg": 1.0,
    }
    assert actor["yaw_offset_deg"] == -180.0


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
                        "connection": {
                            "mode": "remote",
                            "data_transport": "ssh_tunnel",
                        },
                        "ssh": {
                            "host": "192.168.1.20",
                            "user": "uav",
                            "key_path": "key",
                        },
                        "launch": {"manual": True},
                        "control": {"protocol": "unrealcv"},
                        "unrealcv": {
                            "remote_host": "127.0.0.1",
                            "host": "127.0.0.1",
                            "port": 9010,
                            "camera_id": 0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    maps, _ = load_map_config(config)

    assert maps[0].root is None
    assert maps[0].connection_mode == "remote"
    assert maps[0].unrealcv["remote_host"] == "127.0.0.1"


def test_aerial_vln_remote_map_inventory_covers_all_official_scenes():
    maps, policy = load_map_config(
        REPO_ROOT
        / "configs"
        / "maps"
        / "aerial_vln_s_remote.example.json"
    )

    expected_scene_ids = {
        *(str(value) for value in range(1, 19)),
        *(str(value) for value in range(20, 27)),
    }
    assert {package.scene_id for package in maps} == expected_scene_ids
    assert len(maps) == 25
    assert policy.control_port_start == 35001
    assert policy.max_scene_num == 1
    assert all(
        package.launch["existing_process_policy"] == "reject"
        for package in maps
    )
    assert all(package.airsim["port"] == 30001 for package in maps)


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


def test_load_uav_safety_config_defaults_when_file_missing(tmp_path):
    config = load_uav_safety_config(tmp_path / "missing.json")

    assert config == UavSafetyConfig()


def test_load_uav_safety_config_reads_hard_limits(tmp_path):
    config_path = tmp_path / "uav_safety_config.json"
    config_path.write_text(
        json.dumps(
            {
                "max_yaw_step_deg": 35.0,
                "max_linear_velocity_mps": 7.0,
                "max_vertical_velocity_mps": 4.0,
                "max_move_duration_sec": 9.0,
                "max_yaw_scan_accum_deg": 500.0,
            }
        ),
        encoding="utf-8",
    )

    config = load_uav_safety_config(config_path)

    assert config.max_yaw_step_deg == 35.0
    assert config.max_linear_velocity_mps == 7.0
    assert config.max_vertical_velocity_mps == 4.0
    assert config.max_move_duration_sec == 9.0
    assert config.max_yaw_scan_extent_deg == 500.0


def test_load_uav_safety_config_prefers_canonical_yaw_scan_extent(tmp_path):
    config_path = tmp_path / "uav_safety_config.json"
    config_path.write_text(
        json.dumps(
            {
                "max_yaw_scan_extent_deg": 420.0,
                "max_yaw_scan_accum_deg": 500.0,
            }
        ),
        encoding="utf-8",
    )

    config = load_uav_safety_config(config_path)

    assert config.max_yaw_scan_extent_deg == 420.0
    assert config.to_dict()["max_yaw_scan_extent_deg"] == 420.0
    assert "max_yaw_scan_accum_deg" not in config.to_dict()


def test_load_uav_stabilization_config_defaults_when_file_missing(tmp_path):
    config = load_uav_stabilization_config(tmp_path / "missing.json")

    assert config == UavStabilizationConfig()
    with pytest.raises(ValueError, match="finite non-negative"):
        UavStabilizationConfig(initialization_min_settle_sec=-0.1)


def test_load_uav_stabilization_config_reads_runtime_wait_policy(tmp_path):
    config_path = tmp_path / "uav_stabilization_config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": False,
                "velocity_threshold_mps": 0.2,
                "timeout_sec": 4.5,
                "poll_interval_sec": 0.1,
                "initialization_min_settle_sec": 3.5,
                "zero_velocity_brake_duration_sec": 1.5,
                "hover_after_brake": True,
                "motion_tools": ["uav.move_body"],
            }
        ),
        encoding="utf-8",
    )

    config = load_uav_stabilization_config(config_path)

    assert config.enabled is False
    assert config.velocity_threshold_mps == 0.2
    assert config.timeout_sec == 4.5
    assert config.poll_interval_sec == 0.1
    assert config.initialization_min_settle_sec == 3.5
    assert config.zero_velocity_brake_duration_sec == 1.5
    assert config.hover_after_brake is True
    assert config.hover_before_wait is True
    assert config.motion_tools == {"uav.move_body"}
    assert config.to_dict()["hover_after_brake"] is True
    assert config.to_dict()["initialization_min_settle_sec"] == 3.5
    assert config.to_dict()["motion_tools"] == ["uav.move_body"]


def test_load_uav_stabilization_config_reads_legacy_hover_key(tmp_path):
    config_path = tmp_path / "uav_stabilization_config.json"
    config_path.write_text(json.dumps({"hover_before_wait": True}), encoding="utf-8")

    config = load_uav_stabilization_config(config_path)

    assert config.hover_after_brake is True
