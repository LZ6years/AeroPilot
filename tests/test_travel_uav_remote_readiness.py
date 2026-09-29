"""Read-only TravelUAV remote readiness probe tests."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from tests.support.environment import (
    TRAVEL_REMOTE_SNAPSHOT as _REMOTE_SNAPSHOT,
    inspect_travel_remote_readiness as inspect_remote_readiness,
)
from tests.travel_uav_test_support import case, write_index


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeSshRunner:
    def __init__(
        self,
        *,
        listening=False,
        processes=None,
        gpu_rows=None,
        launchable=True,
    ):
        self.listening = listening
        self.processes = processes or []
        self.gpu_rows = gpu_rows or ["3, RTX, 24000, 0, 0"]
        self.launchable = launchable
        self.commands = []

    def run(self, target, remote_args, timeout_sec=30):
        self.commands.append(list(remote_args))
        packages = json.loads(remote_args[3])
        return json.dumps(
            {
                "python": "3.10",
                "scenes": [
                    {
                        "scene_id": item["scene_id"],
                        "map_id": item["map_id"],
                        "executable": item["executable"],
                        "launch_method": item["launch_method"],
                        "graphics_adapter": item["graphics_adapter"],
                        "is_file": True,
                        "is_readable": True,
                        "is_executable": False,
                        "bash_syntax_ok": True,
                        "launchable": self.launchable,
                    }
                    for item in packages
                ],
                "settings": {
                    "readable": True,
                    "sha256": "a" * 64,
                    "error": None,
                    "sim_mode": "Multirotor",
                    "api_server_port": None,
                    "rendered_api_server_port": 35000,
                    "vehicle_present": True,
                    "camera_present": True,
                    "capture_sizes": {"0": [256, 256], "2": [256, 256]},
                },
                "remote_endpoint": {
                    "host": "127.0.0.1",
                    "port": 35000,
                    "listening": self.listening,
                },
                "gpu": {"ok": True, "rows": self.gpu_rows},
                "airsim_processes": self.processes,
            }
        )


def test_readiness_checks_selected_map_without_remote_writes(tmp_path):
    runner = FakeSshRunner()
    result = inspect_remote_readiness(_config(tmp_path), ssh_runner=runner)

    assert result["ready"] is True
    assert len(runner.commands) == 1
    assert runner.commands[0][:2] == ["python3", "-c"]
    assert not any(token in runner.commands[0] for token in ("mkdir", "scp", "start", "stop"))
    assert runner.commands[0][4] == "local"


def test_readiness_accepts_readable_non_executable_bash_script(tmp_path):
    result = inspect_remote_readiness(_config(tmp_path), ssh_runner=FakeSshRunner())

    scene = result["remote_snapshots"][0]["scenes"][0]
    assert scene["launch_method"] == "linux_bash"
    assert scene["is_executable"] is False
    assert result["ready"] is True


def test_readiness_rejects_unlaunchable_script(tmp_path):
    result = inspect_remote_readiness(
        _config(tmp_path), ssh_runner=FakeSshRunner(launchable=False)
    )

    assert result["ready"] is False
    assert any(item["stage"] == "scene_executable" for item in result["errors"])


def test_readiness_rejects_existing_process_or_port(tmp_path):
    result = inspect_remote_readiness(
        _config(tmp_path),
        ssh_runner=FakeSshRunner(listening=True, processes=["123 CarlaUE4"]),
    )
    assert result["ready"] is False
    assert any(item["stage"] == "existing_process" for item in result["errors"])


def test_readiness_redacts_missing_private_key_path(tmp_path):
    config = _config(tmp_path, create_key=False)
    result = inspect_remote_readiness(config, ssh_runner=FakeSshRunner())
    assert result["ready"] is False
    assert "secret-private-key" not in json.dumps(result)


def test_remote_process_detection_uses_executable_not_probe_arguments():
    assert 'Path("/proc")' in _REMOTE_SNAPSHOT
    assert "os.readlink(process_dir / \"exe\")" in _REMOTE_SNAPSHOT
    assert "pgrep" not in _REMOTE_SNAPSHOT


def test_readiness_checks_configured_graphics_adapter(tmp_path):
    result = inspect_remote_readiness(
        _config(tmp_path),
        ssh_runner=FakeSshRunner(gpu_rows=["0, RTX, 24000, 0, 0"]),
    )
    assert result["ready"] is False
    assert any(item["stage"] == "gpu" for item in result["errors"])


def test_repository_travel_map_contract_matches_materialized_layout():
    payload = json.loads(
        (REPO_ROOT / "configs/maps/travel_uav_remote.example.json").read_text(
            encoding="utf-8"
        )
    )
    maps = payload["maps"]

    assert len(maps) == 12
    assert {item["scene_id"] for item in maps}.isdisjoint(
        {"Carla_Town02", "Carla_Town03", "ModularEuropean"}
    )
    for item in maps:
        launch = item["launch"]
        airsim = item["airsim"]
        assert launch["method"] == "linux_bash"
        assert launch["settings_arg_style"] == "single_dash_equals"
        assert "-GraphicsAdapter=3" in launch["args"]
        assert airsim["port"] == 35000
        assert airsim["settings_template_source"] == "local"
        assert airsim["settings_template"] == (
            "configs/airsim/travel_uav_official.settings.example.json"
        )
        executable = launch["linux_executable"]
        if item["scene_id"].startswith("Carla_"):
            assert "/carla_town_envs/Town" in executable
            assert executable.endswith("/LinuxNoEditor/CarlaUE4.sh")
        else:
            assert executable.endswith(f"/closeloop_envs/{item['scene_id']}.sh")


def test_official_settings_template_keeps_front_camera_contract():
    settings = json.loads(
        (
            REPO_ROOT
            / "configs/airsim/travel_uav_official.settings.example.json"
        ).read_text(encoding="utf-8")
    )
    vehicle = settings["Vehicles"]["Drone_1"]
    captures = vehicle["Cameras"]["FrontCamera"]["CaptureSettings"]

    assert settings["SimMode"] == "Multirotor"
    assert settings["ClockSpeed"] == 10
    assert vehicle["VehicleType"] == "SimpleFlight"
    assert {
        int(item["ImageType"]): [int(item["Width"]), int(item["Height"])]
        for item in captures
    } == {0: [256, 256], 2: [256, 256]}


def _config(tmp_path, *, create_key=True):
    index = write_index(tmp_path / "index.json", [case("ready")])
    key = tmp_path / "secret-private-key"
    if create_key:
        key.write_text("fixture", encoding="utf-8")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "SimMode": "Multirotor",
                "ApiServerPort": 35000,
                "Vehicles": {
                    "Drone_1": {
                        "Cameras": {
                            "FrontCamera": {
                                "CaptureSettings": [
                                    {"ImageType": 0, "Width": 256, "Height": 256},
                                    {"ImageType": 2, "Width": 256, "Height": 256},
                                ]
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    maps = tmp_path / "maps.json"
    maps.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "map_id": "travel_MapSeen",
                        "provider": "fixture",
                        "scene_id": "MapSeen",
                        "root": None,
                        "connection": {"mode": "remote", "data_transport": "ssh_tunnel"},
                        "ssh": {
                            "host": "192.0.2.1",
                            "port": 22,
                            "user": "tester",
                            "key_path": str(key),
                        },
                        "launch": {
                            "method": "linux_bash",
                            "linux_executable": "/remote/Travel.sh",
                            "existing_process_policy": "reject",
                            "args": ["-GraphicsAdapter=3"],
                        },
                        "control": {"protocol": "airsim"},
                        "airsim": {
                            "remote_host": "127.0.0.1",
                            "port": 35000,
                            "mode": "Multirotor",
                            "settings_template_source": "local",
                            "settings_template": str(settings),
                            "vehicle_name": "Drone_1",
                            "camera_name": "FrontCamera",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return TravelUavEvalConfig.from_dict(
        {**{"execution_mode": "managed", "save_artifacts": True, "enable_action_planner_debug": True},
            "index_path": str(index),
            "expected_index_sha256": __import__("hashlib").sha256(index.read_bytes()).hexdigest(),
            "map_config_path": str(maps),
            "scene_map": {"MapSeen": "travel_MapSeen"},
            "execution_mode": "managed",
            "limit_per_difficulty": None,
        }
    )
