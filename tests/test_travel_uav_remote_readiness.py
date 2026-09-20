"""Read-only TravelUAV remote readiness probe tests."""

from __future__ import annotations

import json

from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from tests.manual.travel_uav_remote_readiness import (
    _REMOTE_SNAPSHOT,
    inspect_remote_readiness,
)
from tests.travel_uav_test_support import case, write_index


class FakeSshRunner:
    def __init__(self, *, listening=False, processes=None, gpu_rows=None):
        self.listening = listening
        self.processes = processes or []
        self.gpu_rows = gpu_rows or ["3, RTX, 24000, 0, 0"]
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
                        "graphics_adapter": item["graphics_adapter"],
                        "is_file": True,
                        "is_executable": True,
                    }
                    for item in packages
                ],
                "settings": {
                    "readable": True,
                    "sha256": "a" * 64,
                    "error": None,
                    "sim_mode": "Multirotor",
                    "api_server_port": 35000,
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


def _config(tmp_path, *, create_key=True):
    index = write_index(tmp_path / "index.json", [case("ready")])
    key = tmp_path / "secret-private-key"
    if create_key:
        key.write_text("fixture", encoding="utf-8")
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
                            "linux_executable": "/remote/Travel.sh",
                            "existing_process_policy": "reject",
                            "args": ["-GraphicsAdapter=3"],
                        },
                        "control": {"protocol": "airsim"},
                        "airsim": {
                            "remote_host": "127.0.0.1",
                            "port": 35000,
                            "mode": "Multirotor",
                            "settings_template_source": "remote",
                            "settings_template": "/remote/settings/1/settings.json",
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
        {
            "index_path": str(index),
            "map_config_path": str(maps),
            "scene_map": {"MapSeen": "travel_MapSeen"},
            "execution_mode": "dry",
            "limit_per_difficulty": None,
        }
    )
