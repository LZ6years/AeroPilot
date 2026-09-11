"""Test the read-only AerialVLN remote readiness audit."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from tests.manual.aerial_vln_remote_readiness import (
    _REMOTE_SNAPSHOT,
    inspect_remote_readiness,
)


class FakeSshRunner:
    def __init__(self, *, port_listening: bool = False) -> None:
        self.port_listening = port_listening
        self.commands: list[list[str]] = []

    def run(self, target, remote_args, timeout_sec=30):
        self.commands.append(list(remote_args))
        packages = json.loads(remote_args[3])
        return json.dumps(
            {
                "python": "3.10.12",
                "scenes": [
                    {
                        "scene_id": item["scene_id"],
                        "map_id": item["map_id"],
                        "executable": item["executable"],
                        "is_file": True,
                        "is_executable": True,
                    }
                    for item in packages
                ],
                "settings": {
                    "path": "/remote/settings/1/settings.json",
                    "readable": True,
                    "sha256": "a" * 64,
                    "error": None,
                    "sim_mode": "ComputerVision",
                    "api_server_port": 30001,
                    "vehicle_present": True,
                    "camera_present": True,
                    "capture_sizes": {
                        "0": [256, 256],
                        "2": [256, 256],
                    },
                },
                "remote_endpoint": {
                    "host": "127.0.0.1",
                    "port": 30001,
                    "listening": self.port_listening,
                },
                "gpu": {
                    "ok": True,
                    "rows": ["0, NVIDIA RTX, 24576, 0, 0"],
                    "error": None,
                },
                "airsim_processes": [],
                "historical_registry_count": 12,
            }
        )


def test_readiness_process_query_does_not_match_its_own_command_line():
    assert '["pgrep", "-af", "[A]irVLN-Linux-Shipping"]' in (
        _REMOTE_SNAPSHOT
    )


def test_readiness_checks_selected_scenes_without_remote_writes(tmp_path):
    config = _config(tmp_path)
    ssh = FakeSshRunner()

    result = inspect_remote_readiness(config, ssh_runner=ssh)

    assert result["ready"] is True
    assert result["errors"] == []
    assert result["case_selection"]["selected_scene_ids"] == [
        "1",
        "13",
        "26",
    ]
    assert len(ssh.commands) == 1
    assert ssh.commands[0][:3] == ["python3", "-c", ssh.commands[0][2]]
    assert not any(
        token in ssh.commands[0]
        for token in ("mkdir", "scp", "start", "stop")
    )


def test_readiness_rejects_existing_remote_process(tmp_path):
    result = inspect_remote_readiness(
        _config(tmp_path),
        ssh_runner=FakeSshRunner(port_listening=True),
    )

    assert result["ready"] is False
    assert result["errors"][0]["stage"] == "existing_process"


def test_readiness_reports_missing_key_without_leaking_path(tmp_path):
    config = _config(tmp_path, key_path=tmp_path / "secret-missing-key")

    result = inspect_remote_readiness(config, ssh_runner=FakeSshRunner())

    assert result["ready"] is False
    serialized = json.dumps(result)
    assert "secret-missing-key" not in serialized
    assert result["errors"][0]["error_type"] == "FileNotFoundError"


def _config(
    tmp_path: Path,
    *,
    key_path: Path | None = None,
) -> AerialVlnEvalConfig:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    episodes = []
    for index, scene_id in enumerate(("1", "13", "26"), start=1):
        episodes.append(
            {
                "episode_id": f"ep_{scene_id}",
                "trajectory_id": f"traj_{scene_id}",
                "scene_id": scene_id,
                "instruction": {"instruction_text": "Hold position"},
                "start_position": [0, 0, 0],
                "start_rotation": [1, 0, 0, 0],
                "goals": [{"position": [index, 0, 0]}],
                "reference_path": [[0, 0, 0], [index, 0, 0]],
                "actions": [0],
            }
        )
    (dataset / "val_seen.json").write_text(
        json.dumps({"episodes": episodes}), encoding="utf-8"
    )
    private_key = key_path or tmp_path / "id_test"
    if key_path is None:
        private_key.write_text("not-a-real-key", encoding="utf-8")
    maps = tmp_path / "maps.json"
    maps.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "map_id": f"remote_{scene_id}",
                        "provider": "packaged_unreal",
                        "scene_id": scene_id,
                        "root": None,
                        "connection": {
                            "mode": "remote",
                            "data_transport": "ssh_tunnel",
                        },
                        "ssh": {
                            "host": "192.0.2.1",
                            "port": 22,
                            "user": "tester",
                            "key_path": str(private_key),
                        },
                        "launch": {
                            "linux_executable": f"/env_{scene_id}/AirVLN.sh",
                            "existing_process_policy": "reject",
                        },
                        "control": {"protocol": "airsim"},
                        "airsim": {
                            "remote_host": "127.0.0.1",
                            "port": 30001,
                            "mode": "ComputerVision",
                            "settings_template_source": "remote",
                            "settings_template": (
                                "/remote/settings/1/settings.json"
                            ),
                            "vehicle_name": "Drone_1",
                            "camera_name": "front_0",
                        },
                    }
                    for scene_id in ("1", "13", "26")
                ]
            }
        ),
        encoding="utf-8",
    )
    return AerialVlnEvalConfig.from_dict(
        {
            "dataset_root": str(dataset),
            "dataset_family": "aerialvln-s",
            "split": "val_seen",
            "map_config_path": str(maps),
            "scene_map": {
                scene_id: f"remote_{scene_id}"
                for scene_id in ("1", "13", "26")
            },
            "execution_mode": "dry",
        }
    )
