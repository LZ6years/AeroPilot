"""Pure contract tests for the manual remote SSH scene probe."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from tests.manual.remote_ssh_scene_probe import (
    _parse_canonical_pose,
    _remote_port_released,
    _validate_dimensions,
)


def test_probe_dimensions_follow_remote_settings(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "Vehicles": {
                    "Drone_1": {
                        "Cameras": {
                            "front_0": {
                                "CaptureSettings": [
                                    {
                                        "ImageType": 0,
                                        "Width": 224,
                                        "Height": 224,
                                    },
                                    {
                                        "ImageType": 2,
                                        "Width": 256,
                                        "Height": 256,
                                    },
                                ]
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = _validate_dimensions(
        {
            "rgb": {"width": 224, "height": 224},
            "depth": {"width": 256, "height": 256},
        },
        settings,
        vehicle_name="Drone_1",
        camera_name="front_0",
    )

    assert result == {"rgb": [224, 224], "depth": [256, 256]}


def test_probe_dimensions_reject_mismatched_rgb(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "Vehicles": {
                    "Drone_1": {
                        "Cameras": {
                            "front_0": {
                                "CaptureSettings": [
                                    {
                                        "ImageType": 0,
                                        "Width": 224,
                                        "Height": 224,
                                    },
                                    {
                                        "ImageType": 2,
                                        "Width": 256,
                                        "Height": 256,
                                    },
                                ]
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="RGB dimensions"):
        _validate_dimensions(
            {
                "rgb": {"width": 256, "height": 256},
                "depth": {"width": 256, "height": 256},
            },
            settings,
            vehicle_name="Drone_1",
            camera_name="front_0",
        )


def test_probe_verifies_remote_port_release_without_worker_upload():
    class Runner:
        def __init__(self):
            self.commands = []

        def run(self, target, command):
            self.commands.append(command)
            return '{"listening": false}\n'

    runner = Runner()
    remote = SimpleNamespace(
        runner=runner,
        target=lambda package: object(),
    )
    service = SimpleNamespace(remote_scene_control=remote)
    package = SimpleNamespace(
        airsim={"remote_host": "127.0.0.1", "port": 30001}
    )

    assert _remote_port_released(service, package) is True
    assert runner.commands[0][:2] == ["python3", "-c"]


def test_probe_parses_explicit_canonical_pose():
    assert _parse_canonical_pose("1,-2,3,45,-5,2") == {
        "x": 1.0,
        "y": -2.0,
        "z": 3.0,
        "yaw_deg": 45.0,
        "pitch_deg": -5.0,
        "roll_deg": 2.0,
        "ignore_collision": True,
    }


@pytest.mark.parametrize("value", ["1,2,3", "1,2,3,4,5,nan", "x,2,3,4,5,6"])
def test_probe_rejects_invalid_explicit_pose(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_canonical_pose(value)
