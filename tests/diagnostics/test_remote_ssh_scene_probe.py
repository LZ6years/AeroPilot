"""Pure contract tests for the manual remote SSH scene probe."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from tests.manual.remote_ssh_scene_probe import (
    _angular_error_deg,
    _local_settings_path,
    _parse_canonical_pose,
    _remote_port_released,
    _remote_settings_argument_snapshot,
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


def test_probe_verifies_single_dash_settings_argument():
    class Runner:
        def __init__(self):
            self.commands = []

        def run(self, target, command):
            self.commands.append(command)
            return json.dumps(
                {
                    "style": "single_dash_equals",
                    "expected_tokens": [
                        "-settings=/remote/scene/settings.json"
                    ],
                    "present": True,
                    "matches": [{"pid": 12, "command": "Unreal -settings=/remote/scene/settings.json"}],
                }
            )

    runner = Runner()
    service = SimpleNamespace(
        remote_scene_control=SimpleNamespace(
            runner=runner,
            target=lambda package: object(),
        )
    )
    package = SimpleNamespace(
        launch={"settings_arg_style": "single_dash_equals"}
    )
    scene = SimpleNamespace(
        settings_path="/remote/scene/settings.json",
        remote_process_id=12,
    )

    result = _remote_settings_argument_snapshot(service, package, scene)

    assert result["present"] is True
    assert runner.commands[0][-2:] == [
        "/remote/scene/settings.json",
        "single_dash_equals",
    ]


def test_probe_verifies_double_dash_settings_argument():
    class Runner:
        def __init__(self):
            self.commands = []

        def run(self, target, command):
            self.commands.append(command)
            return json.dumps(
                {
                    "style": "double_dash",
                    "expected_tokens": [
                        "--settings",
                        "/remote/settings.json",
                    ],
                    "present": True,
                    "matches": [],
                }
            )

    runner = Runner()
    service = SimpleNamespace(
        remote_scene_control=SimpleNamespace(
            runner=runner,
            target=lambda package: object(),
        )
    )
    package = SimpleNamespace(launch={"settings_arg_style": "double_dash"})
    scene = SimpleNamespace(
        settings_path="/remote/settings.json", remote_process_id=1
    )

    result = _remote_settings_argument_snapshot(service, package, scene)

    assert result["present"] is True
    assert runner.commands[0][-1] == "double_dash"


def test_probe_rejects_settings_style_without_rendered_path():
    service = SimpleNamespace(remote_scene_control=SimpleNamespace())
    package = SimpleNamespace(launch={"settings_arg_style": "none"})
    scene = SimpleNamespace(
        settings_path="/remote/settings.json", remote_process_id=1
    )

    with pytest.raises(AssertionError, match="carries the rendered path"):
        _remote_settings_argument_snapshot(service, package, scene)


def test_probe_resolves_repository_local_settings_template():
    resolved = _local_settings_path(
        "configs/airsim/travel_uav_official.settings.example.json"
    )

    assert resolved.is_file()
    assert resolved.name == "travel_uav_official.settings.example.json"


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


def test_probe_yaw_error_wraps_across_180_degrees():
    assert _angular_error_deg(179.5, -179.5) == pytest.approx(1.0)


@pytest.mark.parametrize("value", ["1,2,3", "1,2,3,4,5,nan", "x,2,3,4,5,6"])
def test_probe_rejects_invalid_explicit_pose(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_canonical_pose(value)
