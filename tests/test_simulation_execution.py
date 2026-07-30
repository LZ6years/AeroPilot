"""Test the two real execution modes and their safe default."""

from __future__ import annotations

import pytest

from src.env_client.schemas import MapPackage
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.server import SimulationServer


def remote_package_with_remote_settings() -> MapPackage:
    """Build a complete remote package for execution-mode validation."""

    return MapPackage.from_dict(
        {
            "map_id": "remote_air",
            "provider": "packaged_unreal",
            "scene_id": "1",
            "root": None,
            "connection": {
                "mode": "remote",
                "data_transport": "ssh_tunnel",
            },
            "ssh": {
                "host": "192.0.2.10",
                "user": "uav",
                "key_path": "missing-key",
            },
            "launch": {
                "method": "linux_bash",
                "linux_executable": "/remote/Environment.sh",
            },
            "control": {"protocol": "airsim"},
            "airsim": {
                "remote_host": "127.0.0.1",
                "port": 41451,
                "mode": "ComputerVision",
                "settings_template_source": "remote",
                "settings_template": "/remote/settings.json",
                "vehicle_name": "Drone_1",
                "camera_name": "front_0",
            },
        }
    )


def test_execution_mode_parses_strings_and_properties():
    assert SimulationExecutionMode.parse("attach") is SimulationExecutionMode.ATTACH
    assert SimulationExecutionMode.parse("managed") is SimulationExecutionMode.MANAGED
    assert not SimulationExecutionMode.ATTACH.may_start_process
    assert SimulationExecutionMode.MANAGED.may_start_process


@pytest.mark.parametrize("value", ["dry-run", "dry_run", "launch-and-maybe-connect"])
def test_execution_mode_rejects_removed_or_unknown_value(value):
    with pytest.raises(ValueError, match="available: attach, managed"):
        SimulationExecutionMode.parse(value)


def test_server_defaults_to_managed(tmp_path):
    server = SimulationServer(
        [remote_package_with_remote_settings()],
        run_dir=tmp_path,
    )

    assert server.execution_mode is SimulationExecutionMode.MANAGED


def test_old_boolean_constructor_parameters_are_removed(tmp_path):
    removed_parameter = "launch_" + "processes"
    with pytest.raises(TypeError, match=removed_parameter):
        SimulationServer(  # type: ignore[call-arg]
            [remote_package_with_remote_settings()],
            run_dir=tmp_path,
            **{removed_parameter: True},
        )
