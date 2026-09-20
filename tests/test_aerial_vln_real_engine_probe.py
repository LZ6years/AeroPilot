"""Pure contract tests for the manual AerialVLN real-engine probe."""

from tests.manual.aerial_vln_real_engine_probe import (
    capability_contract_checks,
)


def test_computer_vision_probe_does_not_require_takeoff():
    assert capability_contract_checks(
        "pose_kinematic",
        {"control_mode": "pose_kinematic", "requires_takeoff": False},
    ) == {
        "control_mode_matches_map_package": True,
        "takeoff_requirement_matches_control_mode": True,
    }


def test_paused_multirotor_probe_does_not_require_takeoff_capability():
    assert capability_contract_checks(
        "paused_multirotor",
        {"control_mode": "paused_multirotor", "requires_takeoff": False},
    ) == {
        "control_mode_matches_map_package": True,
        "takeoff_requirement_matches_control_mode": True,
    }
