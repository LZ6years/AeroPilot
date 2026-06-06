"""Tests for callable skill capabilities."""

from __future__ import annotations

import pytest

from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.agent.planning.tool_contract_validator import ToolContractValidationError, ToolContractValidator
from src.env_client.local_client import LocalEnvClient
from src.env_client.schemas import MapPackage
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.sim_server.server import LocalSimulationServer
from src.tools.dispatcher import ToolDispatcher


def _capability_registry() -> PluginRegistry:
    return PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/capabilities"],
                "enabled": ["capabilities.perception"],
            }
        }
    )


def _context_for_skill_tool() -> dict:
    registry = _capability_registry()
    constraints = ObservationContextBuilder(plugin_registry=registry)._allowed_action_constraints()
    return {
        "current_phase": {"name": "confirm_person"},
        "allowed_action_constraints": constraints,
    }


def test_callable_skill_is_discovered_and_exposes_schema():
    registry = _capability_registry()

    listings = registry.skill_capability_catalog.listings()

    assert [item["call_name"] for item in listings] == ["skill.person_detector.detect"]
    assert listings[0]["skill_name"] == "person_detector"
    assert listings[0]["input_schema"]["required"] == ["image_source"]


def test_context_constraints_merge_builtin_tools_and_skill_calls():
    constraints = _context_for_skill_tool()["allowed_action_constraints"]

    assert constraints["planner_tools"][:3] == ["uav.hover", "uav.turn_relative", "uav.move_body"]
    assert "skill.person_detector.detect" in constraints["planner_tools"]
    assert constraints["action_schemas"]["skill.person_detector.detect"]["source"] == "skill_capability"


def test_skill_tool_validator_accepts_schema_valid_args():
    plan = ShortHorizonPlan(
        "confirm_person",
        [
            PlannedStep.tool_step(
                "skill.person_detector.detect",
                {"image_source": "latest_rgb", "min_confidence": 0.5},
                "confirm person visibility",
            )
        ],
    )

    validated = ToolContractValidator().validate(plan, "confirm_person", _context_for_skill_tool())

    assert validated.steps[0].tool == "skill.person_detector.detect"


def test_skill_tool_validator_rejects_schema_invalid_args():
    plan = ShortHorizonPlan(
        "confirm_person",
        [
            PlannedStep.tool_step(
                "skill.person_detector.detect",
                {"image_source": "old_rgb", "extra": True},
                "bad args",
            )
        ],
    )

    with pytest.raises(ToolContractValidationError):
        ToolContractValidator().validate(plan, "confirm_person", _context_for_skill_tool())


def test_skill_tool_dispatcher_runs_mock_handler(tmp_path):
    package = MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "airvln",
            "scene_id": 1,
            "root": None,
            "launch": {"manual": True},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )
    client = LocalEnvClient(LocalSimulationServer([package], run_dir=tmp_path))
    scene = client.start_scene("m1")
    dispatcher = ToolDispatcher(client, plugin_registry=_capability_registry())

    result = dispatcher.call(
        scene.scene_instance_id,
        "uav_1",
        "skill.person_detector.detect",
        {"image_source": "latest_rgb", "min_confidence": 0.4},
        runtime_context={
            "latest_observation": {
                "frames": {
                    "rgb": {
                        "path": str(tmp_path / "frame.png"),
                        "mock": False,
                    }
                }
            }
        },
    )

    assert result.ok
    assert result.action == "skill.person_detector.detect"
    assert result.data["mock"] is True
    assert result.data["detections"][0]["class"] == "person"
