"""Tests for callable skill capabilities."""

from __future__ import annotations

import pytest

from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.agent.planning.tool_contract_validator import ToolContractValidationError, ToolContractValidator
from src.env_client.simulator_client import SimulatorEnvClient
from src.env_client.schemas import MapPackage
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.sim_server.server import SimulationServer
from src.tools.dispatcher import ToolDispatcher
from tests.simulator_fakes import in_memory_backend_registry


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
    assert listings[0]["wire_name"] == "skill_person_detector_detect"
    assert listings[0]["input_schema"]["required"] == ["observation_id"]
    assert listings[0]["runtime"] == {
        "effect": "read_only",
        "result_timing": "immediate",
        "batching": "parallel_safe",
        "model_exposed": True,
    }


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
                {"observation_id": "obs_0001", "min_confidence": 0.5},
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
                {"observation_id": "obs_0001", "extra": True},
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
            "connection": {"mode": "local"},
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )
    client = SimulatorEnvClient(
        SimulationServer(
            [package],
            run_dir=tmp_path,
            backend_registry=in_memory_backend_registry(),
        )
    )
    scene = client.start_scene("m1")
    dispatcher = ToolDispatcher(client, plugin_registry=_capability_registry())

    result = dispatcher.call(
        scene.scene_instance_id,
        "uav_1",
        "skill.person_detector.detect",
        {"observation_id": "obs_0001", "min_confidence": 0.4},
        runtime_context={
            "latest_observation": {
                "observation_id": "obs_0001",
                "frames": {
                    "rgb": {
                        "id": "frame_0001",
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
    assert result.data["observation_id"] == "obs_0001"
    assert result.data["frame_id"] == "frame_0001"
    assert result.data["detections"][0]["class"] == "person"


def test_skill_tool_rejects_a_stale_observation_id(tmp_path):
    client = SimulatorEnvClient(
        SimulationServer(
            [MapPackage.from_dict({
                "map_id": "m1",
                "provider": "airvln",
                "scene_id": 1,
                "root": None,
                "connection": {"mode": "local"},
                "launch": {"manual": True},
                "control": {"protocol": "airsim"},
                "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
            })],
            run_dir=tmp_path,
            backend_registry=in_memory_backend_registry(),
        )
    )
    scene = client.start_scene("m1")
    dispatcher = ToolDispatcher(client, plugin_registry=_capability_registry())

    result = dispatcher.call(
        scene.scene_instance_id,
        "uav_1",
        "skill.person_detector.detect",
        {"observation_id": "obs_old"},
        runtime_context={
            "latest_observation": {
                "observation_id": "obs_new",
                "frames": {"rgb": {"id": "frame_new"}},
            }
        },
    )

    assert result.ok is False
    assert "only accepts the latest observation_id" in result.message
