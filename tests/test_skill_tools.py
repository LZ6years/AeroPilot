"""Tests for callable skill capabilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.agent.planning.tool_contract_validator import ToolContractValidationError, ToolContractValidator
from src.env_client.simulator_client import SimulatorEnvClient
from src.env_client.schemas import MapPackage
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.plugins.skills.dispatcher import SkillCapabilityDispatcher
from src.sim_server.server import SimulationServer
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import default_registry
from src.tools.validation import (
    ToolArgumentError,
    validate_runtime_bound_tool_args,
    validate_tool_args,
)
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


def _metric_depth_registry() -> PluginRegistry:
    return PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/capabilities"],
                "enabled": ["capabilities.metric_depth"],
            }
        }
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


def test_metric_depth_capability_has_read_only_small_roi_contract():
    registry = _metric_depth_registry()
    capability = registry.skill_capability_catalog.capabilities[0]

    assert capability.call_name == "skill.metric_depth.sample_region"
    assert capability.wire_name == "skill_metric_depth_sample_region"
    assert capability.runtime.effect == "read_only"
    assert capability.runtime.result_timing == "immediate"
    assert capability.runtime.batching == "parallel_safe"
    provider_parameters = default_registry(_metric_depth_registry()).get(
        capability.call_name
    ).to_provider_tool()["function"]["parameters"]
    assert not any(key.startswith("x-flyagent-") for key in provider_parameters)

    with pytest.raises(ToolArgumentError, match="region span"):
        validate_tool_args(
            {
                "observation_id": "obs_0001",
                "x_min": 0.1,
                "y_min": 0.1,
                "x_max": 0.31,
                "y_max": 0.2,
            },
            capability.input_schema,
        )
    with pytest.raises(ToolArgumentError, match="latest observation_id"):
        validate_runtime_bound_tool_args(
            {
                "observation_id": "obs_old",
                "x_min": 0.1,
                "y_min": 0.1,
                "x_max": 0.2,
                "y_max": 0.2,
            },
            capability.input_schema,
            {"observation_id": "obs_new"},
        )
    with pytest.raises(ToolArgumentError, match="at least 3x3"):
        validate_runtime_bound_tool_args(
            {
                "observation_id": "obs_new",
                "x_min": 0.1,
                "y_min": 0.1,
                "x_max": 0.11,
                "y_max": 0.11,
            },
            capability.input_schema,
            {
                "observation_id": "obs_new",
                "frames": {"depth": {"width": 100, "height": 100}},
            },
        )


def test_metric_depth_samples_robust_median_and_keeps_audit_details(tmp_path):
    depth_path = tmp_path / "depth.npy"
    depth = np.full((100, 100), 2.86, dtype=np.float32)
    depth[63:89, 47:56][0, 0] = 50.0
    depth[63:89, 47:56][0, 1] = np.nan
    np.save(depth_path, depth)
    dispatcher = SkillCapabilityDispatcher(_metric_depth_registry())

    result = dispatcher.call(
        "scene_1",
        "uav_1",
        "skill.metric_depth.sample_region",
        {
            "observation_id": "obs_0001",
            "x_min": 0.47,
            "y_min": 0.45,
            "x_max": 0.55,
            "y_max": 0.62,
        },
        runtime_context={
            "latest_observation": {
                "observation_id": "obs_0001",
                "frames": {
                    "depth": {
                        "id": "frame_0002",
                        "path": str(depth_path),
                        "depth_unit": "meter",
                    }
                },
            }
        },
    )

    assert result.ok is True
    assert result.message.startswith("Approximate camera-to-visible-surface distance")
    assert result.data["observation_id"] == "obs_0001"
    assert result.data["distance_m"] == 2.9
    assert result.data["audit_details"]["frame_id"] == "frame_0002"
    assert result.data["audit_details"]["roi_pixels"] == [47, 63, 56, 89]
    assert result.data["audit_details"]["inlier_pixel_count"] < result.data["audit_details"]["valid_pixel_count"]


def test_metric_depth_rejects_region_crossing_inconsistent_surfaces(tmp_path):
    depth_path = tmp_path / "depth.npy"
    depth = np.full((100, 100), 6.0, dtype=np.float32)
    # The display-normalized ROI below maps to scene pixels [40:60, 42:71].
    # A strong depth gradient indicates that the model did not isolate one
    # representative surface, even though every pixel is numerically valid.
    depth[42:71, 40:60] = np.linspace(4.0, 8.0, 20, dtype=np.float32)[None, :]
    np.save(depth_path, depth)
    dispatcher = SkillCapabilityDispatcher(_metric_depth_registry())

    result = dispatcher.call(
        "scene_1",
        "uav_1",
        "skill.metric_depth.sample_region",
        {
            "observation_id": "obs_0001",
            "x_min": 0.4,
            "y_min": 0.3,
            "x_max": 0.6,
            "y_max": 0.5,
        },
        runtime_context={
            "latest_observation": {
                "observation_id": "obs_0001",
                "frames": {
                    "depth": {
                        "id": "frame_depth",
                        "path": str(depth_path),
                        "depth_unit": "meter",
                    }
                },
            }
        },
    )

    assert result.ok is False
    assert "inconsistent depth surfaces" in result.message


def test_metric_depth_rejects_stale_low_coverage_and_tiny_regions(tmp_path):
    depth_path = tmp_path / "depth.npy"
    depth = np.full((20, 20), np.nan, dtype=np.float32)
    depth[0:2, 0:2] = 2.0
    np.save(depth_path, depth)
    dispatcher = SkillCapabilityDispatcher(_metric_depth_registry())
    runtime_context = {
        "latest_observation": {
            "observation_id": "obs_new",
            "frames": {"depth": {"id": "frame_depth", "path": str(depth_path), "depth_unit": "meter"}},
        }
    }

    stale = dispatcher.call(
        "scene_1",
        "uav_1",
        "skill.metric_depth.sample_region",
        {"observation_id": "obs_old", "x_min": 0, "y_min": 0, "x_max": 0.2, "y_max": 0.2},
        runtime_context=runtime_context,
    )
    tiny = dispatcher.call(
        "scene_1",
        "uav_1",
        "skill.metric_depth.sample_region",
        {"observation_id": "obs_new", "x_min": 0, "y_min": 0, "x_max": 0.05, "y_max": 0.05},
        runtime_context=runtime_context,
    )
    sparse = dispatcher.call(
        "scene_1",
        "uav_1",
        "skill.metric_depth.sample_region",
        {"observation_id": "obs_new", "x_min": 0, "y_min": 0, "x_max": 0.2, "y_max": 0.2},
        runtime_context=runtime_context,
    )
    color_bar = dispatcher.call(
        "scene_1",
        "uav_1",
        "skill.metric_depth.sample_region",
        {"observation_id": "obs_new", "x_min": 0, "y_min": 0.8, "x_max": 0.1, "y_max": 0.9},
        runtime_context=runtime_context,
    )

    assert stale.ok is False and "latest observation_id" in stale.message
    assert tiny.ok is False and "at least 3x3" in tiny.message
    assert sparse.ok is False and "insufficient valid metric depth" in sparse.message
    assert color_bar.ok is False and "color bar" in color_bar.message
