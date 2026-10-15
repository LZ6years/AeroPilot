"""Canonical parameter conversion, registration and visibility contracts."""

import copy
from dataclasses import replace

import pytest

from src.tools.registry import ToolRegistry, default_registry
from src.tools.schemas import ToolParameter, ToolRuntimePolicy, ToolSpec
from src.tools.validation import ToolArgumentError, validate_tool_args
from src.uav.config import UavSafetyConfig


def test_plugin_conversion_preserves_nested_constraints_and_extensions():
    schema = {
        "type": "object", "additionalProperties": False,
        "x-flyagent-completion": {"policy": "current_observation_success"},
        "required": ["regions"],
        "properties": {"regions": {
            "type": "array", "description": "Regions to inspect.", "minItems": 1,
            "items": {"type": "object", "required": ["x"], "additionalProperties": False,
                      "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1,
                                             "x-flyagent-coordinate": "normalized"}}},
        }},
    }
    original = copy.deepcopy(schema)
    spec = ToolSpec.from_input_schema("inspect", "inspect", "test", "Inspect regions.", schema,
                                     ToolRuntimePolicy("read_only", "immediate", "single", True))
    assert spec.input_schema == original
    assert spec.parameters[0].required
    schema["properties"].clear()
    exported = spec.input_schema
    exported["properties"].clear()
    assert spec.input_schema == original
    provider = spec.to_provider_tool()["function"]["parameters"]
    assert "x-flyagent-completion" not in provider
    assert "x-flyagent-coordinate" not in provider["properties"]["regions"]["items"]["properties"]["x"]
    validate_tool_args({"regions": [{"x": 0.5}]}, spec.input_schema)
    with pytest.raises(ToolArgumentError):
        validate_tool_args({"regions": [{"x": 2}]}, spec.input_schema)


@pytest.mark.parametrize("options", [{}, {"additionalProperties": True}, {"additionalProperties": False}])
def test_empty_object_options_keep_their_meaning(options):
    spec = ToolSpec.from_input_schema("empty", "empty", "test", "Empty.",
                                     {"type": "object", **options},
                                     ToolRuntimePolicy("read_only", "immediate", "single"))
    assert spec.input_schema == {"type": "object", "properties": {}, **options}


def test_parameter_examples_are_checked_before_registry_mutation():
    registry = ToolRegistry()
    spec = ToolSpec("test", "test", "test", "Test.",
                    (ToolParameter("count", "integer", "Positive count.", True, {"minimum": 1}),),
                    ToolRuntimePolicy("read_only", "immediate", "single", True),
                    input_examples=({"count": 1},))
    registry.register(spec)
    assert "Example arguments:" in spec.to_provider_tool()["function"]["description"]
    with pytest.raises(ToolArgumentError):
        registry.register(replace(spec, wire_name="replacement", input_examples=({"count": 0},)))
    assert registry.get("test") is spec
    with pytest.raises(KeyError):
        registry.from_wire_name("replacement")


def test_allowlist_cannot_reopen_hidden_runtime_tools():
    registry = default_registry()
    assert registry.get("uav.hover").runtime.effect == "physical"
    assert not registry.get("uav.hover").runtime.model_exposed
    assert registry.model_specs({"uav.hover"}) == []
    assert {s.name for s in registry.model_specs({"uav.hover", "uav.turn_relative"})} == {"uav.turn_relative"}
    assert "uav.hover" not in {s.name for s in registry.model_specs()}


def test_move_body_schema_exposes_runtime_safety_ranges():
    safety = UavSafetyConfig(
        max_linear_velocity_mps=5.5,
        max_vertical_velocity_mps=2.5,
        max_move_duration_sec=7.0,
    )
    spec = default_registry(safety_config=safety).get("uav.move_body")
    properties = spec.input_schema["properties"]
    assert properties["vx"]["minimum"] == -5.5
    assert properties["vx"]["maximum"] == 5.5
    assert properties["vy"]["minimum"] == -5.5
    assert properties["vy"]["maximum"] == 5.5
    assert properties["vz"]["minimum"] == -2.5
    assert properties["vz"]["maximum"] == 2.5
    assert properties["duration"]["maximum"] == 7.0
    assert "sqrt(vx^2 + vy^2) * duration" in spec.description
    assert "abs(vz) * duration" in spec.description
    assert "maximum permissions, not default choices" in spec.description
    assert "ToolResult is authoritative" in spec.description
    validate_tool_args(
        {"vx": 5.5, "vy": -5.5, "vz": 2.5, "duration": 7.0},
        spec.input_schema,
    )
    with pytest.raises(ToolArgumentError):
        validate_tool_args(
            {"vx": 5.6, "vy": 0, "vz": 0, "duration": 1},
            spec.input_schema,
        )


def test_parameter_names_cannot_be_duplicated():
    with pytest.raises(ValueError, match="duplicate"):
        ToolSpec("test", "test", "test", "Test.",
                 (ToolParameter("x"), ToolParameter("x")),
                 ToolRuntimePolicy("read_only", "immediate", "single"))


def test_movement_velocity_parameters_declare_units():
    properties = default_registry().get("uav.move_body").input_schema["properties"]
    for name in ("vx", "vy", "vz"):
        assert "m/s" in properties[name]["description"]


def test_visible_depth_schema_accepts_current_arguments_and_rejects_retired_toggle():
    from src.plugins import PluginRegistry
    spec = default_registry(PluginRegistry.from_config()).get("skill.visual_grounding.measure_visible_depth")
    assert set(spec.input_schema["properties"]) == {"observation_id", "query"}
    args = {"observation_id": "obs_1", "query": "visible roof"}
    validate_tool_args(args, spec.input_schema)
    with pytest.raises(ToolArgumentError):
        validate_tool_args({**args, "include_distance": True}, spec.input_schema)
