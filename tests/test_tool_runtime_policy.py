"""Tests for explicit ToolSpec execution policy and wire-name contracts."""

from __future__ import annotations

import pytest

from src.env_client.schemas import ActionResult
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import ToolRegistry
from src.tools.schemas import ToolRuntimePolicy, ToolSpec


class RecordingEnvClient:
    def __init__(self) -> None:
        self.calls = []

    def execute_action(self, scene_instance_id, agent_id, action, args):
        self.calls.append((scene_instance_id, agent_id, action, args))
        return ActionResult(ok=True, action=action, message="done")


def test_provider_projection_exposes_execution_policy_not_private_metadata():
    spec = ToolSpec.from_input_schema("custom.inspect", "custom_inspect", "skill_capability", "Inspect.",
                    {"type": "object"},
                    ToolRuntimePolicy("read_only", "immediate", "parallel_safe", True, 2),
                    metadata={"settings": {"model": "PRIVATE_MODEL"}})
    projected = spec.to_provider_tool()
    description = projected["function"]["description"]
    assert "Independent read-only calls" in description
    assert "sequential" in description
    assert "At most 2 calls per observation" in description
    assert "PRIVATE_MODEL" not in str(projected)
    assert spec.description == "Inspect."


def test_grounding_config_permits_independent_calls_without_changing_quota():
    from src.tools.registry import default_registry
    spec = default_registry().get("skill.visual_grounding.locate")
    assert spec.runtime.batching == "parallel_safe"
    assert spec.runtime.max_calls_per_observation == 2
    assert "candidate locations or target-distance measurements" in spec.description
    blocked = default_registry().get("runtime.report_blocked").to_provider_tool()
    assert "only call in its batch" in blocked["function"]["description"]


@pytest.mark.parametrize("args,path", [({"nested": {"yaw": 91}}, "args.nested.yaw"), ({"nested": {"yaw": "bad"}}, "args.nested.yaw")])
def test_validation_error_carries_deepest_path_without_changing_message(args, path):
    from src.tools.validation import ToolArgumentError, validate_tool_args
    schema = {"type": "object", "properties": {"nested": {"type": "object", "properties": {"yaw": {"type": "number", "maximum": 90}}}}}
    with pytest.raises(ToolArgumentError) as error:
        validate_tool_args(args, schema)
    assert error.value.path == path
    assert error.value.code == "invalid_arguments"
    assert str(error.value).startswith(path)


def test_dispatch_uses_explicit_backend_binding_not_name_prefix():
    registry = ToolRegistry()
    registry.register(
        ToolSpec.from_input_schema(
            name="camera.steady",
            wire_name="camera_steady",
            kind="uav_primitive",
            description="Hold the platform steady.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            runtime=ToolRuntimePolicy(
                effect="physical",
                result_timing="after_batch_observation",
                batching="ordered",
                model_exposed=True,
            ),
            metadata={"backend_action": "hover"},
        )
    )
    client = RecordingEnvClient()
    dispatcher = ToolDispatcher(client, registry=registry)

    result = dispatcher.call("scene", "uav", "camera.steady", {})

    assert result.ok
    assert client.calls == [("scene", "uav", "hover", {})]


def test_registry_rejects_duplicate_provider_wire_name():
    registry = ToolRegistry()
    policy = ToolRuntimePolicy("read_only", "immediate", "single", True)
    schema = {"type": "object", "properties": {}}
    registry.register(ToolSpec.from_input_schema("vision.one", "vision_call", "skill_capability", "one", schema, policy))

    with pytest.raises(ValueError, match="Duplicate tool wire_name"):
        registry.register(ToolSpec.from_input_schema("vision.two", "vision_call", "skill_capability", "two", schema, policy))


@pytest.mark.parametrize(
    "data,missing",
    [
        ({}, "effect"),
        (
            {
                "effect": "read_only",
                "result_timing": "immediate",
                "batching": "single",
            },
            "model_exposed",
        ),
    ],
)
def test_runtime_policy_requires_all_execution_fields(data, missing):
    with pytest.raises(ValueError, match=missing):
        ToolRuntimePolicy.from_dict(data)


def test_runtime_policy_round_trips_optional_observation_quota():
    policy = ToolRuntimePolicy.from_dict(
        {
            "effect": "read_only",
            "result_timing": "immediate",
            "batching": "single",
            "model_exposed": True,
            "max_calls_per_observation": 2,
        }
    )

    assert policy.max_calls_per_observation == 2
    assert policy.to_dict()["max_calls_per_observation"] == 2


@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_runtime_policy_rejects_invalid_observation_quota(value):
    with pytest.raises(ValueError, match="max_calls_per_observation"):
        ToolRuntimePolicy.from_dict(
            {
                "effect": "read_only",
                "result_timing": "immediate",
                "batching": "single",
                "model_exposed": True,
                "max_calls_per_observation": value,
            }
        )
