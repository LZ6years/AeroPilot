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


def test_dispatch_uses_explicit_backend_binding_not_name_prefix():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
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
    registry.register(ToolSpec("vision.one", "vision_call", "skill_capability", "one", schema, policy))

    with pytest.raises(ValueError, match="Duplicate tool wire_name"):
        registry.register(ToolSpec("vision.two", "vision_call", "skill_capability", "two", schema, policy))


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
