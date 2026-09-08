"""Canonical tool contracts used by planners, providers, and Runtime.

Tool names are identifiers only. Runtime behavior is declared explicitly by
``ToolRuntimePolicy`` so no execution layer needs to infer semantics from a
``uav.`` or ``skill.`` prefix.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal


ToolEffect = Literal["read_only", "physical", "runtime_control", "lifecycle"]
ToolResultTiming = Literal[
    "immediate",
    "after_batch_observation",
    "async",
    "user_reply",
]
ToolBatching = Literal["single", "ordered", "parallel_safe", "sidecar"]


@dataclass(frozen=True)
class ToolRuntimePolicy:
    """Describe model exposure, side effects, feedback, and batching."""

    effect: ToolEffect
    result_timing: ToolResultTiming
    batching: ToolBatching
    model_exposed: bool = False
    max_calls_per_observation: int | None = None

    def __post_init__(self) -> None:
        if self.effect not in {"read_only", "physical", "runtime_control", "lifecycle"}:
            raise ValueError(f"unsupported tool effect: {self.effect}")
        if self.result_timing not in {
            "immediate",
            "after_batch_observation",
            "async",
            "user_reply",
        }:
            raise ValueError(f"unsupported tool result_timing: {self.result_timing}")
        if self.batching not in {"single", "ordered", "parallel_safe", "sidecar"}:
            raise ValueError(f"unsupported tool batching policy: {self.batching}")
        if self.max_calls_per_observation is not None and self.max_calls_per_observation <= 0:
            raise ValueError("max_calls_per_observation must be positive when provided")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolRuntimePolicy":
        if not isinstance(data, dict):
            raise ValueError("tool runtime policy must be an object")
        required = {"effect", "result_timing", "batching", "model_exposed"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"tool runtime policy missing fields: {', '.join(missing)}")
        if not isinstance(data["model_exposed"], bool):
            raise ValueError("tool runtime policy model_exposed must be a boolean")
        max_calls = data.get("max_calls_per_observation")
        if max_calls is not None and (
            not isinstance(max_calls, int) or isinstance(max_calls, bool)
        ):
            raise ValueError(
                "tool runtime policy max_calls_per_observation must be an integer"
            )
        return cls(
            effect=str(data["effect"]),
            result_timing=str(data["result_timing"]),
            batching=str(data["batching"]),
            model_exposed=data["model_exposed"],
            max_calls_per_observation=(
                max_calls
                if max_calls is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "effect": self.effect,
            "result_timing": self.result_timing,
            "batching": self.batching,
            "model_exposed": self.model_exposed,
        }
        if self.max_calls_per_observation is not None:
            result["max_calls_per_observation"] = self.max_calls_per_observation
        return result


@dataclass(frozen=True)
class ToolSpec:
    """One canonical tool plus its provider-facing function declaration."""

    name: str
    wire_name: str
    kind: str
    description: str
    input_schema: dict[str, Any]
    runtime: ToolRuntimePolicy
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_provider_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.wire_name,
                "description": self.description,
                "parameters": _provider_schema(self.input_schema),
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """Canonical model-requested call, paired by ``tool_call_id``."""

    tool_call_id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


def _provider_schema(value: Any) -> Any:
    """Remove Runtime-only JSON-Schema extensions from provider contracts."""

    if isinstance(value, dict):
        return {
            key: _provider_schema(item)
            for key, item in value.items()
            if not str(key).startswith("x-flyagent-")
        }
    if isinstance(value, list):
        return [_provider_schema(item) for item in value]
    return copy.deepcopy(value)
