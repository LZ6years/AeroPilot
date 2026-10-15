"""Canonical tool contracts used by planners, providers, and Runtime.

Tool names are identifiers only. Runtime behavior is declared explicitly by
``ToolRuntimePolicy`` so no execution layer needs to infer semantics from a
``uav.`` or ``skill.`` prefix.
"""

from __future__ import annotations

import copy
import json
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
class ReadExecutionHints:
    """Private dispatch hints; never projected into provider tool declarations."""

    category: Literal["light", "capability"] = "light"
    capability_key: str | None = None
    serial_key: str | None = None

    def __post_init__(self):
        if self.category not in {"light", "capability"}:
            raise ValueError("unsupported read execution category")
        if self.category == "capability" and not self.capability_key:
            raise ValueError("capability execution requires an identifier")


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
class ToolParameter:
    """One named argument; constraints use JSON Schema, including nested items.

    Nested object properties and array items remain standard JSON Schema so
    plugin schemas and their unknown extension keywords survive conversion.
    ``required`` belongs to the containing object, not the property schema.
    """

    name: str
    type: str | list[str] | None = None
    description: str | None = None
    required: bool = False
    constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool parameter name must not be empty")
        if {"type", "description"} & self.constraints.keys():
            raise ValueError("parameter type and description must use their dedicated fields")
        object.__setattr__(self, "constraints", copy.deepcopy(self.constraints))
        object.__setattr__(self, "type", copy.deepcopy(self.type))

    def to_schema(self) -> dict[str, Any]:
        schema = copy.deepcopy(self.constraints)
        if self.type is not None:
            schema["type"] = copy.deepcopy(self.type)
        if self.description is not None:
            schema["description"] = self.description
        return schema

    @classmethod
    def from_schema(cls, name: str, schema: dict[str, Any], *, required: bool = False) -> "ToolParameter":
        data = copy.deepcopy(schema)
        return cls(name, data.pop("type", None), data.pop("description", None), required, data)


@dataclass(frozen=True)
class ToolSpec:
    """One canonical tool plus its provider-facing function declaration."""

    name: str
    wire_name: str
    kind: str
    description: str
    parameters: tuple[ToolParameter, ...]
    runtime: ToolRuntimePolicy
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_options: dict[str, Any] = field(default_factory=lambda: {"additionalProperties": False})
    input_examples: tuple[dict[str, Any], ...] = ()
    strict: bool | None = None

    def __post_init__(self) -> None:
        if self.strict is not None and not isinstance(self.strict, bool):
            raise ValueError("tool strict must be a boolean or None")
        if any(not isinstance(item, ToolParameter) for item in self.parameters):
            raise TypeError("parameters must contain ToolParameter instances")
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tool parameter name")
        if {"properties", "required"} & self.schema_options.keys():
            raise ValueError("properties and required are generated from parameters")
        object.__setattr__(self, "parameters", tuple(copy.deepcopy(self.parameters)))
        object.__setattr__(self, "schema_options", copy.deepcopy(self.schema_options))
        object.__setattr__(self, "input_examples", tuple(copy.deepcopy(self.input_examples)))

    @property
    def input_schema(self) -> dict[str, Any]:
        """Generate an isolated schema shared by Provider export and validation."""
        schema = {"type": "object", **copy.deepcopy(self.schema_options)}
        schema["properties"] = {item.name: item.to_schema() for item in self.parameters}
        required = [item.name for item in self.parameters if item.required]
        if required:
            schema["required"] = required
        return schema

    @classmethod
    def from_input_schema(cls, name: str, wire_name: str, kind: str, description: str,
                          input_schema: dict[str, Any], runtime: ToolRuntimePolicy,
                          **kwargs: Any) -> "ToolSpec":
        """Adapt plugin JSON at registration; never retain a second parameter schema."""
        options = copy.deepcopy(input_schema)
        properties = options.pop("properties", {})
        required = options.pop("required", [])
        if options.get("type", "object") != "object":
            raise ValueError("tool input schema must describe an object")
        if not set(required) <= properties.keys():
            raise ValueError("required tool parameters must have property definitions")
        # Preserve required ordering for plugin/public listing stability.
        names = [*required, *(name for name in properties if name not in required)]
        parameters = tuple(ToolParameter.from_schema(name, properties[name], required=name in required)
                           for name in names)
        return cls(name, wire_name, kind, description, parameters, runtime,
                   schema_options=options, **kwargs)

    def to_provider_tool(self) -> dict[str, Any]:
        description = self.description + "\nExecution: " + self._execution_description()
        if self.input_examples:
            description += "\nExample arguments: " + json.dumps(self.input_examples, ensure_ascii=False)
        result = {
            "type": "function",
            "function": {
                "name": self.wire_name,
                "description": description,
                "parameters": _provider_schema(self.input_schema),
            },
        }
        if self.strict is not None:
            result["function"]["strict"] = self.strict
        return result

    def _execution_description(self) -> str:
        """Project public execution rules, never private plugin metadata.

        parallel_safe permits independent calls in one turn; it does not promise
        concurrent backend execution. Configuration remains the single source.
        Cross-tool call and dependency rules are explained once by ToolUser.
        """
        batching = {
            "single": ("At most one instance of this control tool; follow its description for associated calls."
                       if self.runtime.effect == "runtime_control" and self.runtime.result_timing != "immediate"
                       else "This must be the only call in its batch."),
            "ordered": "Execute in the submitted physical batch order.",
            "parallel_safe": "Independent read-only calls may share a turn; execution may be sequential.",
            "sidecar": "May be submitted alone or with independent parallel-safe read-only calls as described by this tool.",
        }
        text = f"effect={self.runtime.effect}; result_timing={self.runtime.result_timing}. {batching[self.runtime.batching]}"
        if self.runtime.max_calls_per_observation is not None:
            text += f" At most {self.runtime.max_calls_per_observation} calls per observation, including same-turn calls."
        return text


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
