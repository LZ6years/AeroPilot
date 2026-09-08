"""Register canonical tools and their explicit Runtime policies."""

from __future__ import annotations

from src.plugins import PluginRegistry
from src.runtime.configs.safety_config import UavSafetyConfig
from src.tools.schemas import ToolRuntimePolicy, ToolSpec
from src.uav.commands import MOVE_BODY_ARGUMENT_SEMANTICS
from src.uav.frames import SIGNED_YAW_SEMANTICS


_EMPTY_OBJECT_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
_ANY_OBJECT_SCHEMA = {"type": "object"}


class ToolRegistry:
    """Bidirectional registry for canonical and provider-safe tool names."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._wire_names: dict[str, str] = {}

    def register(self, spec: ToolSpec) -> None:
        existing = self._wire_names.get(spec.wire_name)
        if existing is not None and existing != spec.name:
            raise ValueError(
                f"Duplicate tool wire_name {spec.wire_name!r}: {existing!r} and {spec.name!r}"
            )
        old = self._tools.get(spec.name)
        if old is not None and old.wire_name != spec.wire_name:
            self._wire_names.pop(old.wire_name, None)
        self._tools[spec.name] = spec
        self._wire_names[spec.wire_name] = spec.name

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def from_wire_name(self, wire_name: str) -> ToolSpec:
        try:
            return self.get(self._wire_names[wire_name])
        except KeyError as exc:
            raise KeyError(f"Unknown provider tool: {wire_name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def model_specs(self, allowed_names: set[str] | None = None) -> list[ToolSpec]:
        specs = list(self._tools.values())
        if allowed_names is None:
            specs = [spec for spec in specs if spec.runtime.model_exposed]
        else:
            specs = [spec for spec in specs if spec.name in allowed_names]
        return sorted(specs, key=lambda spec: spec.name)


def default_registry(
    plugin_registry: PluginRegistry | None = None,
    safety_config: UavSafetyConfig | None = None,
) -> ToolRegistry:
    """Build built-in, plugin, and core orchestration tool contracts."""

    safety = safety_config or UavSafetyConfig()
    registry = ToolRegistry()

    lifecycle = ToolRuntimePolicy("lifecycle", "immediate", "single", False)
    read_only_internal = ToolRuntimePolicy("read_only", "immediate", "single", False)
    physical = ToolRuntimePolicy("physical", "after_batch_observation", "ordered", True)

    for name, backend_action in (
        ("uav.connect", "connect"),
        ("uav.takeoff", "takeoff"),
        ("uav.land", "land"),
    ):
        registry.register(
            _spec(name, "uav_primitive", lifecycle, _EMPTY_OBJECT_SCHEMA, backend_action)
        )
    registry.register(
        _spec("uav.set_pose", "uav_primitive", lifecycle, _ANY_OBJECT_SCHEMA, "set_pose")
    )
    for name, backend_action in (
        ("uav.get_pose", "get_pose"),
        ("uav.get_frames", "get_frames"),
    ):
        registry.register(
            _spec(name, "uav_primitive", read_only_internal, _EMPTY_OBJECT_SCHEMA, backend_action)
        )

    registry.register(
        _spec("uav.hover", "uav_primitive", physical, _EMPTY_OBJECT_SCHEMA, "hover")
    )
    registry.register(
        _spec(
            "uav.turn_relative",
            "uav_primitive",
            physical,
            {
                "type": "object",
                "required": ["yaw_deg"],
                "additionalProperties": False,
                "properties": {
                    "yaw_deg": {
                        "type": "number",
                        "minimum": -abs(safety.max_yaw_step_deg),
                        "maximum": abs(safety.max_yaw_step_deg),
                        "description": SIGNED_YAW_SEMANTICS,
                    }
                },
            },
            "turn_relative",
        )
    )
    registry.register(
        _spec(
            "uav.move_body",
            "uav_primitive",
            physical,
            {
                "type": "object",
                "required": ["vx", "vy", "vz", "duration"],
                "additionalProperties": False,
                "properties": {
                    "vx": {"type": "number", "description": MOVE_BODY_ARGUMENT_SEMANTICS["vx"]},
                    "vy": {"type": "number", "description": MOVE_BODY_ARGUMENT_SEMANTICS["vy"]},
                    "vz": {"type": "number", "description": MOVE_BODY_ARGUMENT_SEMANTICS["vz"]},
                    "duration": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": safety.max_move_duration_sec,
                    },
                },
            },
            "move_body",
        )
    )

    # These convenience primitives remain Runtime-only until the canonical
    # planner contract explicitly supports them.
    for name, backend_action in (("uav.ascend", "ascend"), ("uav.descend", "descend")):
        registry.register(
            _spec(name, "uav_primitive", lifecycle, _EMPTY_OBJECT_SCHEMA, backend_action)
        )
    registry.register(
        ToolSpec(
            name="env.observe",
            wire_name="env_observe",
            kind="observation",
            description="Get a new environment observation. Runtime owns normal observation timing.",
            input_schema={
                "type": "object",
                "required": ["modalities"],
                "properties": {
                    "modalities": {"type": "array", "items": {"type": "string"}}
                },
                "additionalProperties": False,
            },
            runtime=read_only_internal,
        )
    )

    registry.register(
        ToolSpec(
            name="runtime.commit_phase_turn",
            wire_name="runtime_commit_phase_turn",
            kind="runtime_control",
            description=(
                "Commit the current phase assessment and, when incomplete, describe the physical "
                "action window emitted in the same assistant turn."
            ),
            input_schema=_commit_phase_schema(),
            runtime=ToolRuntimePolicy(
                "runtime_control", "after_batch_observation", "sidecar", True
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="runtime.ask_user",
            wire_name="runtime_ask_user",
            kind="runtime_control",
            description="Request user clarification before safe progress can continue.",
            input_schema={
                "type": "object",
                "required": ["question"],
                "properties": {"question": {"type": "string"}, "context": {"type": "string"}},
                "additionalProperties": False,
            },
            runtime=ToolRuntimePolicy("runtime_control", "user_reply", "single", False),
            metadata={"runtime_action": "ask_user"},
        )
    )
    registry.register(
        ToolSpec(
            name="runtime.report_blocked",
            wire_name="runtime_report_blocked",
            kind="runtime_control",
            description=(
                "Stop the current instruction as incomplete when available evidence cannot "
                "support another safe, meaningful action. Cite only prior transcript evidence."
            ),
            input_schema={
                "type": "object",
                "required": ["blocking_condition", "reason", "evidence_refs"],
                "properties": {
                    "blocking_condition": {
                        "type": "string",
                        "enum": [
                            "lost_required_reference",
                            "insufficient_evidence",
                            "unsafe_to_continue",
                            "missing_capability",
                            "irreducible_ambiguity",
                            "no_progress",
                        ],
                    },
                    "reason": {"type": "string", "minLength": 1},
                    "evidence_refs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "suggested_user_action": {"type": "string"},
                },
                "additionalProperties": False,
            },
            runtime=ToolRuntimePolicy(
                "runtime_control", "after_batch_observation", "single", True
            ),
            metadata={"runtime_action": "report_blocked", "core_runtime_control": True},
        )
    )
    registry.register(
        ToolSpec(
            name="runtime.semantic_plan_update",
            wire_name="runtime_semantic_plan_update",
            kind="runtime_control",
            description="Request an asynchronous review of the active semantic phase contract.",
            input_schema={
                "type": "object",
                "required": ["reason"],
                "properties": {
                    "reason": {"type": "string"},
                    "focus": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            runtime=ToolRuntimePolicy("runtime_control", "async", "sidecar", False),
            metadata={"runtime_action": "semantic_plan_update"},
        )
    )

    plugins = plugin_registry or PluginRegistry.from_config()
    for capability in plugins.skill_capability_catalog.capabilities:
        completion_evidence = capability.input_schema.get(
            "x-flyagent-completion-evidence"
        )
        metadata = {"mock": capability.mock}
        if isinstance(completion_evidence, dict):
            metadata["completion_evidence"] = dict(completion_evidence)
        registry.register(
            ToolSpec(
                name=capability.call_name,
                wire_name=capability.wire_name,
                kind="skill_capability",
                description=capability.description,
                input_schema=dict(capability.input_schema),
                runtime=capability.runtime,
                metadata=metadata,
            )
        )
    return registry


def _spec(
    name: str,
    kind: str,
    runtime: ToolRuntimePolicy,
    input_schema: dict,
    backend_action: str,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        wire_name=name.replace(".", "_"),
        kind=kind,
        description=f"FlyAgent tool {name}",
        input_schema=dict(input_schema),
        runtime=runtime,
        metadata={"backend_action": backend_action},
    )


def _commit_phase_schema() -> dict:
    assessment = {
        "type": "object",
        "required": ["done_when"],
        "properties": {
            "done_when": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["index", "status", "evidence"],
                    "properties": {
                        "index": {"type": "integer", "minimum": 0},
                        "status": {"type": "string", "enum": ["supported", "unknown", "contradicted"]},
                        "evidence": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "required": ["phase", "completion_assessment"],
        "properties": {
            "phase": {"type": "string"},
            "completion_assessment": assessment,
            "phase_strategy": {
                "type": "object",
                "required": ["stages", "active_stage", "reason"],
                "properties": {
                    "stages": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string"}},
                    "active_stage": {"type": "integer", "minimum": 0},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "action_window": {
                "type": "object",
                "required": ["intent", "expected_effect"],
                "properties": {
                    "intent": {"type": "string"},
                    "expected_effect": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
