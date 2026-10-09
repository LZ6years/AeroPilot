"""Register canonical tools, parameters and explicit Runtime policies."""

from __future__ import annotations

from src.uav.config import UavSafetyConfig
from src.tools.schemas import ToolParameter, ToolRuntimePolicy, ToolSpec
from src.tools.validation import validate_tool_args
from src.uav.commands import MOVE_BODY_ARGUMENT_SEMANTICS
from src.uav.frames import SIGNED_YAW_SEMANTICS


class ToolRegistry:
    """Bidirectional registry; exposure and example validation have one owner."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._wire_names: dict[str, str] = {}

    def register(self, spec: ToolSpec) -> None:
        existing = self._wire_names.get(spec.wire_name)
        if existing is not None and existing != spec.name:
            raise ValueError(f"Duplicate tool wire_name {spec.wire_name!r}: {existing!r} and {spec.name!r}")
        # Validate before mutating either lookup, including when replacing a tool.
        for example in spec.input_examples:
            validate_tool_args(example, spec.input_schema)
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
        return sorted((spec for spec in self._tools.values()
                       if spec.runtime.model_exposed
                       and (allowed_names is None or spec.name in allowed_names)),
                      key=lambda spec: spec.name)


def default_registry(capability_source=None,
                     safety_config: UavSafetyConfig | None = None) -> ToolRegistry:
    """Build built-ins and explicitly injected capability specs.

    Plugin discovery belongs to RuntimeFactory.  This module accepts a
    duck-typed contribution source only so Tools never imports or discovers
    plugins on its own.
    """
    safety = safety_config or UavSafetyConfig()
    registry = ToolRegistry()
    lifecycle = ToolRuntimePolicy("lifecycle", "immediate", "single", False)
    read_internal = ToolRuntimePolicy("read_only", "immediate", "single", False)
    physical = ToolRuntimePolicy("physical", "after_batch_observation", "ordered", True)
    internal_hold = ToolRuntimePolicy("physical", "after_batch_observation", "ordered", False)

    for action, description, policy in (
        ("connect", "Connect the vehicle control interface during Runtime preflight.", lifecycle),
        ("takeoff", "Take off when the scene requires flight initialization.", lifecycle),
        ("land", "Land the vehicle during caller-selected shutdown.", lifecycle),
        ("get_pose", "Read vehicle pose for Runtime observation and motion measurement.", read_internal),
        ("get_frames", "Capture camera frames through the configured simulator backend.", read_internal),
        ("hover", "Hold the vehicle for Runtime stabilization, safety intervention or cleanup.", internal_hold),
        ("ascend", "Ascend using the configured internal vertical-motion primitive.", lifecycle),
        ("descend", "Descend using the configured internal vertical-motion primitive.", lifecycle),
    ):
        registry.register(_primitive(action, description, policy))
    registry.register(_primitive("set_pose", "Set an episode's initial pose before task observations.",
                                 lifecycle, schema_options={"additionalProperties": True}))
    registry.register(_primitive(
        "turn_relative", "Rotate the vehicle by a signed relative yaw angle. Applicable to heading changes "
        "or inspecting a different bearing. Returns execution feedback. Rotation does not translate the "
        "vehicle or establish physical passage or target-relative distance.",
        physical, (ToolParameter("yaw_deg", "number", SIGNED_YAW_SEMANTICS, True,
                                 {"minimum": -abs(safety.max_yaw_step_deg), "maximum": abs(safety.max_yaw_step_deg)}),),
    ))
    registry.register(_primitive(
        "move_body", "Translate the vehicle using body-frame velocity components for a specified duration. "
        "Applicable to changing position, including horizontal or vertical movement. Returns execution "
        "feedback. Approximate requested horizontal distance is sqrt(vx^2 + vy^2) * duration, and approximate "
        "requested vertical distance is abs(vz) * duration. Parameter limits are maximum permissions, not "
        "default choices. Safety intervention may change the applied motion, so ToolResult is authoritative "
        "for actual displacement; vehicle displacement alone does not establish distance to a target.",
        physical, (
            ToolParameter(
                "vx", "number", MOVE_BODY_ARGUMENT_SEMANTICS["vx"], True,
                {
                    "minimum": -abs(safety.max_linear_velocity_mps),
                    "maximum": abs(safety.max_linear_velocity_mps),
                },
            ),
            ToolParameter(
                "vy", "number", MOVE_BODY_ARGUMENT_SEMANTICS["vy"], True,
                {
                    "minimum": -abs(safety.max_linear_velocity_mps),
                    "maximum": abs(safety.max_linear_velocity_mps),
                },
            ),
            ToolParameter(
                "vz", "number", MOVE_BODY_ARGUMENT_SEMANTICS["vz"], True,
                {
                    "minimum": -abs(safety.max_vertical_velocity_mps),
                    "maximum": abs(safety.max_vertical_velocity_mps),
                },
            ),
            ToolParameter(
                "duration",
                "number",
                MOVE_BODY_ARGUMENT_SEMANTICS["duration"],
                True,
                {"exclusiveMinimum": 0, "maximum": safety.max_move_duration_sec},
            ),
        ),
    ))
    registry.register(ToolSpec(
        name="env.observe", wire_name="env_observe", kind="observation",
        description="Capture a new environment observation at Runtime-owned sampling points.",
        parameters=(ToolParameter("modalities", "array", "Requested canonical observation modalities.",
                                  True, {"items": {"type": "string"}}),), runtime=read_internal,
    ))
    from src.tools.builtins import builtin_tools
    for spec in builtin_tools():
        registry.register(spec)

    catalog = getattr(capability_source, "skill_capability_catalog", None)
    capabilities = getattr(catalog, "capabilities", ())
    if catalog is None and capability_source is not None:
        capabilities = capability_source
    for capability in capabilities:
        metadata = {"mock": capability.mock}
        registry.register(ToolSpec.from_input_schema(
            name=capability.call_name, wire_name=capability.wire_name, kind="skill_capability",
            description=capability.description, input_schema=capability.input_schema,
            runtime=capability.runtime, metadata=metadata,
        ))
    return registry


def _primitive(action: str, description: str, policy: ToolRuntimePolicy,
               parameters: tuple[ToolParameter, ...] = (), **kwargs) -> ToolSpec:
    return ToolSpec(name=f"uav.{action}", wire_name=f"uav_{action}", kind="uav_primitive",
                    description=description, parameters=parameters, runtime=policy,
                    metadata={"backend_action": action}, **kwargs)
