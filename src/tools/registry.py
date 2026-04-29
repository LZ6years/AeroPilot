"""Register the tool names Agent Runtime may call."""

from __future__ import annotations

from src.tools.schemas import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        # Store tool specifications by public tool name.
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        # Add or replace a tool specification.
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        # Look up a tool specification by name.
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def names(self) -> list[str]:
        # Return registered tool names in stable order.
        return sorted(self._tools)


def default_registry() -> ToolRegistry:
    # Build the initial primitive UAV and observation tool registry.
    registry = ToolRegistry()
    for name in (
        "uav.connect",
        "uav.takeoff",
        "uav.hover",
        "uav.land",
        "uav.move_body",
        "uav.turn_relative",
        "uav.ascend",
        "uav.descend",
        "uav.get_pose",
        "uav.get_frames",
    ):
        registry.register(ToolSpec(name=name, kind="uav_primitive", description=f"Primitive {name}"))
    registry.register(
        ToolSpec(
            name="env.observe",
            kind="observation",
            description="Get raw RGB/depth/pose observation through EnvClient",
            required_args=("modalities",),
        )
    )
    return registry
