"""Dispatch enabled callable skill capabilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from src.env_client.schemas import ActionResult
from src.plugins import PluginRegistry


class SkillCapabilityDispatcher:
    def __init__(self, plugin_registry: PluginRegistry | None = None) -> None:
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()

    def call(
        self,
        scene_instance_id: str,
        agent_id: str,
        tool_name: str,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> ActionResult:
        match = self.plugin_registry.skill_capability_catalog.by_call_name().get(tool_name)
        if match is None:
            return ActionResult(
                ok=False,
                action=tool_name,
                message=f"Skill capability is not enabled: {tool_name}",
                data={"tool": tool_name},
            )
        skill, capability = match
        try:
            module = _load_handler_module(capability.handler_path)
            handler = getattr(module, "handle", None)
            if not callable(handler):
                raise ValueError(f"Skill capability handler must export handle(): {capability.handler_path}")
            payload = {
                "scene_instance_id": scene_instance_id,
                "agent_id": agent_id,
                "tool": tool_name,
                "args": dict(args),
                "runtime_context": dict(runtime_context or {}),
                "skill": {
                    "plugin_id": skill.plugin_id,
                    "name": skill.name,
                    "skill_dir": str(skill.skill_dir),
                },
                "capability": capability.listing(),
            }
            result = handler(payload)
            data = dict(result) if isinstance(result, dict) else {"result": result}
            data.setdefault("mock", capability.mock)
            data.setdefault("skill_name", skill.name)
            data.setdefault("plugin_id", skill.plugin_id)
            return ActionResult(ok=True, action=tool_name, message="skill capability completed", data=data)
        except Exception as exc:
            return ActionResult(
                ok=False,
                action=tool_name,
                message=f"Skill capability failed: {type(exc).__name__}: {exc}",
                data={"tool": tool_name, "error_type": type(exc).__name__, "error": str(exc)},
            )


def _load_handler_module(path: Path):
    module_name = f"_uav_agent_skill_handler_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load skill handler: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
