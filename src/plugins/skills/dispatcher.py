"""Dispatch enabled callable skill capabilities."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

from src.env_client.schemas import ActionResult
from src.plugins import PluginRegistry
from src.plugins.contracts.skill import (
    SKILL_ARTIFACT_REFS_KEY,
    SKILL_MODEL_INPUTS_KEY,
    SkillCapabilityFailure,
    SkillCapabilityResult,
)


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
                # Runtime settings are deliberately excluded from public skill
                # listings and provider schemas. Only the selected handler sees
                # the capability-local JSON configuration.
                "settings": copy.deepcopy(capability.settings),
            }
            result = handler(payload)
            if isinstance(result, SkillCapabilityResult):
                message = result.message
                data = dict(result.data)
                artifact_refs = result.artifact_refs
                model_inputs = result.model_inputs
            else:
                message = "skill capability completed"
                data = dict(result) if isinstance(result, dict) else {"result": result}
                data.setdefault("mock", capability.mock)
                data.setdefault("skill_name", skill.name)
                data.setdefault("plugin_id", skill.plugin_id)
                artifact_refs = ()
                model_inputs = ()
            if artifact_refs:
                data[SKILL_ARTIFACT_REFS_KEY] = list(artifact_refs)
            if model_inputs:
                data[SKILL_MODEL_INPUTS_KEY] = [dict(item) for item in model_inputs]
            return ActionResult(
                ok=True,
                action=tool_name,
                message=message,
                data=data,
            )
        except SkillCapabilityFailure as exc:
            data = {
                "tool": tool_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **exc.data,
            }
            if exc.artifact_refs:
                data[SKILL_ARTIFACT_REFS_KEY] = list(exc.artifact_refs)
            if exc.model_inputs:
                data[SKILL_MODEL_INPUTS_KEY] = [
                    dict(item) for item in exc.model_inputs
                ]
            return ActionResult(
                ok=False,
                action=tool_name,
                message=f"Skill capability failed: {exc}",
                data=data,
            )
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
