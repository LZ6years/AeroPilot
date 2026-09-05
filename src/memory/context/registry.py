"""Context module registry and plugin-aware config loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.memory.context.modules import DEFAULT_MODULES
from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.plugins import PluginRegistry
from src.plugins.context import load_plugin_context_module
from src.plugins.skills.selection import ModelSkillSelector


class ContextModuleConfigError(ValueError):
    """Raised when context module configuration is invalid."""


class ContextModuleRegistry:
    DEFAULT_MODULE_ORDER = ("instruction", "skills", "image", "pose", "action", "map")
    MAX_SEMANTIC_EVIDENCE_ITEMS = 8
    MAX_SEMANTIC_EVIDENCE_CHARS = 160

    def __init__(
        self,
        modules: dict[str, ContextModule],
        disabled: list[str] | None = None,
        module_order: list[str] | None = None,
    ) -> None:
        self.modules = dict(modules)
        self.disabled = list(disabled or [])
        self.module_order = list(module_order or modules.keys())

    @classmethod
    def from_config(
        cls,
        path: str | Path = "configs/plugins.json",
        *,
        plugin_registry: PluginRegistry | None = None,
        skill_selector: ModelSkillSelector | None = None,
    ) -> "ContextModuleRegistry":
        plugin_registry = plugin_registry or PluginRegistry.from_config(path)
        config_path = Path(path)
        if not config_path.exists():
            return cls.default(
                plugin_registry=plugin_registry,
                skill_selector=skill_selector,
            )
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        return cls.from_dict(
            data,
            plugin_registry=plugin_registry,
            skill_selector=skill_selector,
        )

    @classmethod
    def default(
        cls,
        *,
        plugin_registry: PluginRegistry | None = None,
        skill_selector: ModelSkillSelector | None = None,
    ) -> "ContextModuleRegistry":
        plugin_registry = plugin_registry or PluginRegistry.from_config()
        return cls(
            _default_modules(plugin_registry, skill_selector),
            module_order=list(cls.DEFAULT_MODULE_ORDER),
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        plugin_registry: PluginRegistry | None = None,
        skill_selector: ModelSkillSelector | None = None,
    ) -> "ContextModuleRegistry":
        plugin_registry = plugin_registry or PluginRegistry.discover(data)
        module_config = dict(data.get("context_modules", {}))
        if not module_config:
            module_config = {key: "default" for key in cls.DEFAULT_MODULE_ORDER}
        modules: dict[str, ContextModule] = {}
        disabled: list[str] = []
        module_order: list[str] = []
        for key, value in module_config.items():
            if value is False:
                disabled.append(key)
                continue
            module_order.append(key)
            if value == "default":
                module_cls = DEFAULT_MODULES.get(key)
                if module_cls is None:
                    raise ContextModuleConfigError(f"No default context module for slot: {key}")
                modules[key] = _instantiate_default_module(key, plugin_registry, skill_selector)
                continue
            if not isinstance(value, str):
                raise ContextModuleConfigError(f"Invalid module config for {key}: {value!r}")
            contribution = plugin_registry.context_module_for(key, value)
            if contribution is None:
                raise ContextModuleConfigError(f"Unknown plugin context module for {key}: {value}")
            modules[key] = load_plugin_context_module(contribution)
            if modules[key].name != key:
                modules[key].name = key
        return cls(modules, disabled, module_order=module_order)

    def build_fragments(self, sources: MemorySourceBundle, stage_context: StageContextInput):
        fragments = []
        for key in self.module_order:
            module = self.modules.get(key)
            if module is None:
                continue
            fragment = module.build(sources, stage_context)
            if fragment.name != key:
                raise ContextModuleConfigError(f"Module {key} returned fragment {fragment.name}")
            fragments.append(fragment)
        return fragments

    def describe_modules(self) -> dict[str, Any]:
        descriptions: dict[str, Any] = {}
        for key in self.module_order:
            module = self.modules.get(key)
            if module is None:
                continue
            descriptions[key] = module.describe_output()
        return descriptions

    def semantic_evidence_capabilities(self) -> list[str]:
        """Return bounded evidence capabilities declared by enabled modules.

        These strings describe configured evidence sources, not facts observed
        in the current scene. Invalid plugin declarations are ignored so a
        documentation-only capability cannot break task parsing.
        """

        capabilities: list[str] = []
        seen: set[str] = set()
        for key in self.module_order:
            if key in self.disabled:
                continue
            module = self.modules.get(key)
            if module is None:
                continue
            try:
                description = module.describe_output()
            except Exception:
                continue
            if not isinstance(description, dict):
                continue
            declared = description.get("semantic_evidence")
            if not isinstance(declared, list):
                continue
            for item in declared:
                if not isinstance(item, str):
                    continue
                normalized = " ".join(item.split())
                if not normalized:
                    continue
                if len(normalized) > self.MAX_SEMANTIC_EVIDENCE_CHARS:
                    normalized = normalized[: self.MAX_SEMANTIC_EVIDENCE_CHARS - 3].rstrip() + "..."
                if normalized in seen:
                    continue
                seen.add(normalized)
                capabilities.append(normalized)
                if len(capabilities) >= self.MAX_SEMANTIC_EVIDENCE_ITEMS:
                    return capabilities
        return capabilities


def _default_modules(
    plugin_registry: PluginRegistry,
    skill_selector: ModelSkillSelector | None,
) -> dict[str, ContextModule]:
    return {
        key: _instantiate_default_module(key, plugin_registry, skill_selector)
        for key in ContextModuleRegistry.DEFAULT_MODULE_ORDER
    }


def _instantiate_default_module(
    key: str,
    plugin_registry: PluginRegistry,
    skill_selector: ModelSkillSelector | None,
) -> ContextModule:
    module_cls = DEFAULT_MODULES[key]
    if key == "skills":
        return module_cls(
            plugin_registry=plugin_registry,
            skill_selector=skill_selector or ModelSkillSelector(enabled=False),
        )
    return module_cls()
