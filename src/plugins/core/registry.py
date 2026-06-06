"""Registry for enabled plugin contributions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.plugins.contracts.context import ContextModuleContribution
from src.plugins.core.discovery import DEFAULT_PLUGIN_ROOTS, PluginDiscovery
from src.plugins.skills.catalog import SkillCatalog, SkillCapabilityCatalog


DEFAULT_PLUGINS_CONFIG_PATH = Path("configs") / "plugins.json"


@dataclass(frozen=True)
class PluginRegistry:
    skill_catalog: SkillCatalog = field(default_factory=lambda: SkillCatalog([]))
    skill_capability_catalog: SkillCapabilityCatalog = field(default_factory=lambda: SkillCapabilityCatalog([]))
    context_modules: list[ContextModuleContribution] = field(default_factory=list)
    context_module_config: dict[str, str] = field(default_factory=dict)
    skills_config: dict[str, Any] = field(default_factory=dict)
    enabled_plugins: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_PLUGINS_CONFIG_PATH) -> "PluginRegistry":
        config_path = Path(path)
        if not config_path.exists():
            return cls.discover({})
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        return cls.discover(data)

    @classmethod
    def discover(cls, data: dict[str, Any]) -> "PluginRegistry":
        plugin_config = dict(data.get("plugins", {})) if isinstance(data.get("plugins", {}), dict) else {}
        roots = plugin_config.get("roots", [str(path) for path in DEFAULT_PLUGIN_ROOTS])
        enabled = [str(item) for item in plugin_config.get("enabled", [])]
        disabled = [str(item) for item in plugin_config.get("disabled", [])]
        result = PluginDiscovery(roots, enabled=enabled, disabled=disabled).discover()
        return cls(
            skill_catalog=SkillCatalog(result.contributions.skills),
            skill_capability_catalog=SkillCapabilityCatalog(result.contributions.skills),
            context_modules=result.contributions.context_modules,
            context_module_config={str(key): str(value) for key, value in dict(data.get("context_modules", {})).items()},
            skills_config=dict(data.get("skills", {})) if isinstance(data.get("skills", {}), dict) else {},
            enabled_plugins=enabled,
            warnings=list(result.warnings),
        )

    def context_module_for(self, slot: str, provider: str) -> ContextModuleContribution | None:
        if provider == "default":
            return None
        for contribution in self.context_modules:
            if contribution.slot == slot and contribution.plugin_id == provider:
                return contribution
        return None
