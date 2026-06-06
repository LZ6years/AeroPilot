"""Directory-convention plugin discovery.

The MVP intentionally has no plugin.json. Plugin ids are derived from paths such
as plugins/knowledge/uav_flight -> knowledge.uav_flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.plugins.contracts.context import ContextModuleContribution
from src.plugins.contracts.skill import Skill
from src.plugins.core.contributions import PluginContributions
from src.plugins.skills.markdown import SkillMarkdownError, parse_skill_markdown


DEFAULT_PLUGIN_ROOTS = (
    Path("plugins") / "knowledge",
    Path("plugins") / "capabilities",
    Path("plugins") / "research",
)


@dataclass(frozen=True)
class PluginDiscoveryResult:
    contributions: PluginContributions
    warnings: list[str] = field(default_factory=list)


class PluginDiscovery:
    def __init__(
        self,
        roots: list[str | Path] | None = None,
        *,
        enabled: list[str] | None = None,
        disabled: list[str] | None = None,
    ) -> None:
        self.roots = [Path(item) for item in (roots or DEFAULT_PLUGIN_ROOTS)]
        self.enabled = {str(item) for item in enabled or []}
        self.disabled = {str(item) for item in disabled or []}

    def discover(self) -> PluginDiscoveryResult:
        skills: list[Skill] = []
        context_modules: list[ContextModuleContribution] = []
        warnings: list[str] = []
        seen_skill_names: set[str] = set()
        for root in self.roots:
            if not root.exists():
                continue
            category = root.name
            for plugin_dir in sorted(item for item in root.iterdir() if item.is_dir()):
                plugin_id = f"{category}.{plugin_dir.name}"
                if self.enabled and plugin_id not in self.enabled:
                    continue
                if plugin_id in self.disabled:
                    continue
                plugin_skills, plugin_warnings = _discover_skills(plugin_dir, plugin_id)
                for skill in plugin_skills:
                    if skill.name in seen_skill_names:
                        warnings.append(f"Duplicate skill name ignored: {skill.name} ({skill.skill_path})")
                        continue
                    seen_skill_names.add(skill.name)
                    skills.append(skill)
                warnings.extend(plugin_warnings)
                context_modules.extend(_discover_context_modules(plugin_dir, plugin_id))
        return PluginDiscoveryResult(
            PluginContributions(skills=skills, context_modules=context_modules),
            warnings=warnings,
        )


def _discover_skills(plugin_dir: Path, plugin_id: str) -> tuple[list[Skill], list[str]]:
    skills: list[Skill] = []
    warnings: list[str] = []
    for skill_dir in sorted(item for item in plugin_dir.iterdir() if item.is_dir() and item.name != "context_modules"):
        path = skill_dir / "SKILL.md"
        if not path.exists():
            warnings.append(f"Skill folder missing SKILL.md: {skill_dir}")
            continue
        try:
            skills.append(parse_skill_markdown(path, plugin_id))
        except SkillMarkdownError as exc:
            warnings.append(str(exc))
    return skills, warnings


def _discover_context_modules(plugin_dir: Path, plugin_id: str) -> list[ContextModuleContribution]:
    modules: list[ContextModuleContribution] = []
    modules_dir = plugin_dir / "context_modules"
    if not modules_dir.exists():
        return modules
    category, _, plugin_name = plugin_id.partition(".")
    for path in sorted(modules_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        slot = path.stem
        module_path = f"plugins.{category}.{plugin_name}.context_modules.{slot}"
        modules.append(
            ContextModuleContribution(
                plugin_id=plugin_id,
                slot=slot,
                module_path=module_path,
                class_name="PluginContextModule",
                source_path=path,
            )
        )
    return modules
