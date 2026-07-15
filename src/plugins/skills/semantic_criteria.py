"""Select bounded knowledge guidance for semantic phase criteria.

This module is the plugin-side bridge used by the model task parser. It selects
skills from an immutable phase structure, validates names against the enabled
catalog, and loads knowledge text only. It does not parse tasks, create phases,
load references, or expose callable skill capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.plugins import PluginRegistry
from src.plugins.skills.loader import SkillContentLoader
from src.plugins.skills.selection import ModelSkillSelector
from src.runtime.configs.model_config import ModelConfig


SEMANTIC_CRITERIA_SKILL_SELECTION_PROMPT = """You select UAV-Agent knowledge skills for one semantic phase.

The phase structure already exists and is immutable. Select all and only the
knowledge skills that are directly useful for defining this phase's observable
completion conditions, continuation conditions, evidence sources, or valid
metrics. Skills must not change the phase name, goal, order, or task meaning.

Use active_instruction and semantic_goal to understand user intent. Use
current_phase.name and current_phase.goal as the primary selection basis. There
is no numeric selection limit, but choose the smallest genuinely useful set;
never select unrelated skills or all skills as insurance.

Return exactly one JSON object:
{"selected_skills": ["skill_name"]}

Choose only exact names from available_skills. Do not request references.
"""


@dataclass(frozen=True)
class SemanticCriteriaSkillGuidance:
    """Validated, bounded guidance selected for one immutable phase."""

    status: str
    selected_skills: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def selected_skill_names(self) -> list[str]:
        return [str(item.get("name", "")) for item in self.selected_skills if item.get("name")]

    def trace(self) -> dict[str, Any]:
        return {
            "selection_status": self.status,
            "selected_skills": self.selected_skill_names,
            "warnings": list(self.warnings),
        }


class SemanticCriteriaSkillResolver:
    """Resolve knowledge skills independently for each semantic phase."""

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        plugin_registry: PluginRegistry | None = None,
        selector: ModelSkillSelector | None = None,
        content_loader: SkillContentLoader | None = None,
    ) -> None:
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()
        selection_mode = str(self.plugin_registry.skills_config.get("selection", "model")).strip().lower()
        self.selector = selector or ModelSkillSelector(
            model_config,
            enabled=selection_mode == "model",
            system_prompt=SEMANTIC_CRITERIA_SKILL_SELECTION_PROMPT,
        )
        self.content_loader = content_loader or SkillContentLoader()

    def resolve(
        self,
        instruction: str,
        semantic_goal: dict[str, Any],
        phase: dict[str, Any],
    ) -> SemanticCriteriaSkillGuidance:
        catalog = self.plugin_registry.skill_catalog
        selection_context = {
            "active_instruction": instruction,
            "semantic_goal": dict(semantic_goal),
            "current_phase": {
                key: phase[key]
                for key in ("index", "name", "goal")
                if key in phase
            },
            "available_skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "summary": skill.summary,
                }
                for skill in catalog.skills
            ],
        }
        selection = self.selector.select(selection_context, catalog)
        accepted, rejected = catalog.validate_selection(selection.accepted_skill_names)
        warnings = list(selection.warnings)
        warnings.extend(f"skill_selection_rejected:{name}" for name in rejected)
        if selection.requested_references:
            warnings.append("semantic_criteria_skill_references_ignored")
        selected = [self.content_loader.load_semantic_guidance(skill) for skill in accepted]
        for item in selected:
            if item.pop("truncated", False):
                warnings.append(f"semantic_criteria_skill_truncated:{item.get('name', '')}")
        return SemanticCriteriaSkillGuidance(
            status=selection.status,
            selected_skills=selected,
            warnings=_dedupe(warnings),
        )


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in items:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
