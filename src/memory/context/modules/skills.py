"""Skill context module backed by the plugin registry."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput
from src.plugins import PluginRegistry
from src.plugins.skills import ModelSkillSelector, SkillContentLoader


class DefaultSkillsContextModule(ContextModule):
    name = "skills"
    priority = 95
    max_items = 8

    def __init__(
        self,
        *,
        plugin_registry: PluginRegistry | None = None,
        skill_selector: ModelSkillSelector | None = None,
        content_loader: SkillContentLoader | None = None,
    ) -> None:
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()
        self.skill_selector = skill_selector or ModelSkillSelector()
        self.content_loader = content_loader or SkillContentLoader()

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        catalog = self.plugin_registry.skill_catalog
        listings = catalog.listings()
        selection_context = _selection_context(stage_context, listings)
        selection = self.skill_selector.select(selection_context, catalog)
        accepted, rejected = catalog.validate_selection(selection.accepted_skill_names)
        selected = [self.content_loader.load_summary(skill) for skill in accepted]
        callable_skills = self.plugin_registry.skill_capability_catalog.listings()
        warnings = list(self.plugin_registry.warnings) + list(selection.warnings)
        warnings.extend(f"skill_selection_rejected:{name}" for name in rejected)
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "available_skills": listings,
                "selected_skills": selected,
                "available_capabilities": callable_skills,
                "selection": selection.to_dict(),
            },
            facts=[],
            inferences=[
                {
                    "type": "skill_guidance_available",
                    "value": [item["name"] for item in listings],
                    "source": self.__class__.__name__,
                }
            ]
            if listings
            else [],
            risks=[] if selected else [{"type": "no_skill_content_loaded", "source": self.__class__.__name__}],
            confidence="medium" if selected else "low",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=20 + 120 * len(selected), max_items=self.max_items),
            warnings=warnings,
            metadata={"enabled_plugins": list(self.plugin_registry.enabled_plugins)},
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Model-selected reusable skills from enabled plugins. Skills are guidance; callable skills also declare skill.* tools.",
            "content_fields": {
                "available_skills": "Enabled skill listing with name, description, and optional capability metadata.",
                "selected_skills": "Validated selected skill bodies and bounded skill-local resources loaded by runtime.",
                "available_capabilities": "Callable skill tools enabled by plugin configuration.",
                "selection": "Selection audit data: mode, status, requested/accepted/rejected skill names, and warnings.",
            },
            "risk_fields": {
                "no_skill_content_loaded": "No skill body was selected or loaded; planner should rely on other context fragments.",
            },
            "inference_fields": {
                "skill_guidance_available": "Reusable guidance exists but remains guidance, not observed fact.",
            },
        }


def _selection_context(stage_context: StageContextInput, listings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "active_instruction": stage_context.active_instruction,
        "semantic_plan": stage_context.semantic_plan,
        "current_phase": stage_context.current_phase,
        "semantic_goal": stage_context.semantic_goal,
        "phase_completion_criteria": stage_context.phase_completion_criteria,
        "phase_execution_summary": stage_context.phase_execution_summary,
        "metadata": dict(stage_context.metadata),
        "available_skills": listings,
        "instruction": (
            "Select zero or more skill names from available_skills that should be loaded before action planning. "
            "Select the smallest useful set, usually 0-2 skills and at most 3 when the task clearly combines "
            "several concerns. Return [] for simple primitive motion without a target, search uncertainty, safety "
            "condition, or completion decision. Do not select every skill by default. "
            "Return JSON: {\"selected_skills\": [\"skill_name\"]}."
        ),
    }
