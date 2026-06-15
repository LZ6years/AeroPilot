"""Skill context module backed by the plugin registry."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput
from src.plugins import PluginRegistry
from src.plugins.skills import ModelSkillSelector, SkillContentLoader, SkillSelectionResult


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
        self._phase_selection_cache: dict[tuple[str, str, str], SkillSelectionResult] = {}

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        catalog = self.plugin_registry.skill_catalog
        listings = catalog.listings()
        selection_context = _selection_context(stage_context, listings)
        cache_key = _phase_cache_key(stage_context)
        cached_selection = self._phase_selection_cache.get(cache_key)
        selection_source = "phase_cache" if cached_selection is not None else "fresh"
        if cached_selection is None:
            selection = self.skill_selector.select(selection_context, catalog)
            self._phase_selection_cache[cache_key] = selection
        else:
            selection = cached_selection
        accepted, rejected = catalog.validate_selection(selection.accepted_skill_names)
        selected = [_planner_skill_summary(self.content_loader.load_summary(skill)) for skill in accepted]
        callable_skills = self.plugin_registry.skill_capability_catalog.listings()
        warnings = list(self.plugin_registry.warnings) + list(selection.warnings)
        warnings.extend(f"skill_selection_rejected:{name}" for name in rejected)
        content: dict[str, Any] = {
            "selected_skills": selected,
        }
        if callable_skills:
            content["callable_capabilities"] = callable_skills
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content=content,
            facts=[],
            inferences=[],
            risks=[] if selected else [{"type": "no_skill_content_loaded", "source": self.__class__.__name__}],
            confidence="medium" if selected else "low",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=20 + 120 * len(selected), max_items=self.max_items),
            warnings=warnings,
            metadata={
                "enabled_plugins": list(self.plugin_registry.enabled_plugins),
                "available_skill_names": [item["name"] for item in listings],
                "selection": selection.to_dict(),
                "selection_source": selection_source,
                "phase_key": cache_key[1],
            },
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Reusable guidance selected from enabled plugins. Skills are guidance, not observed facts.",
            "content_fields": {
                "selected_skills": "Compact selected skill guidance loaded by runtime for this planning step.",
                "callable_capabilities": "Callable skill tools enabled by plugin configuration, present only when non-empty.",
            },
            "risk_fields": {
                "no_skill_content_loaded": "No skill body was selected or loaded; planner should rely on other context fragments.",
            },
            "metadata_fields": {
                "available_skill_names": "Debug/audit list of enabled skill names considered by the selector.",
                "selection": "Debug/audit data for skill selection; not needed for action reasoning.",
                "selection_source": "Whether skills were selected fresh for this semantic phase or reused from the phase cache.",
                "phase_key": "Semantic phase cache key used for phase-local skill reuse.",
            },
        }


def _selection_context(stage_context: StageContextInput, listings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "active_instruction": stage_context.active_instruction,
        "semantic_plan": stage_context.semantic_plan,
        "current_phase": stage_context.current_phase,
        "semantic_goal": stage_context.semantic_goal,
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


def _phase_cache_key(stage_context: StageContextInput) -> tuple[str, str, str]:
    metadata = stage_context.metadata if isinstance(stage_context.metadata, dict) else {}
    run_id = str(metadata.get("run_id") or "")
    phase = stage_context.current_phase if isinstance(stage_context.current_phase, dict) else {}
    phase_key = str(metadata.get("current_phase_key") or _phase_key_from_phase(phase))
    return (run_id, phase_key, stage_context.active_instruction)


def _phase_key_from_phase(phase: dict[str, Any]) -> str:
    index = phase.get("index")
    name = phase.get("name")
    if index is None and not name:
        return "no_phase"
    return f"{index}:{name}"


def _planner_skill_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the selected skill shape that is useful for action planning."""

    compact: dict[str, Any] = {
        "name": summary.get("name", ""),
        "description": summary.get("description", ""),
        "skill_text": summary.get("skill_text", ""),
    }
    references = [
        {
            "name": item.get("path", ""),
            "text": item.get("text", ""),
            "truncated": bool(item.get("truncated", False)),
        }
        for item in summary.get("references", [])
        if isinstance(item, dict) and item.get("text")
    ]
    if references:
        compact["references"] = references
    capability = summary.get("capability")
    if capability:
        compact["capability"] = capability
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}
