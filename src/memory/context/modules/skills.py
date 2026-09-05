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
    _CACHEABLE_SELECTION_STATUSES = {
        "selected",
        "none_selected",
        "disabled",
        "empty_catalog",
    }

    def __init__(
        self,
        *,
        plugin_registry: PluginRegistry | None = None,
        skill_selector: ModelSkillSelector | None = None,
        content_loader: SkillContentLoader | None = None,
    ) -> None:
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()
        # Context modules never discover model credentials.  A composition
        # root must inject an enabled selector; otherwise skill selection is
        # intentionally disabled and remains free of network side effects.
        self.skill_selector = skill_selector or ModelSkillSelector(enabled=False)
        self.content_loader = content_loader or SkillContentLoader()
        self._phase_selection_cache: dict[tuple[str, str, str], SkillSelectionResult] = {}

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        catalog = self.plugin_registry.skill_catalog
        listings = catalog.index_listings()
        selection_context = _selection_context(stage_context, listings)
        cache_key = _phase_cache_key(stage_context)
        cached_selection = self._phase_selection_cache.get(cache_key)
        selection_source = "phase_cache" if cached_selection is not None else "fresh"
        if cached_selection is None:
            selection = self.skill_selector.select(selection_context, catalog)
            # Provider/network failures may be transient.  Caching those
            # results would silently remove skill guidance for the whole phase.
            if selection.status in self._CACHEABLE_SELECTION_STATUSES:
                self._phase_selection_cache[cache_key] = selection
        else:
            selection = cached_selection
        accepted, rejected = catalog.validate_selection(selection.accepted_skill_names)
        selected = [
            self.content_loader.load_planner_guidance(
                skill,
                requested_references=selection.accepted_references.get(skill.name, []),
            )
            for skill in accepted
        ]
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
            budget=BudgetSpec(priority=self.priority, tokens_estimate=80 + 700 * len(selected), max_items=self.max_items),
            warnings=warnings,
            metadata={
                "enabled_plugins": list(self.plugin_registry.enabled_plugins),
                "available_skill_names": [item["name"] for item in listings],
                "available_skill_references": _available_skill_references(listings),
                "selection": selection.to_dict(),
                "selection_source": selection_source,
                "phase_key": cache_key[1],
            },
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Reusable guidance selected from enabled plugins. Skills are guidance, not observed facts.",
            "content_fields": {
                "selected_skills": "Selected skill guidance loaded by runtime for this planning step, including bounded SKILL.md text for accepted skills and optional requested references.",
                "callable_capabilities": "Callable skill tools enabled by plugin configuration, present only when non-empty.",
            },
            "risk_fields": {
                "no_skill_content_loaded": "No skill body was selected or loaded; planner should rely on other context fragments.",
            },
            "metadata_fields": {
                "available_skill_names": "Debug/audit list of enabled skill names considered by the selector.",
                "available_skill_references": "Debug/audit reference manifest considered by the selector.",
                "selection": "Debug/audit data for skill selection; not needed for action reasoning.",
                "selection_source": "Whether skills were selected fresh for this semantic phase or reused from the phase cache.",
                "phase_key": "Semantic phase cache key used for phase-local skill reuse.",
            },
        }


def _selection_context(stage_context: StageContextInput, listings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "active_instruction": stage_context.active_instruction,
        "current_phase": _compact_current_phase(stage_context.current_phase),
        "semantic_goal": stage_context.semantic_goal,
        "phase_progress_brief": _phase_progress_brief(stage_context.phase_execution_summary),
        "available_skills": listings,
        "instruction": (
            "Select zero or more skill names from available_skills that should be loaded before action planning. "
            "Select the smallest useful set, usually 0-2 skills and at most 3 when the task clearly combines several "
            "concerns. Request references only from selected skills and only when the skill summary is insufficient. "
            "Return JSON: {\"selected_skills\": [\"skill_name\"], \"requested_references\": {\"skill_name\": [\"references/file.md\"]}}."
        ),
    }


def _compact_current_phase(phase: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(phase, dict):
        return None
    keys = ("name", "goal", "done_when", "continue_when")
    return {key: phase[key] for key in keys if key in phase}


def _phase_progress_brief(summary: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    tool_facts = summary.get("tool_facts", {}) if isinstance(summary.get("tool_facts"), dict) else {}
    measurements = summary.get("measurements", {}) if isinstance(summary.get("measurements"), dict) else {}
    tools_executed = tool_facts.get("tools_executed", {})
    if not isinstance(tools_executed, dict):
        tools_executed = {}
    return {
        "window_count": summary.get("window_count"),
        "tools_executed": dict(tools_executed),
        "has_translation": _has_translation(measurements, tool_facts, tools_executed),
        "yaw_only_so_far": _yaw_only_so_far(measurements, tool_facts, tools_executed),
    }


def _has_translation(measurements: dict[str, Any], tool_facts: dict[str, Any], tools_executed: dict[str, Any]) -> bool:
    for key in ("forward_m", "right_m", "up_m", "planar_m"):
        try:
            if abs(float(measurements.get(key, 0.0))) > 0.05:
                return True
        except (TypeError, ValueError):
            pass
    try:
        if float(tool_facts.get("move_duration_success_sum_sec", 0.0)) > 0.0:
            return True
    except (TypeError, ValueError):
        pass
    return int(tools_executed.get("uav.move_body", 0) or 0) > 0


def _yaw_only_so_far(measurements: dict[str, Any], tool_facts: dict[str, Any], tools_executed: dict[str, Any]) -> bool:
    if _has_translation(measurements, tool_facts, tools_executed):
        return False
    yaw_delta = 0.0
    turn_yaw = 0.0
    try:
        yaw_delta = abs(float(measurements.get("yaw_delta_deg", 0.0)))
    except (TypeError, ValueError):
        yaw_delta = 0.0
    try:
        turn_yaw = abs(float(tool_facts.get("turn_yaw_success_sum_deg", 0.0)))
    except (TypeError, ValueError):
        turn_yaw = 0.0
    return bool(int(tools_executed.get("uav.turn_relative", 0) or 0) > 0 or yaw_delta > 0.0 or turn_yaw > 0.0)


def _available_skill_references(listings: list[dict[str, Any]]) -> dict[str, list[str]]:
    references: dict[str, list[str]] = {}
    for item in listings:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        refs = item.get("references", [])
        if not name or not isinstance(refs, list):
            continue
        paths = [
            str(ref.get("path", "")).strip()
            for ref in refs
            if isinstance(ref, dict) and str(ref.get("path", "")).strip()
        ]
        if paths:
            references[name] = paths
    return references


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
