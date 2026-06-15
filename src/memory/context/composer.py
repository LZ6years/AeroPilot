"""Compose planner context from registered context fragments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.schemas import ContextFragment, MemorySourceBundle, PlannerContext, StageContextInput


DEFAULT_MAX_TOKENS_ESTIMATE = 2000
DEFAULT_RULES_PATH = Path("memory") / "FLYAGENT.md"
DEFAULT_USER_PREFERENCES_PATH = Path("memory") / "USER.md"


class ContextComposer:
    def __init__(
        self,
        registry: ContextModuleRegistry | None = None,
        rules: dict[str, Any] | None = None,
        user_preferences: dict[str, Any] | None = None,
        max_tokens_estimate: int = DEFAULT_MAX_TOKENS_ESTIMATE,
        rules_path: str | Path = DEFAULT_RULES_PATH,
        user_preferences_path: str | Path = DEFAULT_USER_PREFERENCES_PATH,
    ) -> None:
        self.registry = registry or ContextModuleRegistry.default()
        self.rules = _merge_text_config(_read_markdown_config(rules_path), rules)
        self.user_preferences = _merge_text_config(_read_markdown_config(user_preferences_path), user_preferences)
        self.max_tokens_estimate = int(max_tokens_estimate)

    def compose(
        self,
        sources: MemorySourceBundle,
        stage_context: StageContextInput,
    ) -> dict[str, Any]:
        fragments = self.registry.build_fragments(sources, stage_context)
        fragments.sort(key=lambda item: item.budget.priority, reverse=True)
        original_tokens = _tokens_estimate(fragments)
        fragments, budget_metadata = _apply_budget(fragments, self.max_tokens_estimate)
        memory = sources.flight_memory
        context = PlannerContext(
            active_instruction=stage_context.active_instruction,
            current_phase=stage_context.current_phase,
            semantic_goal=dict(stage_context.semantic_goal),
            fragments=fragments,
            disabled_fragments=list(self.registry.disabled),
            context_schema=_build_context_schema(self.registry.describe_modules()),
            rules=self.rules,
            user_preferences=self.user_preferences,
            safety_constraints=dict(stage_context.safety_constraints),
            allowed_action_constraints=dict(stage_context.allowed_action_constraints),
            observation_policy=dict(stage_context.observation_policy),
            phase_execution_summary=dict(stage_context.phase_execution_summary),
            metadata={
                "run_id": memory.run_id,
                "map_id": memory.map_id,
                "scene_instance_id": memory.scene_instance_id,
                "agent_id": memory.agent_id,
                "reset_epoch": memory.reset_epoch,
                "budget": {
                    "max_tokens_estimate": self.max_tokens_estimate,
                    "original_tokens_estimate": original_tokens,
                    **budget_metadata,
                },
                **dict(stage_context.metadata),
            },
        )
        return context.to_dict()


def _read_markdown_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return {}
    return {"text": text, "source": str(config_path)}


def _merge_text_config(file_config: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(file_config)
    if override:
        data.update(dict(override))
    return data


def _build_context_schema(module_descriptions: dict[str, Any]) -> dict[str, Any]:
    enabled_modules = set(module_descriptions)
    return {
        "planner_context_fields": {
            "active_instruction": "Current user instruction.",
            "current_phase": "Current semantic phase with goal, done_when, observe, continue_when, and metrics.",
            "semantic_goal": "Parsed lightweight task_goal.",
            "fragments": "Enabled context fragments.",
            "rules": "Global FlyAgent behavior guidance.",
            "user_preferences": "User response preferences.",
            "safety_constraints": "Current safety state.",
            "allowed_action_constraints": "Allowed tools, forbidden tools, schemas, and limits.",
            "observation_policy": "Runtime collects observations automatically; do not output observe steps.",
            "phase_execution_summary": "Compact measured progress and tool facts in the current phase.",
        },
        "fragment_fields": {
            "name": "Fragment type.",
            "content": "Planner-facing compact content.",
            "model_inputs": "Attached image/text model inputs.",
            "risks": "Warnings or missing evidence.",
            "metadata": "Debug/audit metadata; do not treat as task evidence.",
        },
        "module_content_fields": {
            name: fields
            for name, fields in _compact_module_content_fields().items()
            if name in enabled_modules
        },
    }


def _compact_module_content_fields() -> dict[str, dict[str, str]]:
    return {
        "instruction": {
            "current_instruction": "Current instruction summary.",
            "instruction_history": "Recent compact instruction summaries.",
            "uncertainties": "Known parser uncertainties.",
        },
        "skills": {
            "selected_skills": "Selected reusable guidance for this planning step.",
            "callable_capabilities": "Enabled callable skill tools, present only when non-empty.",
        },
        "image": {
            "selected_frames": "Attached image roles such as current_observation and phase_start_reference.",
            "visual_evidence": "RGB availability, mock status, attached image count, and interpretation status.",
        },
        "pose": {
            "latest_pose": "Latest UAV pose.",
            "recent_pose_delta": "Delta between recent poses.",
        },
        "action": {
            "recent_actions": "Recent compact actions with tool, args, and ok.",
            "action_trend": "Recent motion pattern such as repeated_yaw, alternating_yaw, repeated_move, or mixed.",
            "motion_history_summary": "History-only compact motion summary for current-phase progress when available and recent yaw behavior.",
        },
        "map": {
            "recalled_map_memories": "Map memory clues; confirm with current observations.",
            "recall_filter": "Map recall filter.",
        },
    }


def _apply_budget(fragments: list[ContextFragment], max_tokens: int) -> tuple[list[ContextFragment], dict[str, Any]]:
    if max_tokens <= 0:
        return fragments, {
            "final_tokens_estimate": _tokens_estimate(fragments),
            "trimmed_fragments": [],
            "budget_applied": False,
        }
    current_tokens = _tokens_estimate(fragments)
    trimmed: list[dict[str, Any]] = []
    if current_tokens <= max_tokens:
        return fragments, {
            "final_tokens_estimate": current_tokens,
            "trimmed_fragments": trimmed,
            "budget_applied": False,
        }

    updated = list(fragments)
    for index in reversed(range(len(updated))):
        if current_tokens <= max_tokens:
            break
        fragment = updated[index]
        trimmed_fragment = _trim_fragment_lists(fragment)
        if trimmed_fragment is fragment:
            continue
        before = fragment.budget.tokens_estimate
        after = trimmed_fragment.budget.tokens_estimate
        current_tokens -= max(0, before - after)
        updated[index] = trimmed_fragment
        trimmed.append({"name": fragment.name, "tokens_before": before, "tokens_after": after})

    return updated, {
        "final_tokens_estimate": current_tokens,
        "trimmed_fragments": trimmed,
        "budget_applied": bool(trimmed),
    }


def _trim_fragment_lists(fragment: ContextFragment) -> ContextFragment:
    content, changed = _trim_dict_lists(fragment.content)
    facts = _trim_list(fragment.facts)
    inferences = _trim_list(fragment.inferences)
    goals = _trim_list(fragment.goals)
    risks = _trim_list(fragment.risks)
    artifact_refs = _trim_list(fragment.artifact_refs)
    warnings = _trim_list(fragment.warnings)
    changed = changed or any(
        [
            len(facts) != len(fragment.facts),
            len(inferences) != len(fragment.inferences),
            len(goals) != len(fragment.goals),
            len(risks) != len(fragment.risks),
            len(artifact_refs) != len(fragment.artifact_refs),
            len(warnings) != len(fragment.warnings),
        ]
    )
    if not changed:
        return fragment
    return replace(
        fragment,
        content=content,
        facts=facts,
        inferences=inferences,
        goals=goals,
        risks=risks,
        artifact_refs=artifact_refs,
        warnings=warnings,
        budget=replace(fragment.budget, tokens_estimate=max(0, fragment.budget.tokens_estimate // 2)),
    )


def _trim_dict_lists(data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, list):
            trimmed = _trim_list(value)
            result[key] = trimmed
            changed = changed or len(trimmed) != len(value)
        else:
            result[key] = value
    return result, changed


def _trim_list(items: list[Any]) -> list[Any]:
    if len(items) <= 1:
        return list(items)
    return [items[-1]]


def _tokens_estimate(fragments: list[ContextFragment]) -> int:
    return sum(max(0, int(fragment.budget.tokens_estimate)) for fragment in fragments)
