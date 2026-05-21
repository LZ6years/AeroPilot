"""Compose planner context from registered context fragments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.schemas import ContextFragment, MemorySourceBundle, PlannerContext, StageContextInput


DEFAULT_MAX_TOKENS_ESTIMATE = 1200
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
            phase_transition_policy=dict(stage_context.phase_transition_policy),
            phase_completion_criteria=[dict(item) for item in stage_context.phase_completion_criteria],
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
    return {
        "planner_context_fields": {
            "active_instruction": "The current user instruction the planner should follow.",
            "current_phase": "The current semantic phase; copy current_phase.name into the action plan response phase.",
            "semantic_goal": "The parsed mission goal for the active instruction.",
            "fragments": "Ordered context fragments from enabled memory modules.",
            "disabled_fragments": "Module names disabled by configuration and therefore absent from fragments.",
            "context_schema": "Model-facing explanations for this context shape and each enabled module's fields.",
            "rules": "Project-level FlyAgent behavior guidance loaded from memory/FLYAGENT.md or runtime overrides.",
            "user_preferences": "User-facing preference guidance loaded from memory/USER.md or runtime overrides.",
            "safety_constraints": "Current runtime safety state that should guide planning before SafetySupervisor review.",
            "allowed_action_constraints": "The action planning contract: allowed UAV tools, forbidden tools, schemas, and limits.",
            "observation_policy": "Runtime perception policy. When automatic, the planner must not output observe steps.",
            "phase_transition_policy": "How runtime judges phase progress after tool execution and automatic observation.",
            "phase_completion_criteria": "Current semantic phase completion contract. Phase names are free-form; these criteria define target axis, target value, tolerance, and whether correction is allowed.",
            "phase_execution_summary": "Clean model-facing alias of latest_state.current_phase_progress. It contains current-phase measured/executed facts, coordinate frame notes, world_pose start/latest, phase_local_frame, last_window_measurements, and phase-local forward/right/up projection fields. Raw velocity and duplicate start_pose/latest_pose fields are intentionally omitted.",
            "metadata": "Run identifiers, map/scene identifiers, budget metadata, and other audit information.",
        },
        "fragment_common_fields": {
            "name": "Fragment category, such as instruction, image, pose, action, or map.",
            "enabled": "Whether this fragment's module was enabled when the fragment was produced.",
            "source": "The module class that produced this fragment.",
            "content": "Module-specific structured content; see module_content_fields for the matching fragment name.",
            "facts": "Observed or verifiable facts. Prefer these over inferences when planning.",
            "inferences": "Hints, guesses, derived values, or map-memory clues that require caution or confirmation.",
            "goals": "Mission or phase goals contributed by this fragment.",
            "risks": "Uncertainty, missing evidence, failures, or reconfirmation requirements.",
            "confidence": "Module confidence label: low, medium, or high.",
            "artifact_refs": "Paths or refs to artifacts such as image files or memory records.",
            "budget": "Priority and token estimate used by ContextComposer ordering and trimming.",
            "warnings": "Non-fatal issues that should make planning more conservative.",
            "metadata": "Auxiliary module metadata such as reset_epoch, map_id, or coordinate frame.",
        },
        "module_content_fields": dict(module_descriptions),
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
