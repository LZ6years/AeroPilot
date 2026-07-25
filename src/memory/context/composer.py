"""Compose planner context from registered context fragments."""

from __future__ import annotations

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
        budget_metadata = _budget_metadata(fragments)
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
    return {
        "planner_context_fields": {
            "active_instruction": "Current user instruction.",
            "current_phase": "Current semantic phase with goal, done_when, continue_when, and metrics.",
            "semantic_goal": "Parsed lightweight task_goal.",
            "fragments": "Enabled context fragments.",
            "rules": "Global FlyAgent behavior guidance.",
            "user_preferences": "User response preferences.",
            "safety_constraints": "Current safety state.",
            "allowed_action_constraints": "Allowed tools, forbidden tools, schemas, and limits.",
            "observation_policy": "Runtime collects observations automatically; do not output observe steps.",
            "phase_execution_summary": "Runtime/debug progress source; model-facing motion is consolidated into the action fragment.",
        },
        "fragment_fields": {
            "name": "Fragment type.",
            "content": "Planner-facing compact content.",
            "model_inputs": "Attached image/text model inputs.",
            "risks": "Warnings or missing evidence.",
            "metadata": "Debug/audit metadata; do not treat as task evidence.",
        },
        "module_content_fields": _module_content_fields(module_descriptions),
    }


def _module_content_fields(module_descriptions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    legacy_fields = _compact_module_content_fields()
    fields: dict[str, dict[str, Any]] = {}
    for name, description in module_descriptions.items():
        content_fields = description.get("content_fields") if isinstance(description, dict) else None
        if isinstance(content_fields, dict) and content_fields:
            fields[name] = dict(content_fields)
            continue
        if name in legacy_fields:
            fields[name] = dict(legacy_fields[name])
    return fields


def _compact_module_content_fields() -> dict[str, dict[str, str]]:
    return {
        "instruction": {
            "current_instruction": "Current instruction summary.",
            "instruction_history": "Recent compact instruction summaries.",
        },
        "skills": {
            "selected_skills": "Selected skill summaries, bounded SKILL.md guidance text, and any explicitly requested reference text.",
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
            "motion_history_summary": "Current-phase UAV motion plus up to five successful actions ordered oldest to newest.",
        },
        "map": {
            "recalled_map_memories": "Map memory clues; confirm with current observations.",
            "recall_filter": "Map recall filter.",
        },
    }


def _budget_metadata(fragments: list[ContextFragment]) -> dict[str, Any]:
    tokens = _tokens_estimate(fragments)
    return {
        "final_tokens_estimate": tokens,
        "trimmed_fragments": [],
        "budget_applied": False,
    }


def _tokens_estimate(fragments: list[ContextFragment]) -> int:
    return sum(max(0, int(fragment.budget.tokens_estimate)) for fragment in fragments)
