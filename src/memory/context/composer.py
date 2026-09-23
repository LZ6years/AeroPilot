"""Compose planner context from registered context fragments."""

from __future__ import annotations

from pathlib import Path
import copy
from typing import Any

from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.schemas import ContextFragment, MemorySourceBundle, TaskContextInput


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
        stage_context: TaskContextInput,
    ) -> dict[str, Any]:
        fragments = self.registry.build_fragments(sources, stage_context)
        fragments.sort(key=lambda item: item.budget.priority, reverse=True)
        memory = sources.flight_memory
        return {
            "active_instruction": stage_context.active_instruction,
            "protocol_context": stage_context.protocol_context,
            "task_plan": copy.deepcopy(stage_context.task_plan),
            "fragments": [fragment.to_dict() for fragment in fragments],
            "disabled_fragments": list(self.registry.disabled),
            "context_schema": {
                "planner_context_fields": list(TaskContextInput.__dataclass_fields__),
                "fragment_fields": list(ContextFragment.__dataclass_fields__),
                "module_content_fields": {
                    name: copy.deepcopy(description.get("content_fields", {}))
                    for name, description in self.registry.describe_modules().items()
                },
            },
            "rules": copy.deepcopy(self.rules),
            "user_preferences": copy.deepcopy(self.user_preferences),
            "safety_constraints": copy.deepcopy(stage_context.safety_constraints),
            "observation_policy": copy.deepcopy(stage_context.observation_policy),
            "metadata": {
                "run_id": memory.run_id, **copy.deepcopy(stage_context.metadata),
                "budget": {
                    "max_tokens_estimate": self.max_tokens_estimate,
                    "original_tokens_estimate": sum(item.budget.tokens_estimate for item in fragments),
                    "final_tokens_estimate": sum(item.budget.tokens_estimate for item in fragments),
                    "budget_applied": False, "trimmed_fragments": [],
                },
            },
        }


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
