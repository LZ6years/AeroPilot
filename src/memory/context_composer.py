"""Compose planner context from registered context fragments."""

from __future__ import annotations

from typing import Any

from src.memory.registry import ContextModuleRegistry
from src.memory.schemas import MemorySourceBundle, PlannerContext


class ContextComposer:
    def __init__(
        self,
        registry: ContextModuleRegistry | None = None,
        rules: dict[str, Any] | None = None,
        user_preferences: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry or ContextModuleRegistry.default()
        self.rules = dict(rules or {})
        self.user_preferences = dict(user_preferences or {})

    def compose(
        self,
        sources: MemorySourceBundle,
        task_context: dict[str, Any],
        safety_constraints: dict[str, Any] | None = None,
        allowed_action_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fragments = self.registry.build_fragments(sources, task_context)
        fragments.sort(key=lambda item: item.budget.priority, reverse=True)
        memory = sources.flight_memory
        context = PlannerContext(
            active_instruction=memory.active_instruction,
            current_phase=task_context.get("current_phase"),
            semantic_goal=dict(task_context.get("semantic_goal", {})),
            fragments=fragments,
            disabled_fragments=list(self.registry.disabled),
            rules=self.rules,
            user_preferences=self.user_preferences,
            safety_constraints=dict(safety_constraints or {}),
            allowed_action_constraints=dict(allowed_action_constraints or {}),
            metadata={
                "run_id": memory.run_id,
                "map_id": memory.map_id,
                "scene_instance_id": memory.scene_instance_id,
                "agent_id": memory.agent_id,
                "reset_epoch": memory.reset_epoch,
            },
        )
        return context.to_dict()
