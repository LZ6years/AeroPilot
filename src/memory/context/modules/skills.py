"""Skill context module backed by the plugin registry."""

from __future__ import annotations

from typing import Any
import copy

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, TaskContextInput
from src.plugins import PluginRegistry

class DefaultSkillsContextModule(ContextModule):
    """Only expose the enabled knowledge directory and instruction load index."""
    name = "skills"
    priority = 95

    def __init__(self, *, plugin_registry: PluginRegistry | None = None) -> None:
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextFragment:
        # The context offers a directory, never calls a model or loads a
        # body. The instruction-scoped knowledge service owns delivery.
        catalog = [{"name": skill.name, "description": skill.description}
                   for skill in self.plugin_registry.skill_catalog.skills
                   if skill.capability is None]
        return ContextFragment(
            name=self.name, enabled=True, source=self.__class__.__name__,
            content={"knowledge_catalog": catalog,
                     "loaded_knowledge": copy.deepcopy(stage_context.loaded_knowledge)},
            confidence="high", budget=BudgetSpec(priority=self.priority),
            warnings=list(self.plugin_registry.warnings),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Enabled knowledge directory and loaded versions, not knowledge bodies.",
            "content_fields": {
                "knowledge_catalog": "Enabled knowledge names and descriptions, not loaded bodies.",
                "loaded_knowledge": "Instruction-scoped names, versions and references already loaded.",
            },
        }
