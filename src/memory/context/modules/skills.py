"""Skill context module backed by the plugin registry."""

from __future__ import annotations

from typing import Any
from src.memory.context.modules.base import ContextModule
from src.memory.context.contracts import (
    ContextFragment,
    ContextModuleResult,
    FragmentAudit,
    MemorySourceBundle,
    TaskContextInput,
)

class DefaultSkillsContextModule(ContextModule):
    """Only expose the enabled knowledge directory and instruction load index."""
    name = "skills"
    priority = 95

    def __init__(self, *, warnings: list[str] | None = None) -> None:
        self.warnings = tuple(warnings or ())

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextModuleResult:
        # Catalog entries are instruction-stable and live in InstructionEnvelope.
        return ContextModuleResult(
            ContextFragment(self.name, {}),
            FragmentAudit(
                name=self.name,
                source=self.__class__.__name__,
                confidence="high",
                warnings=self.warnings,
            ),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Enabled knowledge directory and loaded versions, not knowledge bodies.",
            "model_fields": {},
            "audit_fields": {},
        }
