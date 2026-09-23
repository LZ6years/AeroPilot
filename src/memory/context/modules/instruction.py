"""Instruction context module.

The flight memory keeps full instruction records for audit and replay, but the
planner only needs a compact view of the active instruction plus recent history.
This module therefore summarizes instruction records deterministically instead
of forwarding complete semantic plans every planning window.
"""

from __future__ import annotations

from typing import Any
import copy

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, TaskContextInput


class DefaultInstructionContextModule(ContextModule):
    name = "instruction"
    priority = 100
    max_items = 5

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextFragment:
        return ContextFragment(
            name=self.name, enabled=True, source=self.__class__.__name__,
            content={"instruction": stage_context.active_instruction,
                     "protocol_context": stage_context.protocol_context,
                     "task_plan": copy.deepcopy(stage_context.task_plan)},
            confidence="high", budget=BudgetSpec(priority=self.priority),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Raw instruction, applicable protocol and optional advisory plan.",
            "content_fields": {"instruction": "Original user instruction.",
                               "protocol_context": "Caller-owned protocol.",
                               "task_plan": "Optional advisory plan; not environmental evidence."},
        }
