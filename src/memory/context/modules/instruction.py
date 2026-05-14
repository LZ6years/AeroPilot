"""Instruction history context module."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput


class DefaultInstructionContextModule(ContextModule):
    name = "instruction"
    priority = 100
    max_items = 5

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        memory = sources.flight_memory
        reader = memory.memory_reader
        plan = stage_context.semantic_plan
        current_phase = stage_context.current_phase
        history = reader.read_refs(memory.instruction_history[-self.max_items :]) if reader else []
        goals = [
            {
                "type": "mission_goal",
                "value": plan.get("goal", {}),
                "source": self.__class__.__name__,
            }
        ]
        if current_phase:
            goals.append({"type": "phase_goal", "value": current_phase, "source": self.__class__.__name__})
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "active_instruction": memory.active_instruction,
                "instruction_history": history,
                "semantic_intent": plan.get("intent", ""),
                "uncertainties": plan.get("uncertainties", []),
            },
            goals=goals,
            confidence="high",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=220, max_items=self.max_items),
            metadata={"reset_epoch": memory.reset_epoch},
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Current instruction, recent instruction history, and semantic goals for the active phase.",
            "content_fields": {
                "active_instruction": "The current user instruction that should drive this planning step.",
                "instruction_history": "Recent instruction records loaded from memory refs; use only as short context, not as active commands.",
                "semantic_intent": "Compact intent label or phrase from the task parser.",
                "uncertainties": "Known uncertainties from task parsing that should make planning more conservative.",
            },
            "goal_fields": {
                "mission_goal": "Overall parsed mission goal.",
                "phase_goal": "The current phase objective that should guide the next 1-3 actions.",
            },
        }
