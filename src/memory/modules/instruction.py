"""Instruction history context module."""

from __future__ import annotations

from typing import Any

from src.memory.modules.base import ContextModule
from src.memory.schemas import BudgetSpec, ContextFragment, MemorySourceBundle


class DefaultInstructionContextModule(ContextModule):
    name = "instruction"
    priority = 100
    max_items = 5

    def build(self, sources: MemorySourceBundle, task_context: dict[str, Any]) -> ContextFragment:
        memory = sources.flight_memory
        plan = task_context.get("semantic_plan", {})
        current_phase = task_context.get("current_phase")
        history = memory.instruction_history[-self.max_items :]
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
