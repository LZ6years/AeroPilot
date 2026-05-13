"""Primitive action history context module."""

from __future__ import annotations

from typing import Any

from src.memory.modules.base import ContextModule
from src.memory.schemas import BudgetSpec, ContextFragment, MemorySourceBundle


class DefaultActionContextModule(ContextModule):
    name = "action"
    priority = 90
    max_items = 8

    def build(self, sources: MemorySourceBundle, task_context: dict[str, Any]) -> ContextFragment:
        actions = sources.flight_memory.recent_actions[-self.max_items :]
        facts = [
            {
                "type": "action_result",
                "tool": item.get("tool"),
                "args": item.get("args", {}),
                "ok": item.get("result", {}).get("ok"),
                "source": self.__class__.__name__,
            }
            for item in actions
        ]
        failed = [item for item in actions if item.get("result", {}).get("ok") is False]
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={"recent_actions": actions},
            facts=facts,
            inferences=[
                {
                    "type": "latest_action_effect",
                    "value": actions[-1],
                    "source": self.__class__.__name__,
                }
            ]
            if actions
            else [],
            risks=[
                {
                    "type": "recent_action_failure",
                    "value": failed[-1],
                    "source": self.__class__.__name__,
                }
            ]
            if failed
            else [],
            confidence="high",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=260, max_items=self.max_items),
        )
