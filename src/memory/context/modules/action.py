"""Primitive action history context module."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput


class DefaultActionContextModule(ContextModule):
    name = "action"
    priority = 90
    max_items = 5

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        memory = sources.flight_memory
        reader = memory.memory_reader
        actions = reader.read_refs(memory.recent_action_refs[-self.max_items :]) if reader else []
        compact_actions = [_compact_action(item) for item in actions]
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
        risks = [
            {
                "type": "recent_action_failure",
                "value": failed[-1],
                "source": self.__class__.__name__,
            }
        ] if failed else []
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "recent_actions": compact_actions,
            },
            facts=facts,
            risks=risks,
            confidence="high",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=260, max_items=self.max_items),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Recent primitive actions and compact execution results.",
            "content_fields": {
                "recent_actions": "Up to five recent actions with tool, args, and compact result fields.",
                "recent_actions[].result.ok": "Whether the tool execution reported success.",
                "recent_actions[].result.action": "Runtime action label returned by the tool result.",
                "recent_actions[].result.message": "Short runtime message from the tool result.",
            },
            "fact_fields": {
                "action_result": "Observed execution result for a recent action; use this to avoid repeating known failures.",
            },
            "risk_fields": {
                "recent_action_failure": "Most recent failed action record, if any.",
            },
        }


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    result = action.get("result", {}) if isinstance(action, dict) else {}
    return {
        "tool": action.get("tool"),
        "args": dict(action.get("args", {})),
        "result": {
            "ok": result.get("ok"),
            "action": result.get("action"),
            "message": result.get("message"),
        },
    }
