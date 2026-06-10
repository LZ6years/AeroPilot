"""Compact primitive action history context module."""

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
        failed = [item for item in actions if item.get("result", {}).get("ok") is False]
        risks = [
            {
                "type": "recent_action_failure",
                "value": _compact_action(failed[-1]),
                "source": self.__class__.__name__,
            }
        ] if failed else []
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "recent_actions": compact_actions,
                "action_trend": _action_trend(compact_actions),
            },
            risks=risks,
            confidence="high",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=150, max_items=self.max_items),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Recent primitive actions and compact motion trend.",
            "content_fields": {
                "recent_actions": "Up to five recent actions with tool, args, and ok status.",
                "recent_actions[].ok": "Whether the tool execution reported success.",
                "action_trend": "Lightweight summary of recent tool sequence, repetition, and broad motion pattern. It describes history only and is not a task-completion fact.",
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
        "ok": result.get("ok"),
    }


def _action_trend(actions: list[dict[str, Any]]) -> dict[str, Any]:
    tools = [str(item.get("tool") or "") for item in actions if item.get("tool")]
    return {
        "tool_sequence": tools,
        "repetition": _repetition(tools),
        "motion_pattern": _motion_pattern(actions),
    }


def _repetition(tools: list[str]) -> dict[str, Any]:
    if not tools:
        return {"last_tool": None, "consecutive_count": 0}
    last_tool = tools[-1]
    count = 0
    for tool in reversed(tools):
        if tool != last_tool:
            break
        count += 1
    return {"last_tool": last_tool, "consecutive_count": count}


def _motion_pattern(actions: list[dict[str, Any]]) -> dict[str, str]:
    if len(actions) < 2:
        return {"type": "none", "summary": "Not enough recent actions to identify a motion pattern.", "confidence": "low"}
    yaw_values = [
        _numeric_arg(item, "yaw_deg")
        for item in actions
        if item.get("tool") == "uav.turn_relative"
    ]
    move_directions = [
        str(item.get("args", {}).get("direction"))
        for item in actions
        if item.get("tool") == "uav.move_body" and item.get("args", {}).get("direction")
    ]
    if len(yaw_values) >= 3 and _alternates_sign(yaw_values[-3:]):
        return {
            "type": "alternating_yaw",
            "summary": "Recent turn_relative yaw commands alternate between positive and negative corrections.",
            "confidence": "high",
        }
    if len(yaw_values) >= 2 and all(value > 0 for value in yaw_values[-2:]):
        return {"type": "repeated_yaw", "summary": "Recent yaw commands repeatedly turn positive.", "confidence": "medium"}
    if len(yaw_values) >= 2 and all(value < 0 for value in yaw_values[-2:]):
        return {"type": "repeated_yaw", "summary": "Recent yaw commands repeatedly turn negative.", "confidence": "medium"}
    if len(move_directions) >= 2 and len(set(move_directions[-2:])) == 1:
        return {
            "type": "repeated_move",
            "summary": f"Recent move_body commands repeatedly move {move_directions[-1]}.",
            "confidence": "medium",
        }
    if len(set(item.get("tool") for item in actions if item.get("tool"))) > 1:
        return {"type": "mixed", "summary": "Recent actions mix multiple tool types.", "confidence": "medium"}
    return {"type": "none", "summary": "No clear repeated or alternating motion pattern detected.", "confidence": "low"}


def _numeric_arg(action: dict[str, Any], name: str) -> float:
    try:
        return float(action.get("args", {}).get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _alternates_sign(values: list[float]) -> bool:
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in values]
    if any(sign == 0 for sign in signs):
        return False
    return all(left != right for left, right in zip(signs, signs[1:]))
