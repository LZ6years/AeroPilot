"""Compact primitive action history context module."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput


class DefaultActionContextModule(ContextModule):
    name = "action"
    priority = 90
    max_items = 5
    exposed_recent_actions = 2

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        memory = sources.flight_memory
        reader = memory.memory_reader
        actions = reader.read_refs(memory.recent_action_refs[-self.max_items :]) if reader else []
        compact_actions = [_compact_action(item) for item in actions]
        exposed_actions = compact_actions[-self.exposed_recent_actions :]
        motion_history_summary = _motion_history_summary(compact_actions, stage_context.phase_execution_summary)
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
                "recent_actions": exposed_actions,
                "action_trend": _action_trend(compact_actions),
                "motion_history_summary": motion_history_summary,
            },
            risks=risks,
            confidence="high",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=180, max_items=self.max_items),
            metadata={
                "motion_history_summary_sources": {
                    "phase_scope": "stage_context.phase_execution_summary when provided",
                    "recent_motion": "recent action refs",
                }
            },
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Recent primitive actions and compact motion trend.",
            "content_fields": {
                "recent_actions": "Up to two history-only recent action examples with tool, args, ok status, and history_only=true. They are not the current plan and not visual evidence.",
                "recent_actions[].ok": "Whether the tool execution reported success.",
                "action_trend": "Lightweight summary of recent tool sequence, repetition, and broad motion pattern. It describes history only and is not a task-completion fact.",
                "motion_history_summary": "History-only compact motion summary for the current phase when available plus recent yaw behavior. It helps detect repeated or alternating corrections without exposing raw action logs.",
            },
            "risk_fields": {
                "recent_action_failure": "Most recent failed action record, if any.",
            },
            "metadata_fields": {
                "motion_history_summary_sources": "Debug-only source notes for the compact motion summary.",
            },
        }


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    result = action.get("result", {}) if isinstance(action, dict) else {}
    return {
        "tool": action.get("tool"),
        "args": dict(action.get("args", {})),
        "ok": result.get("ok"),
        "history_only": True,
    }


def _action_trend(actions: list[dict[str, Any]]) -> dict[str, Any]:
    tools = [str(item.get("tool") or "") for item in actions if item.get("tool")]
    return {
        "tool_sequence": tools,
        "repetition": _repetition(tools),
        "motion_pattern": _motion_pattern(actions),
    }


def _motion_history_summary(
    actions: list[dict[str, Any]],
    phase_execution_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    phase_summary = phase_execution_summary if isinstance(phase_execution_summary, dict) else {}
    tool_facts = phase_summary.get("tool_facts", {})
    if not isinstance(tool_facts, dict):
        tool_facts = {}
    measurements = phase_summary.get("measurements", {})
    if not isinstance(measurements, dict):
        measurements = {}

    yaw_values = [
        _numeric_arg(item, "yaw_deg")
        for item in actions
        if item.get("tool") == "uav.turn_relative"
    ]
    tools_executed = tool_facts.get("tools_executed", {})
    if not isinstance(tools_executed, dict):
        tools_executed = {}

    return {
        "scope": "current_phase_when_available",
        "phase": phase_summary.get("phase"),
        "window_count": phase_summary.get("window_count"),
        "turn_count": tools_executed.get("uav.turn_relative"),
        "net_yaw_deg": _phase_yaw_delta(measurements, tool_facts),
        "recent_yaw_deg": yaw_values,
        "recent_abs_yaw_sum_deg": round(sum(abs(value) for value in yaw_values), 3),
        "recent_yaw_sign_changes": _sign_change_count(yaw_values),
        "recent_yaw_pattern": _yaw_pattern(yaw_values),
        "history_only": True,
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
    if len(yaw_values) >= 3 and _alternates_sign(yaw_values):
        return {
            "type": "alternating_yaw",
            "summary": "Recent turn_relative yaw commands alternate between positive and negative corrections.",
            "confidence": "high",
        }
    nonzero_yaw = [value for value in yaw_values if value != 0]
    if len(nonzero_yaw) >= 2 and all(value > 0 for value in nonzero_yaw):
        return {"type": "repeated_yaw", "summary": "Recent yaw commands repeatedly turn positive.", "confidence": "medium"}
    if len(nonzero_yaw) >= 2 and all(value < 0 for value in nonzero_yaw):
        return {"type": "repeated_yaw", "summary": "Recent yaw commands repeatedly turn negative.", "confidence": "medium"}
    if len(nonzero_yaw) >= 2:
        return {
            "type": "mixed",
            "summary": "Recent yaw commands include both positive and negative turns; treat them as history-only mixed corrections.",
            "confidence": "medium",
        }
    if len(move_directions) >= 2 and len(set(move_directions[-2:])) == 1:
        return {
            "type": "repeated_move",
            "summary": f"Recent move_body commands repeatedly move {move_directions[-1]}.",
            "confidence": "medium",
        }
    if len(set(item.get("tool") for item in actions if item.get("tool"))) > 1:
        return {"type": "mixed", "summary": "Recent actions mix multiple tool types.", "confidence": "medium"}
    return {"type": "none", "summary": "No clear repeated or alternating motion pattern detected.", "confidence": "low"}


def _phase_yaw_delta(measurements: dict[str, Any], tool_facts: dict[str, Any]) -> float | None:
    for source, key in ((measurements, "yaw_delta_deg"), (tool_facts, "turn_yaw_success_sum_deg")):
        value = source.get(key)
        if value is None:
            continue
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            continue
    return None


def _sign_change_count(values: list[float]) -> int:
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in values]
    nonzero = [sign for sign in signs if sign != 0]
    return sum(1 for left, right in zip(nonzero, nonzero[1:]) if left != right)


def _yaw_pattern(values: list[float]) -> str:
    nonzero = [value for value in values if value != 0]
    if len(nonzero) < 2:
        return "none"
    if _alternates_sign(nonzero):
        return "alternating_yaw"
    if all(value > 0 for value in nonzero):
        return "repeated_positive_yaw"
    if all(value < 0 for value in nonzero):
        return "repeated_negative_yaw"
    return "mixed"


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
