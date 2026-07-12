"""Generate compact model-facing recommendations for runtime control actions.

This module belongs to the runtime-control layer. It turns already-available
runtime evidence into short recommendations for the planner without changing
UAV tool execution, phase judgment, safety checks, or reusable skills.
"""

from __future__ import annotations

from typing import Any


SEMANTIC_PLAN_UPDATE_RECOMMENDATION = (
    "review_now: multiple action-observation cycles completed; request only if confidence in "
    "goal/done_when/continue_when/metrics is now low"
)
MIN_PHASE_WINDOW_COUNT = 4
MIN_TOTAL_TOOL_COUNT = 4


def runtime_control_recommendations(
    *,
    available: list[str],
    status: dict[str, str] | None = None,
    phase_execution_summary: dict[str, Any] | None = None,
    completed: bool = False,
    stopped: bool = False,
) -> dict[str, str]:
    """Return short recommendations for currently available runtime control actions.

    Runtime only decides when enough action-observation history exists to make a
    semantic-contract review useful. The planner remains responsible for deciding
    whether confidence in the current contract is actually low enough to request
    an update.
    """

    if "semantic_plan_update" not in available:
        return {}
    if completed or stopped:
        return {}
    if (status or {}).get("semantic_plan_update") == "pending":
        return {}
    if not isinstance(phase_execution_summary, dict):
        return {}
    if _safe_int(phase_execution_summary.get("window_count")) < MIN_PHASE_WINDOW_COUNT:
        return {}
    if _total_tool_count(phase_execution_summary) < MIN_TOTAL_TOOL_COUNT:
        return {}
    return {"semantic_plan_update": SEMANTIC_PLAN_UPDATE_RECOMMENDATION}


def _total_tool_count(phase_execution_summary: dict[str, Any]) -> int:
    tool_facts = phase_execution_summary.get("tool_facts", {})
    if not isinstance(tool_facts, dict):
        return 0
    tools_executed = tool_facts.get("tools_executed", {})
    if not isinstance(tools_executed, dict):
        return 0
    return sum(_safe_int(count) for count in tools_executed.values())


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0
