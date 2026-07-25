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
        actions = reader.read_refs(memory.recent_action_refs) if reader else []
        successful_actions = [item for item in actions if item.get("result", {}).get("ok") is True]
        recent_execution = [_compact_action(item) for item in successful_actions[-self.max_items :]]
        motion_history_summary = _motion_history_summary(
            recent_execution,
            stage_context.phase_execution_summary,
        )
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
            "summary": "Current-phase UAV motion and recent successful actions in execution order.",
            "content_fields": {
                "motion_history_summary": (
                    "Current-phase UAV displacement plus up to five recent successful actions, "
                    "ordered oldest to newest. It is history evidence, not the current plan or target distance."
                ),
            },
            "risk_fields": {
                "recent_action_failure": "Most recent failed action record, if any.",
            },
            "metadata_fields": {
                "motion_history_summary_sources": "Debug-only source notes for the compact motion summary.",
            },
        }


def _compact_action(action: dict[str, Any]) -> dict[str, Any]:
    tool = action.get("tool")
    return {
        "tool": tool,
        "args": _model_facing_action_args(tool, action.get("args", {})),
    }


def _model_facing_action_args(tool: Any, args: Any) -> dict[str, Any]:
    """Remove legacy named-motion shorthands from model-facing history.

    The command layer still accepts old named motions for manual/internal
    compatibility, but planner context should teach the model only the numeric
    velocity contract.
    """

    if not isinstance(args, dict):
        return {}
    compact = dict(args)
    if tool == "uav.move_body" and "direction" in compact:
        compact.pop("direction", None)
        compact["legacy_named_motion"] = True
    return compact


def _motion_history_summary(
    recent_execution: list[dict[str, Any]],
    phase_execution_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    phase_summary = phase_execution_summary if isinstance(phase_execution_summary, dict) else {}
    measurements = phase_summary.get("measurements", {})
    if not isinstance(measurements, dict):
        measurements = {}

    return {
        "window_count": _safe_int(phase_summary.get("window_count")),
        "phase_start_relative_uav_motion": {
            key: _safe_float(measurements.get(key))
            for key in ("forward_m", "right_m", "up_m", "planar_m", "yaw_delta_deg")
        },
        "recent_execution_order": "oldest_to_newest",
        "recent_execution": list(recent_execution),
    }


def _safe_float(value: Any) -> float:
    try:
        return round(float(value or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
