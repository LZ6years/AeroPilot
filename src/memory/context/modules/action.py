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
        visual_search_progress = _visual_search_progress(
            stage_context.safety_constraints
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
                "field_semantics": {
                    "phase_start_relative_uav_motion": (
                        "Cumulative UAV self-motion since the current phase "
                        "began, expressed in the phase-start body frame; it is "
                        "not target-relative distance or bearing."
                    ),
                    "forward_m": "Positive is forward from the phase-start UAV heading.",
                    "right_m": "Positive is right from the phase-start UAV heading.",
                    "up_m": "Positive is upward.",
                    "yaw_delta_deg": (
                        "Cumulative UAV heading change; it does not indicate "
                        "where a target is."
                    ),
                },
            },
            risks=risks,
            confidence="high",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=230, max_items=self.max_items),
            metadata={
                "motion_history_summary_sources": {
                    "phase_scope": "stage_context.phase_execution_summary when provided",
                    "recent_motion": "recent action refs",
                },
                # Search budgets and recharge counters are Runtime policy
                # internals. Preserve them for review without asking the model
                # to interpret them as task progress.
                "visual_search_progress": visual_search_progress,
            },
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Current-phase UAV motion and recent successful actions in execution order.",
            "semantic_evidence": [
                "Action history: phase-relative UAV self-motion and recent actions; not target distance."
            ],
            "content_fields": {
                "motion_history_summary": (
                    "Current-phase UAV displacement and up to five "
                    "recent successful actions ordered oldest to newest. It is self-motion history, not "
                    "the current plan or target distance."
                ),
                "field_semantics": "Compact meanings for ambiguous self-motion fields.",
            },
            "risk_fields": {
                "recent_action_failure": "Most recent failed action record, if any.",
            },
            "metadata_fields": {
                "motion_history_summary_sources": "Debug-only source notes for the compact motion summary.",
                "visual_search_progress": "Debug-only Runtime yaw-search budget and recharge state.",
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


def _visual_search_progress(
    safety_constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep search-policy state available to audit without model exposure."""

    safety_summary = safety_constraints if isinstance(safety_constraints, dict) else {}
    return {
        "scope": "current_stationary_viewpoint",
        "yaw_offset_deg": _safe_float(safety_summary.get("yaw_scan_offset_deg")),
        "yaw_extent_deg": _safe_float(safety_summary.get("yaw_scan_extent_deg")),
        "yaw_remaining_deg": _safe_float(safety_summary.get("yaw_scan_remaining_deg")),
        "turn_count": _safe_int(safety_summary.get("yaw_scan_turn_count")),
        "recharge_count": _safe_int(safety_summary.get("yaw_scan_recharge_count")),
        "recharge_distance_m": _safe_float(
            safety_summary.get("yaw_scan_recharge_distance_m")
        ),
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
