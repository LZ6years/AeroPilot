"""Optional adapter for measured UAV motion in the current phase.

Exact model intent, tool calls, failures, and results belong to the canonical
AgentTranscript. This module intentionally exposes only state derived from
pose measurements.
"""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput


class DefaultActionContextModule(ContextModule):
    name = "action"
    priority = 90
    max_items = 5

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        motion_history_summary = _motion_history_summary(
            stage_context.phase_execution_summary,
        )
        visual_search_progress = _visual_search_progress(
            stage_context.safety_constraints
        )
        content = {
            "motion_history_summary": motion_history_summary,
            "field_semantics": {
                "phase_start_relative_uav_motion": (
                    "Cumulative UAV self-motion since the current phase "
                    "began, expressed in the phase-start body frame; it is "
                    "not target-relative distance or bearing."
                ),
                "last_window_uav_motion": (
                    "UAV self-motion during the most recently completed action window."
                ),
                "forward_m": "Positive is forward from the phase-start UAV heading.",
                "right_m": "Positive is right from the phase-start UAV heading.",
                "up_m": "Positive is upward.",
                "yaw_delta_deg": (
                    "Cumulative UAV heading change; it does not indicate "
                    "where a target is."
                ),
            },
        }
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content=content,
            confidence="high",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=350, max_items=self.max_items),
            metadata={
                "motion_history_summary_sources": {
                    "phase_scope": "stage_context.phase_execution_summary when provided",
                    "recent_motion": "stage_context.phase_execution_summary.last_window",
                },
                # Search budgets and recharge counters are Runtime policy
                # internals. Preserve them for review without asking the model
                # to interpret them as task progress.
                "visual_search_progress": visual_search_progress,
            },
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Current-phase and last-window measured UAV self-motion.",
            "semantic_evidence": [
                "Measured phase-relative UAV self-motion; not target distance."
            ],
            "content_fields": {
                "motion_history_summary": (
                    "Current-phase and last-window UAV displacement. It is self-motion, not "
                    "the current plan, tool transcript, or target distance."
                ),
                "field_semantics": "Compact meanings for ambiguous self-motion fields.",
            },
            "risk_fields": {},
            "metadata_fields": {
                "motion_history_summary_sources": "Debug-only source notes for the compact motion summary.",
                "visual_search_progress": "Debug-only Runtime yaw-search budget and recharge state.",
            },
        }


def _motion_history_summary(
    phase_execution_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    phase_summary = phase_execution_summary if isinstance(phase_execution_summary, dict) else {}
    measurements = phase_summary.get("measurements", {})
    if not isinstance(measurements, dict):
        measurements = {}
    last_window = phase_summary.get("last_window", {})
    if not isinstance(last_window, dict):
        last_window = {}

    return {
        "phase_start_relative_uav_motion": {
            key: _safe_float(measurements.get(key))
            for key in ("forward_m", "right_m", "up_m", "planar_m", "yaw_delta_deg")
        },
        "last_window_uav_motion": {
            key: _safe_float(last_window.get(key))
            for key in ("forward_m", "right_m", "up_m", "planar_m", "yaw_delta_deg")
        },
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
