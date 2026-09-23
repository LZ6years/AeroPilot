"""Optional adapter for measured UAV motion in the current phase.

Exact model intent, tool calls, failures, and results belong to the canonical
AgentTranscript. This module intentionally exposes only state derived from
pose measurements.
"""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, TaskContextInput


class DefaultActionContextModule(ContextModule):
    name = "action"
    priority = 90
    max_items = 5

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextFragment:
        summary = stage_context.motion_summary
        content = {
            "instruction_start_relative_uav_motion": {
                key: _optional_float(summary.get("measurements", {}).get(key))
                for key in ("forward_m", "right_m", "up_m", "planar_m", "yaw_delta_deg")},
            "last_window_uav_motion": {
                key: _optional_float(summary.get("last_window", {}).get(key))
                for key in ("forward_m", "right_m", "up_m", "planar_m", "yaw_delta_deg")},
            "field_semantics": (
                "Measured UAV self-motion, not target distance. Cumulative motion is in the "
                "instruction-start body frame; last-window motion uses that batch's starting body frame. "
                "Unknown measurements remain null."
            ),
        }
        return ContextFragment(
            name=self.name, enabled=True, source=self.__class__.__name__, content=content,
            confidence="high", budget=BudgetSpec(priority=self.priority),
            metadata={"visual_search_progress": _visual_search_progress(stage_context.safety_constraints)},
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Measured instruction-relative and last-window UAV self-motion.",
            "semantic_evidence": ["Measured instruction-relative UAV self-motion, not distance to a target."],
            "content_fields": {
                "instruction_start_relative_uav_motion": "Cumulative measured motion from instruction origin.",
                "last_window_uav_motion": "Measured motion across the latest physical batch.",
                "field_semantics": "Self-motion is not target-relative distance.",
            },
            "metadata_fields": {"visual_search_progress": "Safety search budget, audit only."},
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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


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
