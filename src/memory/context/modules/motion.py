"""Expose measured UAV self-motion without replaying action history."""

from __future__ import annotations

from typing import Any

from src.memory.context.contracts import (
    ContextFragment,
    ContextModuleResult,
    FragmentAudit,
    MemorySourceBundle,
    TaskContextInput,
)
from src.memory.context.modules.base import ContextModule


class DefaultMotionContextModule(ContextModule):
    name = "motion"
    priority = 90

    def build(
        self, sources: MemorySourceBundle, stage_context: TaskContextInput
    ) -> ContextModuleResult:
        summary = stage_context.motion_summary
        content = {
            "instruction_start_relative_uav_motion": {
                key: _optional_float(summary.get("measurements", {}).get(key))
                for key in ("forward_m", "right_m", "up_m", "planar_m", "yaw_delta_deg")
            },
            "last_window_uav_motion": {
                key: _optional_float(summary.get("last_window", {}).get(key))
                for key in ("forward_m", "right_m", "up_m", "planar_m", "yaw_delta_deg")
            },
            "field_semantics": (
                "Measured UAV self-motion, not target distance. Cumulative motion is in the "
                "instruction-start body frame; last-window motion uses that batch's starting "
                "body frame. Unknown measurements remain null."
            ),
            "safety_constraints": dict(stage_context.safety_constraints),
            "observation_policy": dict(stage_context.observation_policy),
        }
        return ContextModuleResult(
            ContextFragment(self.name, content),
            FragmentAudit(
                name=self.name,
                source=self.__class__.__name__,
                confidence="high",
                metadata={"physical_window_count": stage_context.metadata.get("physical_window_count")},
            ),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Measured instruction-relative and last-window UAV self-motion.",
            "semantic_evidence": [
                "Measured instruction-relative UAV self-motion, not distance to a target."
            ],
            "model_fields": {
                "instruction_start_relative_uav_motion": "Cumulative measured motion from instruction origin.",
                "last_window_uav_motion": "Measured motion across the latest physical batch.",
                "field_semantics": "Self-motion is not target-relative distance.",
                "safety_constraints": "Current dynamic flight and stationary-scan constraints.",
                "observation_policy": "Automatic observation timing and requested modalities.",
            },
            "audit_fields": {
                "physical_window_count": "Number of completed physical windows."
            },
        }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None

