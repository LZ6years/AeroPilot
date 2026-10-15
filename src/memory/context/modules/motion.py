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
        }
        return ContextModuleResult(
            ContextFragment(self.name, content),
            FragmentAudit(
                name=self.name,
                source=self.__class__.__name__,
                confidence="high",
                metadata={
                    "physical_window_count": stage_context.metadata.get("physical_window_count"),
                    "safety_constraints": dict(stage_context.safety_constraints),
                    # Acquisition modalities do not imply model-visible images.
                    "observation_policy": dict(stage_context.observation_policy),
                },
            ),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Measured instruction-relative and last-window UAV self-motion.",
            "semantic_evidence": [
                "Measured instruction-relative UAV self-motion, not distance to a target."
            ],
            "model_fields": {
                "instruction_start_relative_uav_motion": (
                    "Net measured UAV self-motion in the instruction-start body frame, "
                    "not target distance; distances in meters, yaw_delta_deg in degrees. "
                    "planar_m is net horizontal displacement, not travelled path length; "
                    "yaw_delta_deg is heading difference, not accumulated turning. "
                    "Unknown measurements remain null."
                ),
                "last_window_uav_motion": (
                    "Measured UAV self-motion across the latest single-action window in that window's "
                    "starting body frame, not target distance; distances in meters, "
                    "yaw_delta_deg in degrees. Unknown measurements remain null."
                ),
            },
            "audit_fields": {
                "physical_window_count": "Number of completed physical windows.",
                "safety_constraints": "Runtime-only dynamic flight and stationary-scan diagnostics.",
                "observation_policy": "Acquisition timing and modalities, not a declaration of attached model inputs.",
            },
        }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None
