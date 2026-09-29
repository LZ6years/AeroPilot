"""Keep raw UAV pose for audit while exposing only decision-useful AGL."""

from __future__ import annotations

import math
from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.contracts import (
    ContextFragment,
    ContextModuleResult,
    FragmentAudit,
    MemorySourceBundle,
    TaskContextInput,
)
from src.uav import clean_pose_for_model, pose_delta


class DefaultPoseContextModule(ContextModule):
    name = "pose"
    priority = 95
    max_items = 3

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextModuleResult:
        memory = sources.flight_memory
        reader = memory.memory_reader
        pose_records = reader.read_refs(memory.recent_pose_refs[-self.max_items :]) if reader else []
        poses = [
            clean_pose_for_model(item.get("pose", {}))
            for item in pose_records
            if isinstance(item, dict) and isinstance(item.get("pose"), dict)
        ]
        latest_pose = poses[-1] if poses else {}
        recent_delta = pose_delta(poses[-2], poses[-1]) if len(poses) >= 2 else {}
        content = _model_pose_content(latest_pose)
        return ContextModuleResult(
            ContextFragment(name=self.name, model_content=content),
            FragmentAudit(
                name=self.name,
                source=self.__class__.__name__,
                confidence="high" if latest_pose else "low",
                warnings=() if latest_pose else ("pose_missing",),
                metadata={
                "reset_epoch": memory.reset_epoch,
                "pose_history_count": len(poses),
                # Absolute world pose is essential for trace/debug work but is
                # not target-relative evidence. Keep it out of model content.
                "latest_pose": latest_pose,
                "recent_pose_delta": recent_delta,
                },
            ),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Optional UAV above-ground clearance; absolute pose remains debug-only.",
            "semantic_evidence": [
                "UAV ground clearance when calibrated; self-state, not target-relative geometry."
            ],
            "model_fields": {
                "agl_m": "Optional vertical clearance above the configured local landing reference, in meters.",
                "field_semantics": "Compact meaning and evidence boundary for agl_m.",
            },
            "audit_fields": {
                "pose_history_count": "Number of recent pose refs considered before compacting the planner-facing pose context.",
                "latest_pose": "Debug-only latest canonical world pose; never target-relative evidence.",
                "recent_pose_delta": "Debug-only delta between the latest two canonical poses.",
            },
            "coordinate_note": "Raw coordinate-frame values remain in metadata and are not sent to the model.",
        }


def _model_pose_content(latest_pose: dict[str, Any]) -> dict[str, Any]:
    """Expose one optional scalar instead of ambiguous absolute world pose."""

    try:
        agl_m = float(latest_pose["agl_m"])
    except (KeyError, TypeError, ValueError):
        return {}
    if not math.isfinite(agl_m):
        return {}
    return {
        "agl_m": agl_m,
        "field_semantics": {
            "agl_m": (
                "UAV vertical clearance above the configured local landing "
                "reference, in meters; smaller means closer to ground. This "
                "is UAV self-state, not target distance."
            )
        },
    }
