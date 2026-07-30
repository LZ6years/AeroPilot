"""Compact pose context module."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput
from src.uav import clean_pose_for_model, pose_delta


class DefaultPoseContextModule(ContextModule):
    name = "pose"
    priority = 95
    max_items = 3

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
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
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "latest_pose": latest_pose,
                "recent_pose_delta": recent_delta,
            },
            risks=[] if latest_pose else [{"type": "pose_missing", "source": self.__class__.__name__}],
            confidence="high" if latest_pose else "low",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=90, max_items=self.max_items),
            metadata={"reset_epoch": memory.reset_epoch, "pose_history_count": len(poses)},
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Latest UAV pose and recent pose delta.",
            "semantic_evidence": [
                "UAV pose: self position and orientation plus recent pose change; not target-relative geometry."
            ],
            "content_fields": {
                "latest_pose": "Most recent model-facing pose from pose refs; raw velocity is intentionally removed.",
                "recent_pose_delta": "Difference between the last two poses as dx, dy, dz, and normalized dyaw_deg when available.",
            },
            "metadata_fields": {
                "pose_history_count": "Number of recent pose refs considered before compacting the planner-facing pose context.",
            },
            "risk_fields": {
                "pose_missing": "No recent pose is available; avoid precise movement decisions.",
            },
            "coordinate_note": "Pose values follow the runtime environment coordinate convention, currently AirSim NED for map-related context.",
        }
