"""Pose history context module."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput


class DefaultPoseContextModule(ContextModule):
    name = "pose"
    priority = 95
    max_items = 3

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        memory = sources.flight_memory
        reader = memory.memory_reader
        pose_records = reader.read_refs(memory.recent_pose_refs[-self.max_items :]) if reader else []
        poses = [
            dict(item.get("pose", {}))
            for item in pose_records
            if isinstance(item, dict) and isinstance(item.get("pose"), dict)
        ]
        latest_pose = poses[-1] if poses else {}
        pose_delta = _pose_delta(poses[-2], poses[-1]) if len(poses) >= 2 else {}
        facts = []
        if latest_pose:
            facts.append({"type": "latest_pose", "value": latest_pose, "source": self.__class__.__name__})
        inferences = []
        if pose_delta:
            inferences.append({"type": "recent_pose_delta", "value": pose_delta, "source": self.__class__.__name__})
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "latest_pose": latest_pose,
                "recent_pose_history": poses,
                "recent_pose_delta": pose_delta,
            },
            facts=facts,
            inferences=inferences,
            risks=[] if latest_pose else [{"type": "pose_missing", "source": self.__class__.__name__}],
            confidence="high" if latest_pose else "low",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=180, max_items=self.max_items),
            metadata={"reset_epoch": memory.reset_epoch},
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Latest UAV pose, short pose history, and recent pose delta.",
            "content_fields": {
                "latest_pose": "Most recent pose from pose refs; expected fields include x, y, z, and yaw_deg when available.",
                "recent_pose_history": "Up to three recent pose records in chronological order.",
                "recent_pose_delta": "Difference between the last two poses for x, y, z, and yaw_deg when available.",
            },
            "fact_fields": {
                "latest_pose": "Observed latest pose and the strongest pose fact for planning.",
            },
            "inference_fields": {
                "recent_pose_delta": "Motion delta inferred from recent pose history; useful for checking whether recent actions changed state.",
            },
            "risk_fields": {
                "pose_missing": "No recent pose is available; avoid precise movement decisions.",
            },
            "coordinate_note": "Pose values follow the runtime environment coordinate convention, currently AirSim NED for map-related context.",
        }


def _pose_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = ("x", "y", "z", "yaw_deg")
    return {
        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in keys
        if key in before or key in after
    }
