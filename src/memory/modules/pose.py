"""Pose history context module."""

from __future__ import annotations

from typing import Any

from src.memory.modules.base import ContextModule
from src.memory.schemas import BudgetSpec, ContextFragment, MemorySourceBundle


class DefaultPoseContextModule(ContextModule):
    name = "pose"
    priority = 95
    max_items = 5

    def build(self, sources: MemorySourceBundle, task_context: dict[str, Any]) -> ContextFragment:
        memory = sources.flight_memory
        poses = [
            dict(item.get("pose", {}))
            for item in memory.recent_observations[-self.max_items :]
            if isinstance(item, dict) and isinstance(item.get("pose"), dict)
        ]
        latest_pose = poses[-1] if poses else {}
        facts = []
        if latest_pose:
            facts.append({"type": "latest_pose", "value": latest_pose, "source": self.__class__.__name__})
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={"latest_pose": latest_pose, "recent_pose_history": poses},
            facts=facts,
            risks=[] if latest_pose else [{"type": "pose_missing", "source": self.__class__.__name__}],
            confidence="high" if latest_pose else "low",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=180, max_items=self.max_items),
            metadata={"reset_epoch": memory.reset_epoch},
        )
