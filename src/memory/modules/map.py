"""Map memory recall context module."""

from __future__ import annotations

from typing import Any

from src.memory.modules.base import ContextModule
from src.memory.schemas import BudgetSpec, ContextFragment, MemorySourceBundle


class DefaultMapContextModule(ContextModule):
    name = "map"
    priority = 60
    max_items = 5

    def build(self, sources: MemorySourceBundle, task_context: dict[str, Any]) -> ContextFragment:
        memory = sources.flight_memory
        recalled = []
        if sources.map_memory is not None:
            recalled = sources.map_memory.recall(memory.map_id, memory.scene_instance_id, limit=self.max_items)
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={"recalled_map_memories": recalled},
            inferences=[
                {"type": "map_memory_hint", "value": item, "source": self.__class__.__name__}
                for item in recalled
            ],
            risks=[
                {"type": "map_memory_requires_reconfirmation", "source": self.__class__.__name__}
            ]
            if recalled
            else [],
            confidence="low" if recalled else "medium",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=220, max_items=self.max_items),
            metadata={
                "map_id": memory.map_id,
                "scene_instance_id": memory.scene_instance_id,
                "coordinate_frame": "airsim_ned",
            },
        )
