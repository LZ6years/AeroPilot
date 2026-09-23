"""Map memory recall context module."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, TaskContextInput


class DefaultMapContextModule(ContextModule):
    name = "map"
    priority = 60
    max_items = 5

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextFragment:
        memory = sources.flight_memory
        recalled = []
        if sources.map_memory is not None:
            recalled = sources.map_memory.recall(memory.map_id, memory.scene_instance_id, limit=self.max_items)
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "recalled_map_memories": recalled,
                "recall_filter": {
                    "map_id": memory.map_id,
                    "scene_instance_id": memory.scene_instance_id,
                    "limit": self.max_items,
                },
            },
            inferences=[
                {"type": "map_memory_hint", "value": item, "source": self.__class__.__name__}
                for item in recalled
            ],
            risks=[
                {
                    "type": "map_memory_requires_reconfirmation",
                    "value": [
                        item.get("landmark_description") or item.get("region_hint")
                        for item in recalled
                        if item.get("requires_reconfirmation", True)
                    ],
                    "source": self.__class__.__name__,
                }
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

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Map or scene memory clues recalled for the current map and scene.",
            "semantic_evidence": [
                "Map memory: recalled scene clues requiring current confirmation; not current observation facts."
            ],
            "content_fields": {
                "recalled_map_memories": "Map memory entries matching the current map/scene filter; use as clues, not current visual facts.",
                "recall_filter": "The map_id, scene_instance_id, and limit used for map memory recall.",
            },
            "inference_fields": {
                "map_memory_hint": "A recalled map clue that may help search or orientation but should be confirmed against current observations.",
            },
            "risk_fields": {
                "map_memory_requires_reconfirmation": "Recalled clues that should be reconfirmed before high-confidence or risky movement.",
            },
            "metadata_fields": {
                "map_id": "Current map identifier.",
                "scene_instance_id": "Current scene instance identifier.",
                "coordinate_frame": "Coordinate convention used by map-related hints.",
            },
        }
