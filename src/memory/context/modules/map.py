"""Map memory recall context module."""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.contracts import (
    ContextFragment,
    ContextModuleResult,
    FragmentAudit,
    MemorySourceBundle,
    TaskContextInput,
)


class DefaultMapContextModule(ContextModule):
    name = "map"
    priority = 60
    max_items = 5

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextModuleResult:
        memory = sources.flight_memory
        recalled = []
        if sources.map_memory is not None:
            recalled = sources.map_memory.recall(memory.map_id, memory.scene_instance_id, limit=self.max_items)
        content = {
                "recalled_map_memories": recalled,
                "requires_reconfirmation": any(
                    item.get("requires_reconfirmation", True) for item in recalled
                ),
            }
        return ContextModuleResult(
            ContextFragment(self.name, content),
            FragmentAudit(
                name=self.name,
                source=self.__class__.__name__,
                confidence="low" if recalled else "medium",
                metadata={
                "map_id": memory.map_id,
                "scene_instance_id": memory.scene_instance_id,
                "coordinate_frame": "airsim_ned",
                "recall_limit": self.max_items,
                },
            ),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Map or scene memory clues recalled for the current map and scene.",
            "semantic_evidence": [
                "Map memory: recalled scene clues requiring current confirmation; not current observation facts."
            ],
            "model_fields": {
                "recalled_map_memories": "Map memory entries matching the current map/scene filter; use as clues, not current visual facts.",
                "requires_reconfirmation": "Whether any recalled clue requires current confirmation.",
            },
            "audit_fields": {
                "map_id": "Current map identifier.",
                "scene_instance_id": "Current scene instance identifier.",
                "coordinate_frame": "Coordinate convention used by map-related hints.",
                "recall_limit": "Maximum recalled entries considered.",
            },
        }
