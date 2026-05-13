"""Lightweight map memory store for MVP recall."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MapMemoryStore:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, entry: dict[str, Any]) -> None:
        self.entries.append(dict(entry))

    def recall(self, map_id: str, scene_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        matches = []
        for entry in self.entries:
            if entry.get("map_id") != map_id:
                continue
            if scene_id is not None and entry.get("scene_id") not in (None, scene_id):
                continue
            matches.append(dict(entry))
        return matches[-limit:]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
