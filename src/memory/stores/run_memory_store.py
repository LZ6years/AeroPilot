"""Placeholder run memory store for future retrospection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunMemoryStore:
    notes: list[dict[str, Any]] = field(default_factory=list)

    def add(self, note: dict[str, Any]) -> None:
        self.notes.append(dict(note))

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        return [dict(item) for item in self.notes[-limit:]]
