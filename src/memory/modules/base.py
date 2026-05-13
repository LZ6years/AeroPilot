"""Base interface for pluggable context modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.memory.schemas import ContextFragment, MemorySourceBundle


class ContextModule(ABC):
    name: str
    priority: int = 50
    max_items: int = 5

    @abstractmethod
    def build(self, sources: MemorySourceBundle, task_context: dict[str, Any]) -> ContextFragment:
        """Return one auditable context fragment."""
