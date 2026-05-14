"""Base interface for pluggable context modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.memory.context.schemas import ContextFragment, MemorySourceBundle, StageContextInput


class ContextModule(ABC):
    name: str
    priority: int = 50
    max_items: int = 5

    @abstractmethod
    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        """Return one auditable context fragment."""

    @abstractmethod
    def describe_output(self) -> dict[str, Any]:
        """Return model-facing descriptions for this module's fragment fields."""
