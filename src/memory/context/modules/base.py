"""Base interface for pluggable context modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.memory.context.contracts import (
    ContextModuleResult,
    HistoryImageCandidate,
    MemorySourceBundle,
    TaskContextInput,
)


class ContextModule(ABC):
    name: str
    priority: int = 50
    max_items: int = 5

    def configure(self, options: dict[str, Any]) -> None:
        """Apply module-owned JSON options; the base contract accepts none."""
        if options:
            raise ValueError(f"{self.name} does not support options: {sorted(options)}")

    def validate_history_policy(self, config: dict[str, Any]) -> None:
        """Optional extension; the base contract supports full-history mode."""
        if set(config) - {"mode"} or config.get("mode", "all") != "all":
            raise ValueError(f"{self.name} does not support history policy: {config}")

    def select_history_images(
        self, candidates: tuple[HistoryImageCandidate, ...], config: dict[str, Any]
    ) -> tuple[str, ...]:
        """Pure selection hook; do not read files or modify module/runtime state."""
        self.validate_history_policy(config)
        return tuple(item.observation_id for item in candidates)

    @abstractmethod
    def build(
        self, sources: MemorySourceBundle, stage_context: TaskContextInput
    ) -> ContextModuleResult:
        """Return separate model-facing data and human-facing provenance."""

    @abstractmethod
    def describe_output(self) -> dict[str, Any]:
        """Describe model fields, audit fields and semantic evidence."""
