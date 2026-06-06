"""Contracts for plugin-provided context modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContextModuleContribution:
    plugin_id: str
    slot: str
    module_path: str
    class_name: str
    source_path: Path
