"""Contracts for plugin-provided skills and optional callable capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillCapability:
    call_name: str
    description: str
    input_schema: dict[str, Any]
    handler: str
    handler_path: Path
    mock: bool = False

    def listing(self) -> dict[str, Any]:
        return {
            "call_name": self.call_name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "handler": self.handler,
            "mock": self.mock,
        }


@dataclass(frozen=True)
class Skill:
    plugin_id: str
    name: str
    description: str
    skill_dir: Path
    skill_path: Path
    reference_paths: tuple[Path, ...] = ()
    script_paths: tuple[Path, ...] = ()
    body: str = ""
    capability: SkillCapability | None = None

    @property
    def source_path(self) -> Path:
        return self.skill_path

    def listing(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "description": self.description,
            "source_path": str(self.skill_path),
            "callable": self.capability is not None,
        }
        if self.capability is not None:
            data["capability"] = self.capability.listing()
        return data
