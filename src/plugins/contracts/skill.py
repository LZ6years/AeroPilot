"""Contracts for plugin-provided skills and optional callable capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.tools.schemas import ToolRuntimePolicy
from src.tools.diagnostics import Diagnostic

SKILL_ARTIFACT_REFS_KEY = "_flyagent_artifact_refs"
SKILL_MODEL_INPUTS_KEY = "_flyagent_model_inputs"


@dataclass(frozen=True)
class SkillCapabilityResult:
    """Typed handler result with compact data and optional visual evidence."""

    message: str
    data: dict[str, Any]
    artifact_refs: tuple[str, ...] = ()
    model_inputs: tuple[dict[str, Any], ...] = ()


class SkillCapabilityFailure(Exception):
    """Capability failure that retains safe evidence for model correction."""

    def __init__(
        self,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        artifact_refs: tuple[str, ...] = (),
        model_inputs: tuple[dict[str, Any], ...] = (),
        diagnostic: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.data = dict(data or {})
        self.artifact_refs = tuple(artifact_refs)
        self.model_inputs = tuple(dict(item) for item in model_inputs)
        self.diagnostic = Diagnostic(**diagnostic).to_dict() if diagnostic is not None else None


@dataclass(frozen=True)
class SkillCapability:
    call_name: str
    wire_name: str
    description: str
    input_schema: dict[str, Any]
    runtime: ToolRuntimePolicy
    handler: str
    handler_path: Path
    mock: bool = False
    settings: dict[str, Any] = field(default_factory=dict)

    def listing(self) -> dict[str, Any]:
        return {
            "call_name": self.call_name,
            "wire_name": self.wire_name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "runtime": self.runtime.to_dict(),
            "handler": self.handler,
            "mock": self.mock,
        }


@dataclass(frozen=True)
class Skill:
    plugin_id: str
    name: str
    description: str
    summary: str
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
            "summary": self.summary,
            "source_path": str(self.skill_path),
            "callable": self.capability is not None,
        }
        if self.capability is not None:
            data["capability"] = self.capability.listing()
        return data

    def index_listing(self) -> dict[str, Any]:
        """Return the Hermes-style lightweight listing used for skill selection."""

        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "summary": self.summary,
            "references": self.reference_manifest(),
            "callable": self.capability is not None,
        }
        return {key: value for key, value in data.items() if value not in (None, "", [], {})}

    def reference_manifest(self) -> list[dict[str, str]]:
        """List reference files without loading their text."""

        return [
            {
                "path": _relative_to_skill(self, path),
                "description": _reference_description(path),
            }
            for path in self.reference_paths
        ]


def _relative_to_skill(skill: Skill, path: Path) -> str:
    try:
        return str(path.relative_to(skill.skill_dir)).replace("\\", "/")
    except ValueError:
        return str(path)


def _reference_description(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    if not stem:
        return "Optional skill reference."
    return f"Optional reference for {stem}."
