"""Load bounded text and skill-local resources for selected skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.plugins.contracts.skill import Skill


DEFAULT_MAX_SKILL_CHARS = 2400
DEFAULT_MAX_REFERENCE_CHARS = 1600


@dataclass(frozen=True)
class SkillResourceLoader:
    max_reference_chars: int = DEFAULT_MAX_REFERENCE_CHARS
    load_reference_files: bool = True

    def load_references(self, skill: Skill) -> list[dict[str, Any]]:
        if not self.load_reference_files:
            return []
        return [
            _load_reference(skill, path, self.max_reference_chars)
            for path in skill.reference_paths
        ]

    def available_references(self, skill: Skill) -> list[str]:
        return [_relative_to_skill(skill, path) for path in skill.reference_paths]

    def script_assets(self, skill: Skill) -> list[str]:
        return [_relative_to_skill(skill, path) for path in skill.script_paths]


@dataclass(frozen=True)
class SkillContentLoader:
    max_skill_chars: int = DEFAULT_MAX_SKILL_CHARS
    resource_loader: SkillResourceLoader = SkillResourceLoader()

    def __init__(
        self,
        max_skill_chars: int = DEFAULT_MAX_SKILL_CHARS,
        max_reference_chars: int = DEFAULT_MAX_REFERENCE_CHARS,
        load_reference_files: bool = True,
        resource_loader: SkillResourceLoader | None = None,
    ) -> None:
        object.__setattr__(self, "max_skill_chars", max_skill_chars)
        object.__setattr__(
            self,
            "resource_loader",
            resource_loader
            or SkillResourceLoader(
                max_reference_chars=max_reference_chars,
                load_reference_files=load_reference_files,
            ),
        )

    def load_summary(self, skill: Skill) -> dict[str, Any]:
        resource_loader = self.resource_loader
        return {
            "plugin_id": skill.plugin_id,
            "name": skill.name,
            "description": skill.description,
            "source_path": str(skill.skill_path),
            "skill_dir": str(skill.skill_dir),
            "skill_text": skill.body[: self.max_skill_chars].strip(),
            "skill_text_truncated": len(skill.body) > self.max_skill_chars,
            "references": resource_loader.load_references(skill),
            "available_references": resource_loader.available_references(skill),
            "capability": skill.capability.listing() if skill.capability else None,
            "assets": {
                "scripts": resource_loader.script_assets(skill),
            },
        }


def _load_reference(skill: Skill, path, max_chars: int) -> dict[str, Any]:
    resolved = path.resolve()
    skill_root = skill.skill_dir.resolve()
    try:
        resolved.relative_to(skill_root)
    except ValueError:
        return {
            "path": str(path),
            "text": "",
            "truncated": False,
            "error": "reference_outside_skill_dir",
        }
    text = path.read_text(encoding="utf-8-sig")
    return {
        "path": _relative_to_skill(skill, path),
        "text": text[:max_chars].strip(),
        "truncated": len(text) > max_chars,
    }


def _relative_to_skill(skill: Skill, path) -> str:
    try:
        return str(path.relative_to(skill.skill_dir))
    except ValueError:
        return str(path)
