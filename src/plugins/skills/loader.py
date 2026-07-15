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

    def load_reference_paths(self, skill: Skill, requested_paths: list[str]) -> list[dict[str, Any]]:
        """Load only the selected skill-local references."""

        if not self.load_reference_files:
            return []
        by_relative_path = {
            _relative_to_skill(skill, path).replace("\\", "/"): path
            for path in skill.reference_paths
        }
        loaded: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_path in requested_paths:
            relative_path = str(raw_path).strip().replace("\\", "/")
            if not relative_path or relative_path in seen:
                continue
            seen.add(relative_path)
            path = by_relative_path.get(relative_path)
            if path is not None:
                loaded.append(_load_reference(skill, path, self.max_reference_chars))
        return loaded

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
            "summary": skill.summary,
            "source_path": str(skill.skill_path),
            "skill_dir": str(skill.skill_dir),
            "skill_text": skill.body[: self.max_skill_chars].strip(),
            "skill_text_truncated": len(skill.body) > self.max_skill_chars,
            "references": [],
            "available_references": resource_loader.available_references(skill),
            "capability": skill.capability.listing() if skill.capability else None,
            "assets": {
                "scripts": resource_loader.script_assets(skill),
            },
        }

    def load_planner_guidance(
        self,
        skill: Skill,
        *,
        requested_references: list[str] | None = None,
    ) -> dict[str, Any]:
        """Load model-facing guidance for a selected skill.

        Selection remains index-only, but once a skill is accepted the planner
        receives the bounded SKILL.md body. References stay opt-in so large
        supporting material only appears when explicitly requested.
        """

        resource_loader = self.resource_loader
        data: dict[str, Any] = {
            "name": skill.name,
            "description": skill.description,
            "skill_text": skill.body.strip(),
        }
        references = resource_loader.load_reference_paths(skill, requested_references or [])
        if references:
            data["requested_references"] = [
                {
                    "path": item.get("path", ""),
                    "text": item.get("text", ""),
                    "truncated": bool(item.get("truncated", False)),
                }
                for item in references
                if isinstance(item, dict) and item.get("text")
            ]
        capability = skill.capability.listing() if skill.capability else None
        if capability:
            data["capability"] = capability
        return {key: value for key, value in data.items() if value not in (None, "", [], {})}

    def load_semantic_guidance(self, skill: Skill) -> dict[str, Any]:
        """Load bounded knowledge text for semantic phase-criteria generation.

        Semantic parsing needs domain guidance, not skill-local resources or
        callable capability metadata. Keeping this shape separate prevents the
        parser from learning runtime tools while still allowing selected
        knowledge to influence completion and continuation conditions.
        """

        body = skill.body.strip()
        bounded = _bounded_semantic_body(body, self.max_skill_chars)
        return {
            "name": skill.name,
            "description": skill.description,
            "skill_text": bounded,
            "truncated": len(body) > len(bounded),
        }


def _bounded_semantic_body(body: str, max_chars: int) -> str:
    """Keep both operating guidance and end-of-file completion guidance."""

    if len(body) <= max_chars:
        return body
    marker = "\n\n[... bounded middle omitted ...]\n\n"
    if max_chars <= len(marker):
        return body[:max_chars].rstrip()
    available = max(0, max_chars - len(marker))
    head_chars = (available * 2) // 3
    tail_chars = available - head_chars
    return f"{body[:head_chars].rstrip()}{marker}{body[-tail_chars:].lstrip()}"


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
        return str(path.relative_to(skill.skill_dir)).replace("\\", "/")
    except ValueError:
        return str(path)
