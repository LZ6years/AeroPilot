"""Catalogs for enabled skills and callable skill capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.plugins.contracts.skill import Skill, SkillCapability


@dataclass(frozen=True)
class SkillCatalog:
    skills: list[Skill]

    def listings(self) -> list[dict[str, Any]]:
        return [skill.listing() for skill in self.skills]

    def index_listings(self) -> list[dict[str, Any]]:
        """Return lightweight selection-time listings without source paths or bodies."""

        return [skill.index_listing() for skill in self.skills]

    def by_name(self) -> dict[str, Skill]:
        return {skill.name: skill for skill in self.skills}

    def validate_selection(self, requested_names: list[str]) -> tuple[list[Skill], list[str]]:
        by_name = self.by_name()
        by_qualified_name = {f"{skill.plugin_id}.{skill.name}": skill for skill in self.skills}
        accepted: list[Skill] = []
        rejected: list[str] = []
        seen_requested: set[str] = set()
        seen_skills: set[str] = set()
        for raw_name in requested_names:
            name = str(raw_name).strip()
            if not name or name in seen_requested:
                continue
            seen_requested.add(name)
            skill = by_name.get(name) or by_qualified_name.get(name) or _unique_suffix_match(name, by_name)
            if skill is None:
                rejected.append(name)
                continue
            if skill.name in seen_skills:
                continue
            seen_skills.add(skill.name)
            accepted.append(skill)
        return accepted, rejected

    def validate_reference_selection(
        self,
        accepted_skills: list[Skill],
        requested_references: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Validate requested reference paths against the already selected skills."""

        accepted_by_name = {skill.name: skill for skill in accepted_skills}
        valid: dict[str, list[str]] = {}
        rejected: dict[str, list[str]] = {}
        for raw_skill_name, raw_paths in requested_references.items():
            skill_name = str(raw_skill_name).strip()
            if not skill_name:
                continue
            paths = raw_paths if isinstance(raw_paths, list) else []
            skill = accepted_by_name.get(skill_name)
            if skill is None:
                rejected[skill_name] = [str(path).strip() for path in paths if str(path).strip()]
                continue
            allowed = {item["path"] for item in skill.reference_manifest()}
            seen: set[str] = set()
            for raw_path in paths:
                path = str(raw_path).strip().replace("\\", "/")
                if not path or path in seen:
                    continue
                seen.add(path)
                target = valid if path in allowed else rejected
                target.setdefault(skill.name, []).append(path)
        return valid, rejected


@dataclass(frozen=True)
class SkillCapabilityCatalog:
    skills: list[Skill]

    @property
    def capabilities(self) -> list[SkillCapability]:
        return [skill.capability for skill in self.skills if skill.capability is not None]

    def listings(self) -> list[dict[str, Any]]:
        return [
            {
                "skill_name": skill.name,
                "plugin_id": skill.plugin_id,
                **skill.capability.listing(),
            }
            for skill in self.skills
            if skill.capability is not None
        ]

    def by_call_name(self) -> dict[str, tuple[Skill, SkillCapability]]:
        return {
            skill.capability.call_name: (skill, skill.capability)
            for skill in self.skills
            if skill.capability is not None
        }

    def action_schemas(self) -> dict[str, dict[str, Any]]:
        return {
            capability.call_name: {
                "description": capability.description,
                "input_schema": dict(capability.input_schema),
                "wire_name": capability.wire_name,
                "runtime": capability.runtime.to_dict(),
                "source": "skill_capability",
                "mock": capability.mock,
            }
            for capability in self.capabilities
        }

    def planner_tools(self) -> list[str]:
        return sorted(capability.call_name for capability in self.capabilities)


def _unique_suffix_match(name: str, by_name: dict[str, Skill]) -> Skill | None:
    if "." not in name:
        return None
    suffix = name.rsplit(".", 1)[-1]
    matches = [skill for skill_name, skill in by_name.items() if skill_name == suffix]
    return matches[0] if len(matches) == 1 else None
