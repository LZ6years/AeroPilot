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
