"""Aggregate plugin contributions discovered from plugin asset folders."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.plugins.contracts.context import ContextModuleContribution
from src.plugins.contracts.skill import Skill


@dataclass(frozen=True)
class PluginContributions:
    skills: list[Skill] = field(default_factory=list)
    context_modules: list[ContextModuleContribution] = field(default_factory=list)

    @property
    def knowledge_skills(self) -> list[Skill]:
        return self.skills
