"""Placeholder contract for future plugin-provided capabilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilitySkillContribution:
    plugin_id: str
    name: str
    description: str = ""
