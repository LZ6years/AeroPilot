"""Placeholder contract for future policy rule plugins."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyRuleContribution:
    plugin_id: str
    name: str
    description: str = ""
