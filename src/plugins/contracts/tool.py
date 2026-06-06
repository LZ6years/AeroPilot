"""Placeholder contract for future plugin-provided tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContribution:
    plugin_id: str
    name: str
    description: str = ""
