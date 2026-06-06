"""Placeholder contract for future response protocol plugins."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseProtocolContribution:
    plugin_id: str
    name: str
    description: str = ""
