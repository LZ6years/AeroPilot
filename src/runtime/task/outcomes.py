"""Typed results exchanged by instruction, turn and physical-window services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.memory.context.contracts import CheckpointReason


TurnKind = Literal[
    "protocol_rejected",
    "runtime_control",
    "read_only",
    "physical",
]


@dataclass(frozen=True)
class PhysicalWindowOutcome:
    """Evidence produced by one completed physical action window."""

    executed_steps: tuple[dict[str, Any], ...]
    failed_call_index: int | None
    observation_after: dict[str, Any]
    measurements: dict[str, Any]


@dataclass(frozen=True)
class TurnOutcome:
    """Loop-facing result of one provider turn."""

    kind: TurnKind
    checkpoint_reason: CheckpointReason | None = None
    reset_rejections: bool = False
    details: dict[str, Any] = field(default_factory=dict)
