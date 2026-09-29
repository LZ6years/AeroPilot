"""Mutable state for one model-driven instruction execution.

The facade creates one session per ``run_task`` call. Loop and turn services
share this object instead of storing instruction counters on ``TaskRuntime`` or
passing large, weakly-related argument lists between helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.transcript import AgentTranscript
from src.env_client import SceneInstance
from src.memory.flight_memory import FlightMemory


@dataclass
class InstructionSession:
    """Instruction-local state; a new instance is required for every request."""

    scene: SceneInstance
    agent_id: str
    state: MissionState
    memory: FlightMemory
    transcript: AgentTranscript
    decision_index: int = 0
    seen_call_ids: set[str] = field(default_factory=set)
    executed_read_signatures: set[tuple[str, str, str]] = field(default_factory=set)
    consecutive_rejections: int = 0
    latest_turn_messages: list[dict[str, Any]] = field(default_factory=list)

    def next_decision_id(self) -> str:
        self.decision_index += 1
        return f"decision_{self.decision_index:04d}"

