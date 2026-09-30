"""Mutable state for one model-driven instruction execution.

The facade creates one session per ``run_task`` call. Loop and turn services
share this object instead of storing instruction counters on ``TaskRuntime`` or
passing large, weakly-related argument lists between helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent.context.compression import (
    HistoryCompactionCoordinator,
    HistoryProjectionState,
)
from src.agent.mission_state import MissionState
from src.agent.transcript import AgentTranscript
from src.env_client import SceneInstance
from src.memory.flight_memory import FlightMemory

if TYPE_CHECKING:
    from src.runtime.context import ContextCheckpointService


@dataclass
class TaskSession:
    """Task-local state; a new instance is required for every ``run_task``."""

    scene: SceneInstance
    agent_id: str
    state: MissionState
    memory: FlightMemory
    transcript: AgentTranscript
    checkpoints: "ContextCheckpointService"
    knowledge_service: Any
    history_projection_state: HistoryProjectionState = field(
        default_factory=HistoryProjectionState
    )
    history_compaction_coordinator: HistoryCompactionCoordinator = field(
        default_factory=HistoryCompactionCoordinator
    )
    decision_index: int = 0
    seen_call_ids: set[str] = field(default_factory=set)
    executed_read_signatures: set[tuple[str, str, str]] = field(default_factory=set)
    consecutive_rejections: int = 0

    def next_decision_id(self) -> str:
        self.decision_index += 1
        return f"decision_{self.decision_index:04d}"
