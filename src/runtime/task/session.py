"""Mutable state for one model-driven instruction execution.

The facade creates one session per ``run_task`` call. Loop and turn services
share this object instead of storing instruction counters on ``TaskRuntime`` or
passing large, weakly-related argument lists between helpers.
"""

from __future__ import annotations

import copy
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
    from src.runtime.control.runtime_control import RuntimeControl


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
    control: "RuntimeControl | None" = None
    history_projection_state: HistoryProjectionState = field(
        default_factory=HistoryProjectionState
    )
    history_compaction_coordinator: HistoryCompactionCoordinator = field(
        default_factory=HistoryCompactionCoordinator
    )
    decision_index: int = 0
    seen_call_ids: set[str] = field(default_factory=set)
    executed_read_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    consecutive_rejections: int = 0

    def next_decision_id(self) -> str:
        self.decision_index += 1
        return f"decision_{self.decision_index:04d}"

    def current_task_state(self) -> dict[str, Any]:
        """Return the transient authoritative state for the next model turn."""

        loaded = self.knowledge_service.loaded_index()
        self.state.loaded_knowledge = copy.deepcopy(loaded)
        return {
            "task_plan": (
                self.state.task_plan.to_dict() if self.state.task_plan else None
            ),
            "loaded_knowledge": copy.deepcopy(loaded),
        }
