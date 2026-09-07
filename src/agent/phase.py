"""Phase transition contracts for FlyAgent missions.

This module is intentionally model- and runtime-neutral: planners and judges
can read these schemas, while TaskRuntime remains the only component that
mutates MissionState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.messages import AgentMessage


CONTINUE_CURRENT_PHASE = "continue_current_phase"
ADVANCE_TO_NEXT_PHASE = "advance_to_next_phase"
COMPLETE_TASK = "complete_task"
STOP_TASK_INCOMPLETE = "stop_task_incomplete"
VALID_PHASE_TRANSITIONS = {
    CONTINUE_CURRENT_PHASE,
    ADVANCE_TO_NEXT_PHASE,
    COMPLETE_TASK,
    STOP_TASK_INCOMPLETE,
}

@dataclass(frozen=True)
class PhaseTransition:
    """Model-facing phase progression suggestion for the current phase window."""

    decision: str
    reason: str = ""
    agent_messages: list[AgentMessage] = field(default_factory=list)

    @classmethod
    def continue_current_phase(cls, reason: str = "continue collecting evidence for current phase") -> "PhaseTransition":
        return cls(CONTINUE_CURRENT_PHASE, reason)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseTransition":
        payload = dict(data)
        illegal_targets = sorted(set(payload) & {"phase", "phase_name", "phase_index", "next_phase", "next_phase_index"})
        if illegal_targets:
            raise ValueError(
                "phase_decision must not specify target phase fields: "
                + ", ".join(illegal_targets)
            )
        decision = str(payload.get("decision", "")).strip()
        if decision not in VALID_PHASE_TRANSITIONS:
            raise ValueError(f"Unsupported phase_decision decision: {decision}")
        messages_data = payload.get("agent_messages", [])
        if messages_data is None:
            messages_data = []
        if not isinstance(messages_data, list):
            raise ValueError("phase_decision agent_messages must be a list")
        return cls(
            decision=decision,
            reason=str(payload.get("reason", "")).strip(),
            agent_messages=[AgentMessage.from_dict(dict(item)) for item in messages_data],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "agent_messages": [message.to_dict() for message in self.agent_messages],
        }
