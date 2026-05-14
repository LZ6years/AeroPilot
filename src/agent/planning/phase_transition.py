"""Phase transition decisions emitted with a short-horizon action plan.

The model may suggest whether the current semantic phase should continue,
advance, or finish the task, but it must not directly choose arbitrary phase
names or mutate MissionState. Runtime owns applying this decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CONTINUE_CURRENT_PHASE = "continue_current_phase"
ADVANCE_TO_NEXT_PHASE = "advance_to_next_phase"
COMPLETE_TASK = "complete_task"
VALID_PHASE_TRANSITIONS = {
    CONTINUE_CURRENT_PHASE,
    ADVANCE_TO_NEXT_PHASE,
    COMPLETE_TASK,
}


@dataclass(frozen=True)
class PhaseTransition:
    """Model-facing phase progression suggestion for the current phase window."""

    decision: str
    reason: str = ""

    @classmethod
    def default_advance(cls, reason: str = "legacy short-horizon plan advances after one window") -> "PhaseTransition":
        return cls(ADVANCE_TO_NEXT_PHASE, reason)

    @classmethod
    def continue_current_phase(cls, reason: str = "continue collecting evidence for current phase") -> "PhaseTransition":
        return cls(CONTINUE_CURRENT_PHASE, reason)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseTransition":
        payload = dict(data)
        illegal_targets = sorted(set(payload) & {"phase", "phase_name", "phase_index", "next_phase", "next_phase_index"})
        if illegal_targets:
            raise ValueError(
                "phase_transition must not specify target phase fields: "
                + ", ".join(illegal_targets)
            )
        decision = str(payload.get("decision", "")).strip()
        if decision not in VALID_PHASE_TRANSITIONS:
            raise ValueError(f"Unsupported phase_transition decision: {decision}")
        return cls(decision=decision, reason=str(payload.get("reason", "")).strip())

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "reason": self.reason}
