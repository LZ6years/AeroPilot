"""Track task execution state for short-horizon runtime loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.schemas import SemanticPlan


@dataclass
class MissionState:
    run_id: str
    user_instruction: str
    semantic_plan: SemanticPlan
    current_phase_index: int = 0
    recent_actions: list[dict[str, Any]] = field(default_factory=list)
    recent_observations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    connected: bool = False
    airborne: bool = False
    hover_required: bool = False
    completed: bool = False

    @property
    def current_phase_name(self) -> str:
        if self.current_phase_index >= len(self.semantic_plan.phases):
            return "completed"
        return self.semantic_plan.phases[self.current_phase_index].name

    def advance_phase(self) -> None:
        # Move to the next semantic phase after one short window has completed.
        self.current_phase_index += 1
        if self.current_phase_index >= len(self.semantic_plan.phases):
            self.completed = True

    def record_action(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        # Keep a bounded action history and update minimal runtime flags.
        entry = {"tool": tool, "args": dict(args), "result": dict(result)}
        self.recent_actions.append(entry)
        self.recent_actions = self.recent_actions[-10:]
        if result.get("ok") is True:
            if tool == "uav.connect":
                self.connected = True
            if tool == "uav.takeoff":
                self.airborne = True
            if tool == "uav.land":
                self.airborne = False

    def record_observation(self, observation: dict[str, Any]) -> None:
        # Keep a bounded observation history for later VLM/context work.
        self.recent_observations.append(dict(observation))
        self.recent_observations = self.recent_observations[-5:]

    def record_error(self, error_type: str, message: str) -> None:
        self.errors.append({"error_type": error_type, "message": message})
        self.hover_required = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user_instruction": self.user_instruction,
            "semantic_plan": self.semantic_plan.to_dict(),
            "current_phase_index": self.current_phase_index,
            "current_phase": self.current_phase_name,
            "recent_actions": list(self.recent_actions),
            "recent_observations": list(self.recent_observations),
            "errors": list(self.errors),
            "safety": {"hover_required": self.hover_required},
            "connected": self.connected,
            "airborne": self.airborne,
            "completed": self.completed,
        }
