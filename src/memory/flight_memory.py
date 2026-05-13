"""Continuous per-scene UAV flight memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FlightMemory:
    run_id: str
    map_id: str
    scene_instance_id: str
    agent_id: str
    active_instruction: str = ""
    instruction_history: list[dict[str, Any]] = field(default_factory=list)
    recent_observations: list[dict[str, Any]] = field(default_factory=list)
    recent_actions: list[dict[str, Any]] = field(default_factory=list)
    reset_epoch: int = 0
    rolling_summary: dict[str, Any] = field(default_factory=dict)

    def update_instruction(self, instruction: str, semantic_plan: dict[str, Any] | None = None) -> None:
        self.active_instruction = instruction
        entry = {"instruction": instruction}
        if semantic_plan is not None:
            entry["semantic_plan"] = semantic_plan
        self.instruction_history.append(entry)
        self.instruction_history = self.instruction_history[-10:]

    def append_observation(self, observation: dict[str, Any]) -> None:
        self.recent_observations.append(dict(observation))
        self.recent_observations = self.recent_observations[-5:]

    def append_action(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        self.recent_actions.append({"tool": tool, "args": dict(args), "result": dict(result)})
        self.recent_actions = self.recent_actions[-10:]

    def reset_pose_epoch(self, active_instruction: str | None = None) -> None:
        self.reset_epoch += 1
        if active_instruction is not None:
            self.active_instruction = active_instruction
        self.recent_observations = []
        self.recent_actions = []
        self.rolling_summary = {"reset_reason": "pose_reset", "reset_epoch": self.reset_epoch}

    @property
    def latest_observation(self) -> dict[str, Any] | None:
        return self.recent_observations[-1] if self.recent_observations else None

    @property
    def latest_pose(self) -> dict[str, Any]:
        latest = self.latest_observation or {}
        pose = latest.get("pose")
        return dict(pose) if isinstance(pose, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
