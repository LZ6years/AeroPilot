"""Continuous per-scene UAV flight memory manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MAX_INSTRUCTION_REFS = 50
MAX_RECENT_OBSERVATION_REFS = 50
MAX_RECENT_ACTION_REFS = 50
MAX_RECENT_POSE_REFS = 50
MAX_RECENT_FRAME_REFS = 100
MAX_RECENT_AGENT_MESSAGE_REFS = 50


@dataclass
class FlightMemory:
    run_id: str
    map_id: str
    scene_instance_id: str
    agent_id: str
    active_instruction: str = ""
    active_instruction_id: str | None = None
    instruction_history: list[dict[str, Any]] = field(default_factory=list)
    recent_observation_refs: list[dict[str, Any]] = field(default_factory=list)
    recent_action_refs: list[dict[str, Any]] = field(default_factory=list)
    recent_pose_refs: list[dict[str, Any]] = field(default_factory=list)
    recent_frame_refs: list[dict[str, Any]] = field(default_factory=list)
    recent_agent_message_refs: list[dict[str, Any]] = field(default_factory=list)
    observation_anchors: dict[str, dict[str, Any]] = field(default_factory=dict)
    reset_epoch: int = 0
    rolling_summary: dict[str, Any] = field(default_factory=dict)
    memory_reader: Any = field(default=None, repr=False, compare=False)

    def update_instruction_ref(self, ref: dict[str, Any], instruction: str) -> None:
        self.active_instruction = instruction
        self.active_instruction_id = ref.get("id")
        self.instruction_history.append(dict(ref))
        self.instruction_history = self.instruction_history[-MAX_INSTRUCTION_REFS:]
        self.observation_anchors = {}

    def append_observation_ref(self, ref: dict[str, Any]) -> None:
        self.recent_observation_refs.append(dict(ref))
        self.recent_observation_refs = self.recent_observation_refs[-MAX_RECENT_OBSERVATION_REFS:]

    def append_action_ref(self, ref: dict[str, Any]) -> None:
        self.recent_action_refs.append(dict(ref))
        self.recent_action_refs = self.recent_action_refs[-MAX_RECENT_ACTION_REFS:]

    def append_pose_ref(self, ref: dict[str, Any] | None) -> None:
        if ref is None:
            return
        self.recent_pose_refs.append(dict(ref))
        self.recent_pose_refs = self.recent_pose_refs[-MAX_RECENT_POSE_REFS:]

    def append_frame_refs(self, refs: list[dict[str, Any]] | None) -> None:
        for ref in refs or []:
            self.recent_frame_refs.append(dict(ref))
        self.recent_frame_refs = self.recent_frame_refs[-MAX_RECENT_FRAME_REFS:]

    def append_agent_message_ref(self, ref: dict[str, Any]) -> None:
        self.recent_agent_message_refs.append(dict(ref))
        self.recent_agent_message_refs = self.recent_agent_message_refs[-MAX_RECENT_AGENT_MESSAGE_REFS:]

    def set_observation_anchor(self, name: str, ref: dict[str, Any] | None, *, overwrite: bool = True) -> None:
        """Record a stable observation role for the active instruction.

        Recent observation history is a rolling list, so roles such as "task
        initial" or "phase start" should be stored explicitly instead of being
        inferred later from whichever frames remain in memory.
        """

        if ref is None:
            return
        key = str(name).strip()
        if not key:
            return
        if not overwrite and key in self.observation_anchors:
            return
        self.observation_anchors[key] = dict(ref)

    def reset_pose_epoch(self, active_instruction: str | None = None) -> None:
        self.reset_epoch += 1
        if active_instruction is not None:
            self.active_instruction = active_instruction
        self.recent_observation_refs = []
        self.recent_action_refs = []
        self.recent_pose_refs = []
        self.recent_frame_refs = []
        self.rolling_summary = {"reset_reason": "pose_reset", "reset_epoch": self.reset_epoch}

    @property
    def latest_pose(self) -> dict[str, Any]:
        if self.memory_reader is None or not self.recent_pose_refs:
            return {}
        record = self.memory_reader.read_ref(self.recent_pose_refs[-1])
        pose = record.get("pose") if isinstance(record, dict) else None
        return dict(pose) if isinstance(pose, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "map_id": self.map_id,
            "scene_instance_id": self.scene_instance_id,
            "agent_id": self.agent_id,
            "active_instruction": self.active_instruction,
            "active_instruction_id": self.active_instruction_id,
            "instruction_history": [dict(item) for item in self.instruction_history],
            "recent_observation_refs": [dict(item) for item in self.recent_observation_refs],
            "recent_action_refs": [dict(item) for item in self.recent_action_refs],
            "recent_pose_refs": [dict(item) for item in self.recent_pose_refs],
            "recent_frame_refs": [dict(item) for item in self.recent_frame_refs],
            "recent_agent_message_refs": [dict(item) for item in self.recent_agent_message_refs],
            "observation_anchors": {key: dict(value) for key, value in self.observation_anchors.items()},
            "reset_epoch": self.reset_epoch,
            "rolling_summary": dict(self.rolling_summary),
        }
