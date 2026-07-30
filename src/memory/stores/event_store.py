"""Append-only memory event store for run-local FlightMemory records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryEventWriter:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.memory_dir = self.run_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._counters: dict[str, int] = {}

    def write_instruction(self, instruction: str, semantic_plan: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._append(
            "instructions.jsonl",
            "inst",
            {
                "instruction": instruction,
                "semantic_plan": dict(semantic_plan or {}),
            },
        )

    def write_action(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        action_ref = self._append(
            "actions.jsonl",
            "act",
            {
                "tool": tool,
                "args": dict(args),
                "result": dict(result),
            },
        )
        pose_ref = None
        pose = _result_pose(result)
        if pose:
            pose_ref = self.write_pose(pose, source="action_result", action_id=action_ref["id"])
        return {"action_ref": action_ref, "pose_ref": pose_ref}

    def write_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs_id = self._next_id("obs")
        timestamp = str(observation.get("timestamp") or _timestamp())
        pose_ref = None
        pose = observation.get("pose")
        if isinstance(pose, dict):
            pose_ref = self.write_pose(pose, source="observation", observation_id=obs_id, timestamp=timestamp)

        frame_refs = []
        frames = observation.get("frames", {})
        if isinstance(frames, dict):
            for modality, frame in frames.items():
                if isinstance(frame, dict):
                    frame_refs.append(self.write_frame(obs_id, modality, frame, timestamp=timestamp))

        observation_ref = self._append_with_id(
            "observations.jsonl",
            obs_id,
            {
                "source": observation.get("source"),
                "timestamp": timestamp,
                "scene_instance_id": observation.get("scene_instance_id"),
                "agent_id": observation.get("agent_id"),
                "modalities": list(observation.get("modalities", [])),
                "pose_ref": pose_ref,
                "frame_refs": frame_refs,
            },
        )
        return {"observation_ref": observation_ref, "pose_ref": pose_ref, "frame_refs": frame_refs}

    def write_pose(
        self,
        pose: dict[str, Any],
        source: str,
        action_id: str | None = None,
        observation_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        return self._append(
            "poses.jsonl",
            "pose",
            {
                "pose": dict(pose),
                # Pose producers own coordinate semantics. The memory store
                # preserves a self-described frame and uses an honest generic
                # marker when older clients omit it instead of guessing AirSim.
                "coordinate_frame": str(
                    pose.get("coordinate_frame")
                    or "canonical_unspecified"
                ),
                "source": source,
                "action_id": action_id,
                "observation_id": observation_id,
                "timestamp": timestamp or _timestamp(),
            },
        )

    def write_frame(
        self,
        observation_id: str,
        modality: str,
        frame: dict[str, Any],
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        return self._append(
            "frames.jsonl",
            "frame",
            {
                "observation_id": observation_id,
                "modality": modality,
                "path": frame.get("path"),
                "metadata": dict(frame),
                "timestamp": timestamp or _timestamp(),
            },
        )

    def write_agent_message(
        self,
        source: str,
        phase: str | None,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one user-facing AgentMessage record to run memory."""

        return self._append(
            "agent_messages.jsonl",
            "msg",
            {
                "source": source,
                "phase": phase,
                "message": dict(message),
            },
        )

    def _append(self, filename: str, prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._append_with_id(filename, self._next_id(prefix), payload)

    def _append_with_id(self, filename: str, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self.memory_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        line_number = self._next_line(filename)
        record = {"id": record_id, "timestamp": _timestamp(), **payload}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "id": record_id,
            "ref": f"memory/{filename}#{line_number}",
            "timestamp": record["timestamp"],
        }

    def _next_id(self, prefix: str) -> str:
        key = f"id:{prefix}"
        self._counters[key] = self._counters.get(key, 0) + 1
        return f"{prefix}_{self._counters[key]:04d}"

    def _next_line(self, filename: str) -> int:
        key = f"line:{filename}"
        if key not in self._counters:
            path = self.memory_dir / filename
            self._counters[key] = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
        self._counters[key] += 1
        return self._counters[key]


class MemoryRecordReader:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def read_ref(self, ref: dict[str, Any] | str | None) -> dict[str, Any] | None:
        if ref is None:
            return None
        ref_value = ref.get("ref") if isinstance(ref, dict) else ref
        if not isinstance(ref_value, str) or "#" not in ref_value:
            return None
        relative_path, line_text = ref_value.split("#", 1)
        try:
            line_number = int(line_text)
        except ValueError:
            return None
        path = self.run_dir / relative_path
        if not path.exists():
            return None
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if index == line_number:
                return json.loads(line)
        return None

    def read_refs(self, refs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        records = []
        for ref in refs or []:
            record = self.read_ref(ref)
            if record is not None:
                records.append(record)
        return records


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_pose(result: dict[str, Any]) -> dict[str, Any] | None:
    data = result.get("data", {}) if isinstance(result, dict) else {}
    pose = data.get("pose") if isinstance(data, dict) else None
    return dict(pose) if isinstance(pose, dict) else None
