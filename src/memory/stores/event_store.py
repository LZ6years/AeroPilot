"""Append-only memory event store for run-local FlightMemory records."""

from __future__ import annotations

import json
import asyncio
import base64
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from threading import get_ident
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _completion_payload(value):
    """Copy JSON result metadata without duplicating inline image data."""
    if isinstance(value, dict):
        return {key: _completion_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_completion_payload(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[provider_media_rehydrated_from_artifact]"
    return value


def _frame_sha256(frame: dict[str, Any]) -> str | None:
    """Capture immutable media identity while the observation is persisted."""

    for key in ("artifact_path", "path"):
        raw_path = frame.get(key)
        if isinstance(raw_path, str) and raw_path:
            try:
                return hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
            except OSError:
                continue
    encoded = frame.get("base64")
    if isinstance(encoded, str) and encoded:
        try:
            return hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest()
        except (ValueError, TypeError):
            return None
    return None


class MemoryEventWriter:
    def __init__(self, run_dir: str | Path, *, persist_inline_media: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.persist_inline_media = persist_inline_media
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
                    frame_ref = self.write_frame(obs_id, modality, frame, timestamp=timestamp)
                    frame_refs.append({**frame_ref, "modality": str(modality)})

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
        metadata = dict(frame)
        digest = _frame_sha256(metadata)
        if digest is not None:
            metadata["sha256"] = digest
        if not self.persist_inline_media:
            # Keep the original in-memory observation intact. Remove a duplicate
            # only after the capture owner has persisted its exact bytes.
            raw_path = metadata.get("artifact_path") or metadata.get("path")
            if raw_path and Path(raw_path).is_file():
                metadata.pop("base64", None)
        return self._append(
            "frames.jsonl",
            "frame",
            {
                "observation_id": observation_id,
                "modality": modality,
                "path": frame.get("path"),
                "metadata": metadata,
                "timestamp": timestamp or _timestamp(),
            },
        )
    def write_tool_completion(self, *, instruction_id, decision_id, call_index,
                              tool_call_id, tool_name, args, result):
        """Save a received result independently of ordered mission submission.

        Audit-only receipts cannot establish terminal evidence. Media stays in
        capability artifacts; inline image payloads are omitted from this log.
        The caller owns synchronization, just as for all other writer methods.
        """
        return self._append("tool_completions.jsonl", "completion", {
            "instruction_id": instruction_id, "decision_id": decision_id,
            "call_index": call_index, "tool_call_id": tool_call_id,
            "tool_name": tool_name, "args": _completion_payload(args),
            "result": _completion_payload(result),
        })

    def write_transcript_event(
        self,
        instruction_id: str | None,
        kind: str,
        payload: dict[str, Any],
        *,
        schema_version: str,
    ) -> dict[str, Any]:
        """Append one canonical model/tool/state exchange event."""

        return self._append(
            "agent_transcript.jsonl",
            "turn",
            {
                "schema_version": schema_version,
                "instruction_id": instruction_id,
                "kind": kind,
                "payload": dict(payload),
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
    """Read independent records, optionally sharing file text within one build.

    Scope state is context-local and owner-checked: asyncio tasks inherit context
    variables, but must not inherit their parent's mutable file cache.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self._read_scope: ContextVar = ContextVar("memory_record_read_scope", default=None)

    @staticmethod
    def _scope_owner():
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return get_ident(), task

    @contextmanager
    def read_scope(self):
        """Reuse successful file reads until the outermost local scope exits.

        Text, rather than mutable parsed records, is cached. Missing files and
        I/O failures remain retryable. A fresh build always sees fresh files.
        """
        owner = self._scope_owner()
        current = self._read_scope.get()
        if current is not None and current[0] == owner:
            yield
            return
        token = self._read_scope.set((owner, {}))
        try:
            yield
        finally:
            self._read_scope.reset(token)

    def _read_lines(self, path: Path) -> list[str] | None:
        current = self._read_scope.get()
        cache = current[1] if current is not None and current[0] == self._scope_owner() else None
        # Path.resolve() can itself query the filesystem on Windows. Repeated
        # references must hit the lexical key before doing that work again.
        key = path
        if cache is not None:
            if path in cache:
                return cache[path]
            key = path.resolve()
            if key in cache:
                cache[path] = cache[key]
                return cache[key]
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        if cache is not None:
            cache[key] = cache[path] = lines
        return lines

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
        lines = self._read_lines(path)
        if lines is None:
            return None
        if 1 <= line_number <= len(lines):
            return json.loads(lines[line_number - 1])
        return None

    def read_refs(self, refs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        records = []
        for ref in refs or []:
            record = self.read_ref(ref)
            if record is not None:
                records.append(record)
        return records

    def read_observation(self, observation_id: str) -> dict[str, Any]:
        """Rehydrate one persisted observation, independent of rolling memory.

        Callers own task authorization. Only frames referenced by this exact
        observation are read; never pair old RGB with a newer depth frame.
        """
        lines = self._read_lines(self.run_dir / "memory/observations.jsonl") or []
        record = next((item for line in lines if line.strip()
                       and (item := json.loads(line)).get("id") == observation_id), None)
        if record is None:
            raise ValueError(f"observation {observation_id!r} is not available in this run")
        frames = {}
        for ref in record.get("frame_refs", []):
            frame = self.read_ref(ref)
            if frame is None or frame.get("observation_id") != observation_id or frame.get("id") != ref.get("id"):
                raise ValueError("observation frame reference is missing or mismatched")
            modality = frame["modality"]
            if modality in frames:
                raise ValueError("observation contains duplicate frame modalities")
            frames[modality] = frame
        pose = self.read_ref(record.get("pose_ref"))
        return {**record, "observation_id": observation_id, "frames": frames,
                "pose": pose.get("pose", {}) if pose else {}}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_pose(result: dict[str, Any]) -> dict[str, Any] | None:
    data = result.get("data", {}) if isinstance(result, dict) else {}
    pose = data.get("pose") if isinstance(data, dict) else None
    return dict(pose) if isinstance(pose, dict) else None
