"""Read saved V3 runs without regenerating inputs, plans or missing evidence."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from src.memory.context.contracts import INSTRUCTION_ENVELOPE_SCHEMA, TRANSCRIPT_SCHEMA
from src.memory.stores import MemoryRecordReader
from src.operator_api.services.event_stream import public_value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """A concurrently written final fragment is not yet a durable event."""
    if not path.is_file():
        return []
    data = path.read_bytes()
    return [json.loads(line.decode("utf-8")) for line in data.split(b"\n")[:-1] if line.strip()]


class RunRepository:
    """Opaque run/resource IDs constrain every read to the configured run root."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._runs: dict[str, Path] = {}
        self._assets: dict[tuple[str, str], Path] = {}
        self._lock = threading.RLock()
        self._last_scan = 0.0

    def register(self, path: Path) -> str:
        path = path.resolve()
        relative = path.relative_to(self.root).as_posix()
        identity = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
        with self._lock:
            self._runs[identity] = path
        return identity

    def _scan(self) -> None:
        if time.monotonic() - self._last_scan < 5:
            return
        if self.root.exists():
            for file in self.root.glob("**/contexts/instruction_envelope.json"):
                try:
                    if read_json(file).get("schema_version") == INSTRUCTION_ENVELOPE_SCHEMA:
                        self.register(file.parent.parent)
                except (OSError, ValueError):
                    continue
        self._last_scan = time.monotonic()

    def get(self, run_id: str) -> Path:
        with self._lock:
            if run_id not in self._runs:
                self._scan()
            if run_id not in self._runs:
                raise KeyError("Unknown run ID")
            path = self._runs[run_id].resolve()
        path.relative_to(self.root)
        return path

    def list(self) -> list[dict[str, Any]]:
        def modified(item):
            try:
                return (item[1] / "state/latest_state.json").stat().st_mtime
            except OSError:
                return 0
        with self._lock:
            self._scan()
            entries = sorted(self._runs.items(), key=modified, reverse=True)[:100]
        output = []
        for identity, path in entries:
            try:
                state = read_json(path / "state/latest_state.json")
                if not state:
                    continue
                output.append({
                    "run_id": identity, "name": path.name,
                    "instruction": state.get("user_instruction", ""),
                    "completed": state.get("completed", False), "stopped": state.get("stopped", False),
                    "map_id": read_json(path / "config/map_package.json").get("map_id"),
                    "windows": state.get("physical_window_count", 0),
                })
            except (OSError, ValueError):
                continue
        return output

    @staticmethod
    def child(run: Path, reference: str) -> Path:
        path = Path(reference)
        resolved = (path if path.is_absolute() else run / path).resolve()
        resolved.relative_to(run.resolve())
        return resolved

    def asset_url(self, run_id: str, reference: str | None) -> str | None:
        if not reference:
            return None
        run = self.get(run_id)
        try:
            path = self.child(run, reference)
        except ValueError:
            return None
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".npy"}:
            return None
        identity = hashlib.sha256(path.relative_to(run).as_posix().encode("utf-8")).hexdigest()[:24]
        with self._lock:
            self._assets[(run_id, identity)] = path
        return f"/api/runs/{run_id}/assets/{identity}"

    def asset(self, run_id: str, asset_id: str) -> Path:
        # On service restart, rebuild allowed references from the captured run.
        with self._lock:
            found = self._assets.get((run_id, asset_id))
        if found is None:
            self.timeline(run_id)
            with self._lock:
                found = self._assets.get((run_id, asset_id))
        if found is None:
            raise KeyError("Unknown artifact ID")
        path = self.child(self.get(run_id), str(found))
        if not path.is_file():
            raise KeyError("Artifact is unavailable")
        return path

    def events(self, run_id: str) -> list[dict[str, Any]]:
        run = self.get(run_id)
        operator_path = run / "operator/events.jsonl"
        if operator_path.exists():
            return read_jsonl(operator_path)
        instruction_id = None
        output = []
        for seq, record in enumerate(read_jsonl(run / "events.jsonl"), 1):
            payload = record.get("payload") or {}
            instruction_id = payload.get("instruction_id") or instruction_id
            output.append({
                "schema_version": "operator_event_v1", "seq": seq,
                "kind": record.get("event"), "timestamp": record.get("timestamp"),
                "instruction_id": instruction_id, "decision_id": payload.get("decision_id"),
                "payload": public_value(payload),
            })
        return output

    def timeline(self, run_id: str) -> dict[str, Any]:
        from .decisions import build_decisions

        run = self.get(run_id)
        envelope = read_json(run / "contexts/instruction_envelope.json")
        if envelope and envelope.get("schema_version") != INSTRUCTION_ENVELOPE_SCHEMA:
            raise ValueError("Only Context V3 runs are supported")
        transcript = read_jsonl(run / "memory/agent_transcript.jsonl")
        if any(row.get("schema_version") != TRANSCRIPT_SCHEMA for row in transcript):
            raise ValueError("Only agent_transcript_v3 records are supported")
        reader = MemoryRecordReader(run)
        observations = []
        def safe_ref(reference):
            if reference:
                self.child(run, str(reference["ref"]).split("#", 1)[0])
            return reader.read_ref(reference)
        with reader.read_scope():
            for row in read_jsonl(run / "memory/observations.jsonl"):
                frame_views = []
                for reference in row.get("frame_refs", []):
                    frame = safe_ref(reference) or {}
                    metadata = frame.get("metadata") or {}
                    asset = metadata.get("artifact_path") or metadata.get("path") or frame.get("path")
                    if metadata.get("encoding") == "npy":
                        asset = metadata.get("preview_path")
                    frame_views.append({
                        "modality": reference.get("modality", frame.get("modality")),
                        "frame_id": reference.get("id"), "width": metadata.get("width"),
                        "height": metadata.get("height"), "url": self.asset_url(run_id, asset),
                    })
                pose = safe_ref(row.get("pose_ref")) or {}
                observations.append({
                    "observation_id": row["id"], "timestamp": row.get("timestamp"),
                    "source": row.get("source"), "pose": pose.get("pose") or {}, "frames": frame_views,
                })
        events = self.events(run_id)
        return {
            "run_id": run_id, "name": run.name,
            "decisions": build_decisions(self, run_id, events, transcript),
            "observations": observations,
            "trajectory": [{"observation_id": row["observation_id"], "timestamp": row["timestamp"], **row["pose"]} for row in observations if row["pose"]],
            "state": public_value(read_json(run / "state/latest_state.json")),
        }

    def decision(self, run_id: str, decision_id: str, instruction_id: str | None = None) -> dict[str, Any]:
        view = self.timeline(run_id)
        matches = [row for row in view["decisions"] if row["decision_id"] == decision_id and (instruction_id is None or row["instruction_id"] == instruction_id)]
        if not matches:
            raise KeyError("Decision is not yet available")
        if len(matches) != 1:
            raise ValueError("Decision ID repeats across instructions; provide instruction_id")
        selected = matches[0]
        by_id = {row["observation_id"]: row for row in view["observations"]}
        selected["observation"] = by_id.get(selected.get("observation_id"))
        selected["observation_after"] = by_id.get(selected.get("observation_after_id"))
        return selected
