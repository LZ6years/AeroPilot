"""Read and materialize simulator-neutral benchmark audit records.

The Runtime owns append-only execution memory.  This module freezes a bounded
episode view before shutdown, reads that view after the async recorder closes,
and copies human-review assets into the evaluation directory.  It never reads
benchmark evaluation truth and never writes back into a Runtime run directory.
"""

from __future__ import annotations

import base64
import binascii
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.memory.context.contracts import (
    INSTRUCTION_ENVELOPE_SCHEMA,
    STATE_CHECKPOINT_SCHEMA,
    TRANSCRIPT_SCHEMA,
)
from src.runtime.context.checkpoints import UnsupportedRunSchema

from .trace import PHYSICAL_WINDOW_EVENTS


_AUDIT_FILES = (
    "memory/actions.jsonl",
    "memory/poses.jsonl",
    "memory/observations.jsonl",
    "memory/frames.jsonl",
    "memory/agent_transcript.jsonl",
    "memory/tool_completions.jsonl",
    "events.jsonl",
    "debug/debug_events.jsonl",
)
_NON_PLANNER_TOOLS = {
    "uav.connect",
    "uav.takeoff",
    "uav.land",
    "uav.set_pose",
    "uav.get_pose",
    "uav.get_frames",
}


@dataclass(frozen=True)
class EpisodeAuditCursor:
    """Freeze append-only audit boundaries before Runtime shutdown."""

    run_dir: str
    end_lines: dict[str, int]
    observation_anchors: dict[str, dict[str, Any]]
    task_initial_observation_ref: str | None
    started_at: str | None
    ended_at: str

    @classmethod
    def freeze(cls, run_dir: str | Path) -> "EpisodeAuditCursor":
        path = Path(run_dir)
        anchors = _observation_anchors(path)
        task_initial = anchors.get("task_initial", {})
        initial_ref = (
            str(task_initial.get("ref"))
            if task_initial.get("ref")
            else None
        )
        if (
            initial_ref is None
            and _line_count(path / "memory" / "observations.jsonl") > 0
        ):
            # Benchmark v1 owns one Runtime episode per run directory.  This
            # deterministic fallback also covers an async state manifest that
            # has not reached disk yet when the pre-shutdown cursor is frozen.
            initial_ref = "memory/observations.jsonl#1"
            first_records = _read_jsonl(
                path / "memory" / "observations.jsonl", 1
            )
            if first_records:
                first = first_records[0]
                anchors["task_initial"] = {
                    "id": first.get("id"),
                    "ref": initial_ref,
                    "timestamp": first.get("timestamp"),
                }
        started_at = _observation_timestamp(path, initial_ref)
        return cls(
            run_dir=str(path),
            end_lines={
                relative: _line_count(path / relative)
                for relative in _AUDIT_FILES
            },
            observation_anchors=anchors,
            task_initial_observation_ref=initial_ref,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkAuditLog:
    """One bounded, benchmark-neutral view of Runtime execution evidence."""

    run_dir: str
    started_at: str | None
    ended_at: str
    task_initial_observation_ref: str | None
    observations: tuple[dict[str, Any], ...]
    frames: tuple[dict[str, Any], ...]
    poses: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    debug_events: tuple[dict[str, Any], ...]
    planner_debug: tuple[dict[str, Any], ...]
    observation_anchors: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    diagnostics: tuple[dict[str, Any], ...] = ()
    # Appended after existing optional fields to preserve positional-call
    # compatibility for external BenchmarkAuditLog constructors.
    agent_transcript: tuple[dict[str, Any], ...] = ()
    tool_completions: tuple[dict[str, Any], ...] = ()

    def uncommitted_tool_completions(self):
        """Receipts are audit facts, never reconstructed canonical evidence.

        Runtime rejects reused call IDs within an instruction, so the canonical
        instruction/call identity also uniquely locates the receipt's decision.
        """
        committed = {(item.get("instruction_id"), item.get("payload", {}).get("tool_call_id"))
                     for item in self.agent_transcript if item.get("kind") == "tool_result"}
        return tuple(item for item in self.tool_completions
                     if (item.get("instruction_id"), item.get("tool_call_id")) not in committed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "task_initial_observation_ref": (
                self.task_initial_observation_ref
            ),
            "observation_anchors": {
                key: dict(value)
                for key, value in self.observation_anchors.items()
            },
            "observations": [dict(item) for item in self.observations],
            "frames": [dict(item) for item in self.frames],
            "poses": [dict(item) for item in self.poses],
            "actions": [dict(item) for item in self.actions],
            "events": [dict(item) for item in self.events],
            "debug_events": [dict(item) for item in self.debug_events],
            "planner_debug": [
                dict(item) for item in self.planner_debug
            ],
            "agent_transcript": [
                dict(item) for item in self.agent_transcript
            ],
            "diagnostics": [dict(item) for item in self.diagnostics],
            "tool_completions": [dict(item) for item in self.tool_completions],
        }


@dataclass(frozen=True)
class ResolvedFrameAsset:
    """Portable review asset derived from one Runtime frame record."""

    frame_id: str
    observation_id: str | None
    modality: str
    timestamp: str | None
    display_path: str | None
    raw_path: str | None
    source: str | None
    unavailable_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkAuditReader:
    """Read records bounded by an :class:`EpisodeAuditCursor`."""

    def read(self, cursor: EpisodeAuditCursor) -> BenchmarkAuditLog:
        run_dir = Path(cursor.run_dir)
        records = {
            relative: _read_jsonl(
                run_dir / relative, cursor.end_lines.get(relative, 0)
            )
            for relative in _AUDIT_FILES
        }
        _require_v2_transcript(records["memory/agent_transcript.jsonl"])
        observations = records["memory/observations.jsonl"]
        start_index = _ref_line(cursor.task_initial_observation_ref)
        if start_index is not None:
            observations = observations[start_index - 1 :]
        elif cursor.started_at:
            observations = _timestamp_slice(
                observations, cursor.started_at, cursor.ended_at
            )

        observation_ids = {
            str(item.get("id"))
            for item in observations
            if item.get("id") not in (None, "")
        }
        frame_refs = {
            ref
            for observation in observations
            for ref in _refs(observation.get("frame_refs"))
        }
        pose_refs = {
            ref
            for observation in observations
            for ref in _refs([observation.get("pose_ref")])
        }
        frames = [
            item
            for index, item in enumerate(
                records["memory/frames.jsonl"], start=1
            )
            if (
                f"memory/frames.jsonl#{index}" in frame_refs
                or str(item.get("observation_id")) in observation_ids
            )
        ]
        poses = [
            item
            for index, item in enumerate(
                records["memory/poses.jsonl"], start=1
            )
            if (
                f"memory/poses.jsonl#{index}" in pose_refs
                or str(item.get("observation_id")) in observation_ids
                or (
                    item.get("source") == "action_result"
                    and _in_time_window(
                        item, cursor.started_at, cursor.ended_at
                    )
                )
            )
        ]
        candidate_actions = [
            {
                **item,
                "_memory_ref": f"memory/actions.jsonl#{index}",
            }
            for index, item in enumerate(
                records["memory/actions.jsonl"], start=1
            )
            if _in_time_window(item, cursor.started_at, cursor.ended_at)
            and str(item.get("tool") or "") not in _NON_PLANNER_TOOLS
        ]
        transcript = _active_instruction_transcript(
            records["memory/agent_transcript.jsonl"],
            cursor.started_at,
            cursor.ended_at,
        )
        events = _timestamp_slice(
            records["events.jsonl"], cursor.started_at, cursor.ended_at
        )
        # Initial plans precede the task-initial camera observation. Retain them
        # by instruction identity, not a guessed earlier timestamp. Frozen line
        # bounds still exclude later tasks and post-freeze mutations.
        instruction_ids = {item.get("instruction_id") for item in transcript
                           if item.get("instruction_id")}
        early_plans = [item for item in records["events.jsonl"]
                       if item.get("event") in {"initial_semantic_plan", "instruction_started"}
                       and isinstance(item.get("payload"), dict)
                       and item["payload"].get("instruction_id") in instruction_ids
                       and item not in events]
        events = early_plans + events
        actions = _formal_planner_actions(candidate_actions, events)
        debug_events = _timestamp_slice(
            records["debug/debug_events.jsonl"],
            cursor.started_at,
            cursor.ended_at,
        )
        planner_debug, diagnostics = _planner_debug_records(run_dir)
        tool_debug, tool_diagnostics = _tool_loop_debug_records(run_dir, events)
        planner_debug.extend(tool_debug)
        diagnostics.extend(tool_diagnostics)
        planner_debug = _associate_phase_completion(
            planner_debug,
            events,
        )
        planner_debug, phase_window_diagnostics = (
            _associate_phase_window_execution(
                planner_debug,
                events,
                run_dir,
            )
        )
        diagnostics.extend(phase_window_diagnostics)
        return BenchmarkAuditLog(
            run_dir=str(run_dir),
            started_at=cursor.started_at,
            ended_at=cursor.ended_at,
            task_initial_observation_ref=(
                cursor.task_initial_observation_ref
            ),
            observations=tuple(observations),
            frames=tuple(frames),
            poses=tuple(poses),
            actions=tuple(actions),
            events=tuple(events),
            debug_events=tuple(debug_events),
            planner_debug=tuple(planner_debug),
            agent_transcript=tuple(transcript),
            tool_completions=tuple(
                item for item in records["memory/tool_completions.jsonl"]
                if (item.get("instruction_id") in instruction_ids if instruction_ids
                    else _in_time_window(item, cursor.started_at, cursor.ended_at))
            ),
            observation_anchors={
                key: dict(value)
                for key, value in cursor.observation_anchors.items()
            },
            diagnostics=tuple(diagnostics),
        )


class FrameAssetResolver:
    """Copy recorded frame data into a portable case review directory."""

    def materialize(
        self,
        audit: BenchmarkAuditLog,
        case_dir: str | Path,
    ) -> tuple[ResolvedFrameAsset, ...]:
        destination = Path(case_dir) / "review_assets" / "frames"
        results = [
            self._materialize_frame(
                frame,
                run_dir=Path(audit.run_dir),
                destination=destination,
            )
            for frame in audit.frames
        ]
        return tuple(results)

    def _materialize_frame(
        self,
        frame: dict[str, Any],
        *,
        run_dir: Path,
        destination: Path,
    ) -> ResolvedFrameAsset:
        frame_id = str(frame.get("id") or "frame")
        modality = str(frame.get("modality") or "unknown")
        metadata = (
            dict(frame.get("metadata"))
            if isinstance(frame.get("metadata"), dict)
            else {}
        )
        stem = _safe_name(f"{frame_id}_{modality}")
        display_source = (
            metadata.get("preview_path")
            if modality == "depth"
            else None
        )
        artifact_source = metadata.get("artifact_path") or frame.get("path")
        artifact_source_name = (
            "artifact_path"
            if metadata.get("artifact_path")
            else "path"
        )
        display_path = None
        raw_path = None
        source = None
        errors: list[str] = []

        if display_source:
            copied = _copy_recorded_path(
                display_source, run_dir, destination, stem + "_preview"
            )
            if copied is not None:
                display_path = str(copied)
                source = "preview_path"
            else:
                errors.append("preview_path_missing")

        if artifact_source:
            copied = _copy_recorded_path(
                artifact_source, run_dir, destination, stem
            )
            if copied is not None:
                raw_path = str(copied)
                if display_path is None and copied.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                }:
                    display_path = str(copied)
                source = source or artifact_source_name
            else:
                errors.append("artifact_path_missing")

        encoded = metadata.get("base64")
        if display_path is None and isinstance(encoded, str) and encoded:
            try:
                payload = base64.b64decode(encoded, validate=True)
                suffix = _payload_suffix(payload)
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / f"{stem}{suffix}"
                target.write_bytes(payload)
                raw_path = raw_path or str(target)
                if suffix in {".png", ".jpg", ".webp"}:
                    display_path = str(target)
                source = source or "base64"
            except (binascii.Error, ValueError, OSError):
                errors.append("invalid_base64")

        unavailable = None
        if display_path is None:
            unavailable = ",".join(errors) or "no_visual_payload"
        return ResolvedFrameAsset(
            frame_id=frame_id,
            observation_id=(
                str(frame.get("observation_id"))
                if frame.get("observation_id") not in (None, "")
                else None
            ),
            modality=modality,
            timestamp=(
                str(frame.get("timestamp"))
                if frame.get("timestamp") not in (None, "")
                else None
            ),
            display_path=display_path,
            raw_path=raw_path,
            source=source,
            unavailable_reason=unavailable,
        )


def write_audit_json(
    path: str | Path,
    audit: BenchmarkAuditLog,
    assets: tuple[ResolvedFrameAsset, ...],
) -> str:
    """Write a stable human-facing audit payload outside Runtime memory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = audit.to_dict()
    payload["frame_assets"] = [item.to_dict() for item in assets]
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(target)


def _copy_recorded_path(
    value: Any,
    run_dir: Path,
    destination: Path,
    stem: str,
) -> Path | None:
    if value in (None, ""):
        return None
    source = Path(str(value))
    candidates = [source]
    if not source.is_absolute():
        candidates.extend((run_dir / source, run_dir.parent / source))
    existing = next((item for item in candidates if item.is_file()), None)
    if existing is None:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{stem}{existing.suffix.lower()}"
    shutil.copy2(existing, target)
    return target


def _payload_suffix(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x93NUMPY"):
        return ".npy"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _tool_loop_debug_records(
    run_dir: Path, events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read only explicitly captured unified decisions inside the frozen events.

    Do not scan the tool-loop directory: later instructions or a partially
    written future turn are not evidence for this episode. IDs must agree in
    the event and artifact, and paths must remain inside the run directory.
    """
    records, diagnostics = [], []
    root = run_dir.resolve()
    for event in events:
        if (event.get("event") or event.get("kind")) != "model_turn_recorded":
            continue
        link = event.get("payload")
        if not isinstance(link, dict):
            continue
        try:
            ref = link.get("debug_ref")
            if not isinstance(ref, str) or not ref:
                raise ValueError("missing debug_ref")
            path = (root / ref).resolve()
            if root not in path.parents:
                raise ValueError("debug_ref_outside_run_dir")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(payload, dict)
                    or payload.get("loop_schema") != "unified_tool_loop_v2"
                    or not link.get("decision_id")
                    or payload.get("decision_id") != link.get("decision_id")
                    or (link.get("instruction_id") is not None
                        and payload.get("instruction_id") != link["instruction_id"])):
                raise ValueError("tool_loop_capture_identity_mismatch")
            records.append({**payload, "path": str(path)})
        except (OSError, ValueError) as exc:
            diagnostics.append({"stage": "tool_loop_debug_read", "debug_ref": link.get("debug_ref"),
                                "error_type": type(exc).__name__, "error": str(exc)})
    return records, diagnostics


def _planner_debug_records(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    folder = run_dir / "debug" / "action_planner"
    if not folder.exists():
        return records, diagnostics
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                records.append({"path": str(path), **payload})
        except (OSError, json.JSONDecodeError) as exc:
            diagnostics.append(
                {
                    "stage": "planner_debug_read",
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if any(isinstance(item.get("decision_index"), int) for item in records):
        records.sort(
            key=lambda item: (
                item.get("decision_index")
                if isinstance(item.get("decision_index"), int)
                else 10**9,
                str(item.get("path") or ""),
            )
        )
    return records, diagnostics


def _associate_phase_completion(
    planner_records: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach Runtime's applied completion result to its planner window.

    The planner debug file contains model and safety details, while the event
    stream records the transition that Runtime actually derived and applied.
    Keeping the join in the benchmark-neutral audit reader lets every
    benchmark render the same truthful window view.
    """

    completions: dict[tuple[str, int], dict[str, Any]] = {}
    completions_by_decision: dict[str, dict[str, Any]] = {}
    for item in events:
        if (item.get("event") or item.get("kind")) != (
            "phase_completion_assessed"
        ):
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        decision_id = payload.get("decision_id")
        if decision_id not in (None, ""):
            completions_by_decision[str(decision_id)] = dict(payload)
        window_index = payload.get("window_index")
        if not isinstance(window_index, int):
            continue
        for phase in (payload.get("phase_key"), payload.get("phase")):
            if phase not in (None, ""):
                completions[(str(phase), window_index)] = dict(payload)

    enriched: list[dict[str, Any]] = []
    for record in planner_records:
        item = dict(record)
        decision_id = item.get("decision_id")
        if decision_id not in (None, ""):
            completion = completions_by_decision.get(str(decision_id))
            if completion is not None:
                item["_phase_completion_result"] = completion
                enriched.append(item)
                continue
        window_index = item.get("window_index")
        if isinstance(window_index, int):
            for phase in (item.get("phase_key"), item.get("phase")):
                completion = completions.get((str(phase), window_index))
                if completion is not None:
                    item["_phase_completion_result"] = completion
                    break
        enriched.append(item)
    return enriched


def _associate_phase_window_execution(
    planner_records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach Runtime-recorded execution outcomes to matching planner windows.

    Planner debug captures the model proposal and safety-reviewed plan. Runtime
    writes the authoritative per-window ``executed_steps`` separately after
    tool execution. The benchmark review needs both views side by side, but
    this reader keeps the join read-only and bounded to the frozen episode.
    """

    executions: dict[tuple[str, int], dict[str, Any]] = {}
    executions_by_decision: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for event in events:
        if (event.get("event") or event.get("kind")) not in PHYSICAL_WINDOW_EVENTS:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        window_index = payload.get("window_index")
        debug_ref = payload.get("debug_ref")
        if not isinstance(window_index, int) or not isinstance(debug_ref, str):
            continue
        execution = _read_phase_window_execution(
            run_dir,
            debug_ref,
            diagnostics,
        )
        if execution is None:
            continue
        decision_id = payload.get("decision_id")
        if decision_id not in (None, ""):
            executions_by_decision[str(decision_id)] = execution
        for phase in (payload.get("phase_key"), payload.get("phase")):
            if phase not in (None, ""):
                executions[(str(phase), window_index)] = execution

    enriched: list[dict[str, Any]] = []
    for record in planner_records:
        item = dict(record)
        decision_id = item.get("decision_id")
        if decision_id not in (None, ""):
            execution = executions_by_decision.get(str(decision_id))
            if execution is not None:
                item["_phase_window_execution"] = execution
            enriched.append(item)
            continue
        window_index = item.get("window_index")
        if isinstance(window_index, int):
            for phase in (item.get("phase_key"), item.get("phase")):
                execution = executions.get((str(phase), window_index))
                if execution is not None:
                    item["_phase_window_execution"] = execution
                    break
        enriched.append(item)
    return enriched, diagnostics


def _read_phase_window_execution(
    run_dir: Path,
    debug_ref: str,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Read only the execution fields review needs from a Runtime debug file."""

    root = run_dir.resolve()
    candidate = (root / debug_ref).resolve()
    if root != candidate and root not in candidate.parents:
        diagnostics.append(
            {
                "stage": "phase_window_execution_read",
                "debug_ref": debug_ref,
                "reason": "debug_ref_outside_run_dir",
            }
        )
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        diagnostics.append(
            {
                "stage": "phase_window_execution_read",
                "debug_ref": debug_ref,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return None
    if not isinstance(payload, dict):
        diagnostics.append(
            {
                "stage": "phase_window_execution_read",
                "debug_ref": debug_ref,
                "reason": "payload_not_object",
            }
        )
        return None
    return {
        "debug_ref": debug_ref,
        "executed_steps": payload.get("executed_steps"),
        "safety_interventions": payload.get("safety_interventions"),
    }


def _formal_planner_actions(
    candidates: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep actions emitted by formal Planner tool steps.

    Runtime safety and shutdown may call a Planner-visible tool such as
    ``uav.hover`` without making it part of the Agent action history. A tool
    name allowlist cannot distinguish those calls, so current runs are joined
    to ``tool_step_completed`` events. If an older structured client did not
    record step events, the bounded candidate list remains the compatibility
    fallback.
    """

    declared_v2 = any(
        isinstance(event.get("payload"), dict)
        and (
            ((event.get("event") or event.get("kind")) == "instruction_started"
             and event["payload"].get("physical_trace_schema") == "physical_window_v2")
            # The instruction marker can precede the initial camera capture.
            # An explicit versioned window is also authoritative; never fall
            # back to timestamps/name allowlists and include read-only calls.
            or ((event.get("event") or event.get("kind")) in PHYSICAL_WINDOW_EVENTS
                and event["payload"].get("schema_version") == "physical_window_v2")
        )
        for event in events
    )
    if declared_v2:
        ordered_refs = []
        for event in events:
            if (event.get("event") or event.get("kind")) not in PHYSICAL_WINDOW_EVENTS:
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict) or payload.get("schema_version") != "physical_window_v2":
                continue
            steps = payload.get("ordered_steps")
            if not isinstance(steps, list):
                continue
            ordered_refs.extend(
                str(step.get("action_ref"))
                for step in steps
                if isinstance(step, dict) and step.get("action_ref")
            )
        by_ref = {
            str(item.get("_memory_ref")): item
            for item in candidates
            if item.get("_memory_ref")
        }
        return [
            {key: value for key, value in by_ref[ref].items() if key != "_memory_ref"}
            for ref in ordered_refs
            if ref in by_ref
        ]

    steps = []
    for event in events:
        if (event.get("event") or event.get("kind")) != (
            "tool_step_completed"
        ):
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            steps.append((event, payload))
    if not steps:
        return [
            {key: value for key, value in item.items() if key != "_memory_ref"}
            for item in candidates
        ]

    available = list(enumerate(candidates))
    used: set[int] = set()
    formal: list[dict[str, Any]] = []
    for event, payload in steps:
        tool = str(payload.get("tool") or "")
        event_timestamp = _timestamp(event)
        matching = [
            (index, action)
            for index, action in available
            if index not in used
            and str(action.get("tool") or "") == tool
            and (
                not event_timestamp
                or not _timestamp(action)
                or _timestamp(action) <= event_timestamp
            )
        ]
        if not matching:
            matching = [
                (index, action)
                for index, action in available
                if index not in used
                and str(action.get("tool") or "") == tool
            ]
        if matching:
            index, action = max(
                matching,
                key=lambda item: _timestamp(item[1]),
            )
            used.add(index)
            formal.append(
                {
                    **{
                        key: value
                        for key, value in action.items()
                        if key != "_memory_ref"
                    },
                    "phase": payload.get("phase"),
                    "step_index": payload.get("step_index"),
                    "tool_step_timestamp": event.get("timestamp"),
                }
            )
            continue
        formal.append(
            {
                "id": None,
                "timestamp": event.get("timestamp"),
                "tool": tool,
                "args": {},
                "result": {"ok": payload.get("ok")},
                "phase": payload.get("phase"),
                "step_index": payload.get("step_index"),
                "source": "tool_step_completed",
            }
        )
    return formal


def _observation_anchors(
    run_dir: Path,
) -> dict[str, dict[str, Any]]:
    path = run_dir / "state" / "flight_memory.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    anchors = payload.get("observation_anchors")
    if not isinstance(anchors, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, value in anchors.items():
        if isinstance(value, dict) and isinstance(value.get("ref"), str):
            result[str(name)] = dict(value)
        elif isinstance(value, str):
            result[str(name)] = {"ref": value}
    for anchor in result.values():
        line = _ref_line(str(anchor.get("ref") or ""))
        if line is None:
            continue
        records = _read_jsonl(
            run_dir / "memory" / "observations.jsonl", line
        )
        if line > len(records):
            continue
        observation = records[line - 1]
        anchor.setdefault("id", observation.get("id"))
        anchor.setdefault("timestamp", observation.get("timestamp"))
    return result


def _observation_timestamp(run_dir: Path, ref: str | None) -> str | None:
    line = _ref_line(ref)
    if line is None:
        return None
    records = _read_jsonl(run_dir / "memory" / "observations.jsonl", line)
    if line > len(records):
        return None
    value = records[line - 1].get("timestamp")
    return str(value) if value not in (None, "") else None


def _refs(values: Any) -> set[str]:
    refs: set[str] = set()
    if not isinstance(values, list):
        return refs
    for value in values:
        if isinstance(value, str):
            refs.add(value)
        elif isinstance(value, dict) and isinstance(value.get("ref"), str):
            refs.add(value["ref"])
    return refs


def _ref_line(ref: str | None) -> int | None:
    if not ref or "#" not in ref:
        return None
    try:
        return int(ref.rsplit("#", 1)[1])
    except ValueError:
        return None


def _timestamp_slice(
    records: list[dict[str, Any]],
    started_at: str | None,
    ended_at: str,
) -> list[dict[str, Any]]:
    if started_at is None:
        return [item for item in records if _timestamp(item) <= ended_at]
    return [
        item
        for item in records
        if started_at <= _timestamp(item) <= ended_at
    ]


def _active_instruction_transcript(
    records: list[dict[str, Any]],
    started_at: str | None,
    ended_at: str,
) -> list[dict[str, Any]]:
    """Return the transcript owning the bounded episode.

    Runtime starts the transcript before it captures the task-initial
    observation, so a plain timestamp slice would drop the user instruction.
    The first transcript record at or after the observation boundary identifies
    the active ``instruction_id``; all bounded records with that ID are kept.
    V2 records always have instruction IDs. The cursor's line limits have
    already excluded anything appended after freeze.
    """

    bounded = [item for item in records if _timestamp(item) <= ended_at]
    if not bounded:
        return []
    candidates = [
        item
        for item in bounded
        if started_at is None or _timestamp(item) >= started_at
    ]
    instruction_id = next(
        (
            str(item.get("instruction_id"))
            for item in candidates
            if item.get("instruction_id") not in (None, "")
        ),
        None,
    )
    if instruction_id is not None:
        return [
            item
            for item in bounded
            if str(item.get("instruction_id") or "") == instruction_id
        ]

    raise UnsupportedRunSchema("agent_transcript_v2 requires instruction_id")


def _require_v2_transcript(records: list[dict[str, Any]]) -> None:
    for record in records:
        if record.get("schema_version") != TRANSCRIPT_SCHEMA:
            raise UnsupportedRunSchema(
                "memory/agent_transcript.jsonl is not agent_transcript_v2"
            )
        kind = record.get("kind")
        payload = record.get("payload")
        required_payload_schema = {
            "instruction_envelope": INSTRUCTION_ENVELOPE_SCHEMA,
            "state_checkpoint": STATE_CHECKPOINT_SCHEMA,
        }.get(kind)
        if required_payload_schema is not None and (
            not isinstance(payload, dict)
            or payload.get("schema_version") != required_payload_schema
        ):
            raise UnsupportedRunSchema(
                f"{kind} payload is not {required_payload_schema}"
            )


def _in_time_window(
    record: dict[str, Any],
    started_at: str | None,
    ended_at: str,
) -> bool:
    timestamp = _timestamp(record)
    return timestamp <= ended_at and (
        started_at is None or timestamp >= started_at
    )


def _timestamp(record: dict[str, Any]) -> str:
    return str(record.get("timestamp") or "")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle, start=1):
            if index > limit:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def _safe_name(value: str) -> str:
    text = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in value
    ).strip("_")
    return text or "frame"
