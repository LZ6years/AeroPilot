"""Build benchmark trajectories from explicit Runtime physical windows.

Runtime observes once after an ordered physical batch. This reader therefore
keeps per-step action-result poses for trajectory export while using the
window's explicit before/after observations for settled backend tracking. It
never associates actions and observations by timestamp.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PHYSICAL_TRACE_SCHEMA_VERSION = "physical_window_v2"
_MEMORY_FILES = ("actions.jsonl", "poses.jsonl", "observations.jsonl")
_MOTION_TOOLS = {"uav.move_body", "uav.turn_relative"}


@dataclass(frozen=True)
class TraceCursor:
    """Freeze the inclusive end of an episode before Runtime shutdown."""

    run_dir: str
    end_lines: dict[str, int]
    task_initial_observation_ref: str | None = None

    @classmethod
    def freeze(cls, run_dir: str | Path) -> "TraceCursor":
        path = Path(run_dir)
        return cls(
            run_dir=str(path),
            end_lines={
                **{
                    filename: _line_count(path / "memory" / filename)
                    for filename in _MEMORY_FILES
                },
                "events.jsonl": _line_count(path / "events.jsonl"),
            },
            task_initial_observation_ref=_task_initial_ref(path),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "end_lines": dict(self.end_lines),
            "task_initial_observation_ref": self.task_initial_observation_ref,
        }


@dataclass(frozen=True)
class EpisodeTraceSample:
    """One initial or formally executed physical-step trajectory sample."""

    pose: dict[str, Any]
    source: str
    tool: str | None
    action_id: str | None
    timestamp: str | None
    coordinate_frame: str
    action_args: dict[str, Any] = field(default_factory=dict)
    # Retained for external constructor compatibility. New window-v2 traces
    # keep settled observations on EpisodePhysicalWindow instead.
    post_action_observation_pose: dict[str, Any] | None = None
    post_action_observation_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodePhysicalStep:
    """One ordered physical call and its durable Runtime references."""

    step_index: int
    tool_call_id: str | None
    requested_tool: str
    tool: str
    effect: str
    status: str
    action_ref: str | None
    result_pose_ref: str | None
    action_id: str | None
    requested_args: dict[str, Any] = field(default_factory=dict)
    executed_args: dict[str, Any] | None = None
    safety_interventions: tuple[dict[str, Any], ...] = ()
    acknowledged_pose: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodePhysicalWindow:
    """One physical batch with exactly one settled observation boundary."""

    decision_id: str
    phase: str | None
    phase_key: str | None
    window_index: int
    status: str
    observation_before_id: str | None
    observation_after_id: str | None
    observation_before_pose: dict[str, Any] | None
    observation_after_pose: dict[str, Any] | None
    observation_after_timestamp: str | None
    ordered_steps: tuple[EpisodePhysicalStep, ...]
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeTrace:
    """Frozen trajectory plus explicit physical-window provenance."""

    samples: tuple[EpisodeTraceSample, ...]
    action_count: int
    coordinate_frames: tuple[str, ...]
    physical_windows: tuple[EpisodePhysicalWindow, ...] = ()
    physical_call_count: int = 0
    physical_window_count: int = 0
    motion_call_count: int = 0
    trace_schema_version: str | None = None
    trace_integrity: str = "unsupported_legacy_trace_schema"
    trace_issues: tuple[str, ...] = ()

    @property
    def poses(self) -> list[dict[str, Any]]:
        return [dict(sample.pose) for sample in self.samples]

    def require_single_coordinate_frame(
        self, expected: str | None = None
    ) -> str:
        if len(self.coordinate_frames) != 1:
            frames = ", ".join(self.coordinate_frames) or "<none>"
            raise ValueError(
                f"Episode trace requires one explicit coordinate frame; got {frames}"
            )
        frame = self.coordinate_frames[0]
        if frame == "canonical_unspecified":
            raise ValueError("Episode trace coordinate frame is unspecified")
        if expected is not None and frame != expected:
            raise ValueError(
                f"Episode trace coordinate frame {frame!r} does not match {expected!r}"
            )
        return frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema_version": self.trace_schema_version,
            "trace_integrity": self.trace_integrity,
            "trace_issues": list(self.trace_issues),
            "samples": [sample.to_dict() for sample in self.samples],
            # action_count is the compatibility alias consumed by other
            # benchmark adapters. It now means formal physical calls only.
            "action_count": self.action_count,
            "physical_call_count": self.physical_call_count,
            "physical_window_count": self.physical_window_count,
            "motion_call_count": self.motion_call_count,
            "physical_windows": [
                window.to_dict() for window in self.physical_windows
            ],
            "coordinate_frames": list(self.coordinate_frames),
        }


class EpisodeTraceReader:
    """Read only explicitly versioned physical-window evidence."""

    def read(self, cursor: TraceCursor) -> EpisodeTrace:
        run_dir = Path(cursor.run_dir)
        observations = _read_jsonl(
            run_dir / "memory" / "observations.jsonl",
            cursor.end_lines.get("observations.jsonl", 0),
        )
        poses = _read_jsonl(
            run_dir / "memory" / "poses.jsonl",
            cursor.end_lines.get("poses.jsonl", 0),
        )
        actions = _read_jsonl(
            run_dir / "memory" / "actions.jsonl",
            cursor.end_lines.get("actions.jsonl", 0),
        )
        events = _read_jsonl(
            run_dir / "events.jsonl",
            cursor.end_lines.get("events.jsonl", 0),
        )
        initial_ref = cursor.task_initial_observation_ref or _task_initial_ref(
            run_dir
        )
        if initial_ref is None and observations:
            initial_ref = "memory/observations.jsonl#1"
        initial_observation = _record_from_ref(initial_ref, observations)
        initial_pose = (
            _record_from_ref(
                _nested_ref(initial_observation.get("pose_ref")), poses
            )
            if initial_observation is not None
            else None
        )
        if initial_observation is None or initial_pose is None:
            return EpisodeTrace(
                samples=(),
                action_count=0,
                coordinate_frames=(),
                trace_issues=("task_initial_observation_unavailable",),
            )

        initial_timestamp = str(initial_observation.get("timestamp") or "")
        samples = [
            _sample_from_pose_record(
                initial_pose,
                source="task_initial",
                tool=None,
                action_id=None,
                timestamp=initial_timestamp or None,
            )
        ]
        declared_schema = _declared_trace_schema(events)
        if declared_schema != PHYSICAL_TRACE_SCHEMA_VERSION:
            return _finish_trace(
                samples,
                windows=(),
                physical_call_count=0,
                motion_call_count=0,
                schema=declared_schema,
                integrity="unsupported_legacy_trace_schema",
                issues=("physical_window_v2_not_declared",),
            )

        observation_by_id = {
            str(item.get("id")): item
            for item in observations
            if item.get("id") not in (None, "")
        }
        windows: list[EpisodePhysicalWindow] = []
        issues: list[str] = []
        physical_call_count = 0
        motion_call_count = 0
        for event in events:
            if (event.get("event") or event.get("kind")) != "phase_window_recorded":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                issues.append("phase_window_payload_not_object")
                continue
            if payload.get("schema_version") != PHYSICAL_TRACE_SCHEMA_VERSION:
                issues.append("phase_window_schema_mismatch")
                continue
            window, window_samples, window_issues = _read_physical_window(
                payload,
                observations=observation_by_id,
                poses=poses,
                actions=actions,
            )
            windows.append(window)
            samples.extend(window_samples)
            issues.extend(window_issues)
            physical_call_count += len(window.ordered_steps)
            motion_call_count += sum(
                step.status == "ok" and step.tool in _MOTION_TOOLS
                for step in window.ordered_steps
            )

        return _finish_trace(
            samples,
            windows=tuple(windows),
            physical_call_count=physical_call_count,
            motion_call_count=motion_call_count,
            schema=declared_schema,
            integrity="ok" if not issues else "invalid",
            issues=tuple(issues),
        )


def _read_physical_window(
    payload: dict[str, Any],
    *,
    observations: dict[str, dict[str, Any]],
    poses: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> tuple[EpisodePhysicalWindow, list[EpisodeTraceSample], list[str]]:
    issues: list[str] = []
    before_id = _optional_string(payload.get("observation_before_id"))
    after_id = _optional_string(payload.get("observation_after_id"))
    before_observation = observations.get(before_id or "")
    after_observation = observations.get(after_id or "")
    before_pose_record = _observation_pose(before_observation, poses)
    after_pose_record = _observation_pose(after_observation, poses)
    if before_id is None or before_observation is None or before_pose_record is None:
        issues.append("window_before_observation_unavailable")
    if after_id is None or after_observation is None or after_pose_record is None:
        issues.append("window_after_observation_unavailable")
    if before_id is not None and before_id == after_id:
        issues.append("window_observation_not_advanced")

    raw_steps = payload.get("ordered_steps")
    if not isinstance(raw_steps, list):
        raw_steps = []
        issues.append("window_ordered_steps_invalid")
    steps: list[EpisodePhysicalStep] = []
    trajectory_candidates: list[tuple[EpisodePhysicalStep, dict[str, Any] | None]] = []
    seen_indexes: set[int] = set()
    for ordinal, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            issues.append("physical_step_not_object")
            continue
        step_index = item.get("step_index")
        if not isinstance(step_index, int) or step_index < 0:
            step_index = ordinal
            issues.append("physical_step_index_invalid")
        if step_index in seen_indexes or step_index != ordinal:
            issues.append("physical_step_order_invalid")
        seen_indexes.add(step_index)
        tool = str(item.get("tool") or "")
        effect = str(item.get("effect") or "")
        if not tool or effect != "physical":
            issues.append("non_physical_step_in_window")
        status = str(item.get("status") or "")
        if status not in {"ok", "error", "skipped"}:
            issues.append("physical_step_status_invalid")
        action_ref = _optional_string(item.get("action_ref"))
        pose_ref = _optional_string(item.get("result_pose_ref"))
        action = _record_from_ref(action_ref, actions)
        acknowledged_pose_record = _record_from_ref(pose_ref, poses)
        if status in {"ok", "error"} and action is None:
            issues.append("physical_step_action_ref_unavailable")
        if action is not None and str(action.get("tool") or "") != tool:
            issues.append("physical_step_action_tool_mismatch")
        executed_args = item.get("executed_args")
        if executed_args is not None and not isinstance(executed_args, dict):
            issues.append("physical_step_executed_args_invalid")
            executed_args = None
        requested_args = item.get("requested_args")
        if not isinstance(requested_args, dict):
            issues.append("physical_step_requested_args_invalid")
            requested_args = {}
        interventions = item.get("safety_interventions")
        if not isinstance(interventions, list):
            issues.append("physical_step_safety_interventions_invalid")
            interventions = []
        step = EpisodePhysicalStep(
            step_index=step_index,
            tool_call_id=_optional_string(item.get("tool_call_id")),
            requested_tool=str(item.get("requested_tool") or tool),
            tool=tool,
            effect=effect,
            status=status,
            action_ref=action_ref,
            result_pose_ref=pose_ref,
            action_id=(
                _optional_string(action.get("id")) if action is not None else None
            ),
            requested_args=dict(requested_args),
            executed_args=(dict(executed_args) if executed_args is not None else None),
            safety_interventions=tuple(
                dict(value) for value in interventions if isinstance(value, dict)
            ),
            acknowledged_pose=(
                _pose_data(acknowledged_pose_record)
                if acknowledged_pose_record is not None
                else None
            ),
        )
        steps.append(step)
        trajectory_candidates.append((step, acknowledged_pose_record))

    successful = [
        (step, pose_record)
        for step, pose_record in trajectory_candidates
        if step.status == "ok"
    ]
    window_samples: list[EpisodeTraceSample] = []
    for success_index, (step, pose_record) in enumerate(successful):
        source = "action_result"
        if pose_record is None and success_index == len(successful) - 1:
            pose_record = after_pose_record
            source = "window_observation_fallback"
        if pose_record is None:
            issues.append("physical_step_result_pose_unavailable")
            continue
        window_samples.append(
            _sample_from_pose_record(
                pose_record,
                source=source,
                tool=step.tool,
                action_id=step.action_id,
                timestamp=(
                    str(after_observation.get("timestamp") or "") or None
                    if source == "window_observation_fallback"
                    and after_observation is not None
                    else None
                ),
                action_args=step.executed_args or {},
            )
        )

    unavailable_reason = None
    if before_pose_record is None:
        unavailable_reason = "pre_window_observation_unavailable"
    elif after_pose_record is None:
        unavailable_reason = "post_window_observation_unavailable"
    elif not steps:
        unavailable_reason = "physical_steps_unavailable"
    window_index = payload.get("window_index")
    if not isinstance(window_index, int):
        issues.append("physical_window_index_invalid")
        window_index = 0
        unavailable_reason = unavailable_reason or "window_index_unavailable"
    window = EpisodePhysicalWindow(
        decision_id=str(payload.get("decision_id") or ""),
        phase=_optional_string(payload.get("phase")),
        phase_key=_optional_string(payload.get("phase_key")),
        window_index=window_index,
        status=str(payload.get("status") or "unknown"),
        observation_before_id=before_id,
        observation_after_id=after_id,
        observation_before_pose=(
            _pose_data(before_pose_record) if before_pose_record is not None else None
        ),
        observation_after_pose=(
            _pose_data(after_pose_record) if after_pose_record is not None else None
        ),
        observation_after_timestamp=(
            str(after_observation.get("timestamp") or "") or None
            if after_observation is not None
            else None
        ),
        ordered_steps=tuple(steps),
        unavailable_reason=unavailable_reason,
    )
    return window, window_samples, issues


def _finish_trace(
    samples: list[EpisodeTraceSample],
    *,
    windows: tuple[EpisodePhysicalWindow, ...],
    physical_call_count: int,
    motion_call_count: int,
    schema: str | None,
    integrity: str,
    issues: tuple[str, ...],
) -> EpisodeTrace:
    frames = {sample.coordinate_frame for sample in samples}
    for window in windows:
        for pose in (
            window.observation_before_pose,
            window.observation_after_pose,
        ):
            if pose and pose.get("coordinate_frame"):
                frames.add(str(pose["coordinate_frame"]))
    return EpisodeTrace(
        samples=tuple(samples),
        action_count=physical_call_count,
        coordinate_frames=tuple(sorted(frames)),
        physical_windows=windows,
        physical_call_count=physical_call_count,
        physical_window_count=len(windows),
        motion_call_count=motion_call_count,
        trace_schema_version=schema,
        trace_integrity=integrity,
        trace_issues=issues,
    )


def _declared_trace_schema(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if (event.get("event") or event.get("kind")) != "instruction_started":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and payload.get("physical_trace_schema"):
            return str(payload["physical_trace_schema"])
    return None


def _observation_pose(
    observation: dict[str, Any] | None,
    poses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if observation is None:
        return None
    return _record_from_ref(_nested_ref(observation.get("pose_ref")), poses)


def _sample_from_pose_record(
    record: dict[str, Any],
    *,
    source: str,
    tool: str | None,
    action_id: str | None,
    timestamp: str | None,
    action_args: dict[str, Any] | None = None,
) -> EpisodeTraceSample:
    pose_data = _pose_data(record)
    frame = str(
        record.get("coordinate_frame")
        or pose_data.get("coordinate_frame")
        or "canonical_unspecified"
    )
    pose_data.setdefault("coordinate_frame", frame)
    return EpisodeTraceSample(
        pose=pose_data,
        source=source,
        tool=tool,
        action_id=action_id,
        timestamp=timestamp or str(record.get("timestamp") or "") or None,
        coordinate_frame=frame,
        action_args=dict(action_args or {}),
    )


def _pose_data(record: dict[str, Any]) -> dict[str, Any]:
    pose = record.get("pose")
    pose_data = dict(pose) if isinstance(pose, dict) else {}
    frame = str(
        record.get("coordinate_frame")
        or pose_data.get("coordinate_frame")
        or "canonical_unspecified"
    )
    pose_data["coordinate_frame"] = frame
    return pose_data


def _task_initial_ref(run_dir: Path) -> str | None:
    path = run_dir / "state" / "flight_memory.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    anchors = payload.get("observation_anchors", {})
    anchor = anchors.get("task_initial") if isinstance(anchors, dict) else None
    return _nested_ref(anchor)


def _nested_ref(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        ref = value.get("ref")
        return str(ref) if isinstance(ref, str) else None
    return None


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _record_from_ref(
    ref: str | None, records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not ref or "#" not in ref:
        return None
    try:
        line_number = int(ref.rsplit("#", 1)[1])
    except ValueError:
        return None
    index = line_number - 1
    if index < 0 or index >= len(records):
        return None
    return records[index]


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
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records
