"""Read one benchmark trajectory from append-only Runtime memory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_MEMORY_FILES = ("actions.jsonl", "poses.jsonl", "observations.jsonl")


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
                filename: _line_count(path / "memory" / filename)
                for filename in _MEMORY_FILES
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
    """One initial or Planner-action trajectory sample."""

    pose: dict[str, Any]
    source: str
    tool: str | None
    action_id: str | None
    timestamp: str | None
    coordinate_frame: str
    action_args: dict[str, Any] = field(default_factory=dict)
    post_action_observation_pose: dict[str, Any] | None = None
    post_action_observation_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeTrace:
    """Frozen episode trajectory with source provenance."""

    samples: tuple[EpisodeTraceSample, ...]
    action_count: int
    coordinate_frames: tuple[str, ...]

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
            "samples": [sample.to_dict() for sample in self.samples],
            "action_count": self.action_count,
            "coordinate_frames": list(self.coordinate_frames),
        }


class EpisodeTraceReader:
    """Build action-aligned samples bounded by a previously frozen cursor."""

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
        initial_ref = cursor.task_initial_observation_ref or _task_initial_ref(
            run_dir
        )
        if initial_ref is None and observations:
            # One executor owns one Runtime/run directory in benchmark v1.
            # This also keeps trace reading deterministic while the async
            # state manifest is still being flushed by the Runtime recorder.
            initial_ref = "memory/observations.jsonl#1"
        initial_observation = _record_from_ref(initial_ref, observations)
        if initial_observation is None:
            return EpisodeTrace(samples=(), action_count=0, coordinate_frames=())

        initial_pose = _record_from_ref(
            _nested_ref(initial_observation.get("pose_ref")), poses
        )
        if initial_pose is None:
            return EpisodeTrace(samples=(), action_count=0, coordinate_frames=())

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
        formal_actions = [
            action
            for action in actions
            if str(action.get("timestamp") or "") >= initial_timestamp
            and _is_formal_action(action)
            and not _is_trailing_runtime_hover(action, observations)
        ]
        for action in formal_actions:
            action_id = str(action.get("id") or "")
            observation_pose_record = _first_observation_pose_after(
                action, observations, poses
            )
            pose_record = next(
                (
                    pose
                    for pose in poses
                    if pose.get("source") == "action_result"
                    and str(pose.get("action_id") or "") == action_id
                ),
                None,
            )
            source = "action_result"
            if pose_record is None:
                pose_record = observation_pose_record
                source = "observation_fallback"
            if pose_record is None:
                continue
            samples.append(
                _sample_from_pose_record(
                    pose_record,
                    source=source,
                    tool=str(action.get("tool") or ""),
                    action_id=action_id,
                    timestamp=str(action.get("timestamp") or "") or None,
                    action_args=(
                        dict(action.get("args"))
                        if isinstance(action.get("args"), dict)
                        else {}
                    ),
                    post_action_observation_pose=(
                        _pose_data(observation_pose_record)
                        if observation_pose_record is not None
                        else None
                    ),
                    post_action_observation_timestamp=(
                        str(observation_pose_record.get("timestamp") or "")
                        or None
                        if observation_pose_record is not None
                        else None
                    ),
                )
            )
        frames = {
            sample.coordinate_frame for sample in samples
        }
        frames.update(
            str(sample.post_action_observation_pose.get("coordinate_frame"))
            for sample in samples
            if sample.post_action_observation_pose is not None
            and sample.post_action_observation_pose.get("coordinate_frame")
        )
        return EpisodeTrace(
            samples=tuple(samples),
            action_count=len(formal_actions),
            coordinate_frames=tuple(sorted(frames)),
        )


def _is_formal_action(action: dict[str, Any]) -> bool:
    tool = str(action.get("tool") or "")
    return tool not in {
        "uav.connect",
        "uav.takeoff",
        "uav.land",
        "uav.set_pose",
        "uav.get_pose",
        "uav.get_frames",
    }


def _is_trailing_runtime_hover(
    action: dict[str, Any], observations: list[dict[str, Any]]
) -> bool:
    """Exclude Runtime's best-effort error hover, not Planner hover steps."""

    if action.get("tool") != "uav.hover" or not observations:
        return False
    last_observation_timestamp = max(
        str(item.get("timestamp") or "") for item in observations
    )
    return str(action.get("timestamp") or "") > last_observation_timestamp


def _first_observation_pose_after(
    action: dict[str, Any],
    observations: list[dict[str, Any]],
    poses: list[dict[str, Any]],
) -> dict[str, Any] | None:
    timestamp = str(action.get("timestamp") or "")
    for observation in observations:
        if str(observation.get("timestamp") or "") < timestamp:
            continue
        pose = _record_from_ref(_nested_ref(observation.get("pose_ref")), poses)
        if pose is not None:
            return pose
    return None


def _sample_from_pose_record(
    record: dict[str, Any],
    *,
    source: str,
    tool: str | None,
    action_id: str | None,
    timestamp: str | None,
    action_args: dict[str, Any] | None = None,
    post_action_observation_pose: dict[str, Any] | None = None,
    post_action_observation_timestamp: str | None = None,
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
        post_action_observation_pose=(
            dict(post_action_observation_pose)
            if post_action_observation_pose is not None
            else None
        ),
        post_action_observation_timestamp=post_action_observation_timestamp,
    )


def _pose_data(record: dict[str, Any]) -> dict[str, Any]:
    """Return one normalized pose payload without changing trace semantics."""

    pose = record.get("pose")
    pose_data = dict(pose) if isinstance(pose, dict) else {}
    frame = str(
        record.get("coordinate_frame")
        or pose_data.get("coordinate_frame")
        or "canonical_unspecified"
    )
    # The recorder-level coordinate frame is authoritative when duplicated
    # inside the pose payload.
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
