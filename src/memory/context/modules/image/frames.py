"""Frame-record selection and binary-free MediaRef construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory.context.contracts import MediaRef
from src.memory.context.media_refs import run_relative_media_ref
from src.memory.context.modules.image.overlay import CURRENT_RGB_DEPTH_OVERLAY_USAGE

CURRENT_IMAGE_USAGE = (
    "Original RGB preserves scene colors. This image records the view at its labeled observation."
)

def _rgb_media_inputs_from_frames(
    frames: list[dict[str, Any]],
    *,
    run_dir: str | Path | None,
) -> list[dict[str, Any]]:
    """Return one run-relative current MediaRef without reading media bytes."""

    frame = frames[-1] if frames else None
    if not frame or frame.get("mock") is True:
        return []
    digest = str(frame.get("sha256") or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return []
    width = _positive_int(frame.get("width"))
    height = _positive_int(frame.get("height"))
    media_path = str(frame.get("artifact_path") or frame.get("path") or "")
    if not media_path:
        return []
    if run_dir is None:
        raise ValueError("current RGB media requires an owning run_dir")
    media = MediaRef(
        ref=run_relative_media_ref(media_path, run_dir),
        sha256=digest,
        mime_type="image/png",
        observation_id=str(frame.get("observation_id") or ""),
        frame_id=str(frame.get("id") or ""),
        role=str(frame.get("selection_role") or "current_observation"),
        usage=str(frame.get("usage") or frame.get("selection_reason") or ""),
        width=width,
        height=height,
        display_width=width,
        display_height=height,
    )
    return [media.to_model_input()]


class _BuildFrameReader:
    """Reuse parsed frame batches for overlapping traversals in one build.

    Keys are the actual frame_refs list identities from the isolated observation
    records, not serialized paths. Retaining the list prevents identity reuse.
    This preserves custom readers' batch semantics and the first access order.
    Incomplete batches are retried so a missing record is never cached. Helpers
    derive fresh frame dictionaries and must not mutate these private records.
    """

    def __init__(self, reader: Any) -> None:
        self.reader = reader
        self.run_dir = getattr(reader, "run_dir", None)
        self._batches: dict[int, tuple[Any, list[dict[str, Any]]]] = {}

    def read_refs(self, refs: Any) -> list[dict[str, Any]]:
        cached = self._batches.get(id(refs))
        if cached is not None and cached[0] is refs:
            return cached[1]
        records = self.reader.read_refs(refs)
        if refs is not None and len(records) == len(refs):
            self._batches[id(refs)] = (refs, records)
        return records


def _recent_rgb_frames(reader: Any, observations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    frames = []
    for observation in observations:
        frames.extend(_rgb_frames_from_observation(reader, observation))
    return frames[-limit:]


def selected_rgb_frames_from_observations(
    reader: Any,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select only the newest RGB frame for this chronological checkpoint."""

    # CURRENT is tied to the latest observation identity.  Falling back to an
    # older frame would turn historical visibility into false current evidence.
    latest_observation = observations[-1] if observations else None
    rgb_frames = _rgb_frames_from_observation(reader, latest_observation)
    if not rgb_frames:
        return []
    latest_frame = rgb_frames[-1]
    item = dict(latest_frame)
    item["selection_role"] = "current_observation"
    item["selection_reason"] = CURRENT_IMAGE_USAGE
    item["input_id"] = _image_input_id(1, "current_observation")
    item["usage"] = CURRENT_IMAGE_USAGE
    return [item]


def selected_frame_summaries(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for frame in frames:
        role = frame.get("selection_role")
        if not isinstance(role, str) or not role:
            role = "latest"
        item = {"role": role}
        for key in ("input_id", "usage"):
            value = frame.get(key)
            if isinstance(value, str) and value:
                item[key] = value
        for source_key, target_key in (
            ("observation_id", "observation_id"),
            ("id", "frame_id"),
            ("selection_reason", "reason"),
        ):
            value = frame.get(source_key)
            if isinstance(value, str) and value:
                item[target_key] = value
        summaries.append(item)
    return summaries


def frame_context_metadata(
    observations: list[dict[str, Any]],
    selected_frames: list[dict[str, Any]],
    recent_frames: list[dict[str, Any]],
    *,
    current_depth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "observation_count": len(observations),
        "selected_frame_metadata": frame_debug_metadata(selected_frames),
    }
    if current_depth is not None:
        metadata["current_depth_metadata"] = frame_debug_metadata([current_depth])[0]
    if not _same_frame_identity_set(selected_frames, recent_frames):
        metadata["recent_rgb_frame_metadata"] = frame_debug_metadata(recent_frames)
    return metadata


def frame_debug_metadata(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for frame in frames:
        item: dict[str, Any] = {}
        role = frame.get("selection_role")
        if isinstance(role, str) and role:
            item["role"] = role
        reason = frame.get("selection_reason")
        if isinstance(reason, str) and reason:
            item["reason"] = reason
        for key in ("input_id", "usage"):
            value = frame.get(key)
            if isinstance(value, str) and value:
                item[key] = value
        if frame.get("id") is not None:
            item["frame_id"] = frame.get("id")
        for source_key, target_key in (
            ("observation_id", "observation_id"),
            ("observation_source", "observation_source"),
            ("path", "path"),
            ("artifact_ref", "artifact_ref"),
            ("width", "width"),
            ("height", "height"),
            ("mock", "mock"),
            ("source", "source"),
        ):
            if source_key in frame:
                item[target_key] = frame.get(source_key)
        metadata.append(item)
    return metadata


def _replace_current_with_depth_overlay_input(
    model_inputs: list[dict[str, Any]],
    depth_overlay_input: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace raw current RGB with its aligned RGB-depth representation.

    The overlay retains the current scene pixels, so attaching both copies asks
    the model to compare duplicate current evidence and consumes an unnecessary
    image slot.
    """

    result: list[dict[str, Any]] = []
    replaced_current = False
    for item in model_inputs:
        if _model_input_role(item) == "current_observation":
            if not replaced_current:
                result.append(depth_overlay_input)
                replaced_current = True
            continue
        result.append(item)
    if replaced_current:
        return result

    for index, item in enumerate(result):
        if item.get("role") == "temporal_evidence_rule":
            result.insert(index, depth_overlay_input)
            return result
    result.append(depth_overlay_input)
    return result


def _model_input_role(item: dict[str, Any]) -> str:
    media_ref = item.get("media_ref")
    if isinstance(media_ref, dict):
        return str(media_ref.get("role") or "")
    return str(item.get("role") or "")


def _image_input_id(index: int, role: str) -> str:
    safe_role = "".join(char if char.isalnum() or char == "_" else "_" for char in role.strip().lower())
    return f"image_{index}_{safe_role or 'rgb'}"


def _frame_identity(frame: dict[str, Any]) -> str:
    for key in ("path", "artifact_path", "artifact_ref", "id", "observation_id"):
        value = frame.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return repr(sorted(frame.items()))


def _same_frame_identity_set(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    left_keys = {_frame_identity(frame) for frame in left}
    right_keys = {_frame_identity(frame) for frame in right}
    return left_keys == right_keys


def _rgb_frame_from_anchor(reader: Any, ref: dict[str, Any] | None) -> dict[str, Any] | None:
    observation = reader.read_ref(ref) if ref else None
    frames = _rgb_frames_from_observation(reader, observation)
    return frames[0] if frames else None


def _rgb_frames_from_observation(reader: Any, observation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(observation, dict):
        return []
    frame_refs = observation.get("frame_refs", [])
    frame_records = reader.read_refs(frame_refs)
    frames: list[dict[str, Any]] = []
    for index, record in enumerate(frame_records):
        if record.get("modality") != "rgb":
            continue
        frames.append(
            {
                "id": record.get("id"),
                "path": record.get("path"),
                "observation_id": record.get("observation_id"),
                "observation_source": observation.get("source"),
                "artifact_ref": _artifact_ref(record, frame_refs[index] if index < len(frame_refs) else None),
                **dict(record.get("metadata", {})),
            }
        )
    return frames


def _depth_frame_for_current_rgb(
    reader: Any,
    observations: list[dict[str, Any]],
    current_rgb: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(current_rgb, dict):
        return None
    current_observation_id = current_rgb.get("observation_id")
    if not isinstance(current_observation_id, str) or not current_observation_id:
        return None
    for observation in reversed(observations):
        for frame in _depth_frames_from_observation(reader, observation):
            if frame.get("observation_id") != current_observation_id:
                continue
            if frame.get("mock") is True:
                return None
            frame["selection_role"] = "current_depth"
            frame["selection_reason"] = CURRENT_RGB_DEPTH_OVERLAY_USAGE
            frame["input_id"] = "image_current_depth"
            frame["usage"] = CURRENT_RGB_DEPTH_OVERLAY_USAGE
            return frame
    return None


def _depth_frames_from_observation(
    reader: Any,
    observation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(observation, dict):
        return []
    frame_refs = observation.get("frame_refs", [])
    frame_records = reader.read_refs(frame_refs)
    frames: list[dict[str, Any]] = []
    for index, record in enumerate(frame_records):
        if record.get("modality") != "depth":
            continue
        frames.append(
            {
                "id": record.get("id"),
                "path": record.get("path"),
                "observation_id": record.get("observation_id"),
                "observation_source": observation.get("source"),
                "artifact_ref": _artifact_ref(
                    record,
                    frame_refs[index] if index < len(frame_refs) else None,
                ),
                **dict(record.get("metadata", {})),
            }
        )
    return frames


def _artifact_ref(record: dict[str, Any], frame_ref: dict[str, Any] | None) -> str | None:
    path = record.get("path")
    if isinstance(path, str) and path:
        return path
    ref = frame_ref.get("ref") if isinstance(frame_ref, dict) else None
    return ref if isinstance(ref, str) and ref else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
