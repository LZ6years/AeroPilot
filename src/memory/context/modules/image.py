"""Compact RGB context module."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput


class DefaultImageContextModule(ContextModule):
    name = "image"
    priority = 80
    max_items = 3

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        memory = sources.flight_memory
        reader = memory.memory_reader
        all_observations = reader.read_refs(memory.recent_observation_refs) if reader else []
        observations = all_observations[-self.max_items :]
        rgb_frames = _recent_rgb_frames(reader, observations, self.max_items) if reader else []
        selected_rgb_frames = (
            selected_rgb_frames_from_observations(reader, all_observations) if reader else []
        )
        latest_rgb = rgb_frames[-1] if rgb_frames else None
        artifact_refs = [item["artifact_ref"] for item in rgb_frames if item.get("artifact_ref")]
        warnings = []
        risks = []
        if not rgb_frames:
            warnings.append("no_recent_image_frames")
            risks.append({"type": "visual_evidence_missing", "source": self.__class__.__name__})
        for frame in rgb_frames:
            if isinstance(frame, dict) and frame.get("mock") is True:
                warnings.append("rgb_frame_is_mock")
                risks.append(
                    {
                        "type": "visual_evidence_is_mock",
                        "value": "rgb",
                        "source": self.__class__.__name__,
                    }
                )
        visual_evidence = {
            "has_rgb": bool(rgb_frames),
            "has_depth": False,
            "has_mock_frames": any(
                isinstance(frame, dict) and frame.get("mock") is True
                for frame in rgb_frames
            ),
            "latest_observation_source": latest_rgb.get("observation_source") if latest_rgb else None,
        }
        model_inputs = rgb_model_inputs_from_frames(selected_rgb_frames)
        selected_frames = selected_frame_summaries(selected_rgb_frames)
        visual_evidence["interpretation_status"] = (
            "image_attached_not_interpreted" if model_inputs else "metadata_only_not_interpreted"
        )
        visual_evidence["attached_image_count"] = len(model_inputs)
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "selected_frames": selected_frames,
                "visual_evidence": visual_evidence,
            },
            model_inputs=model_inputs,
            risks=risks,
            artifact_refs=artifact_refs,
            confidence="medium" if latest_rgb else "low",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=300, max_items=self.max_items),
            warnings=warnings,
            metadata={
                "observation_count": len(observations),
                "selected_frame_metadata": frame_debug_metadata(selected_rgb_frames),
                "recent_rgb_frame_metadata": frame_debug_metadata(rgb_frames),
            },
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Selected RGB frame roles, attached image count, and visual evidence availability.",
            "content_fields": {
                "selected_frames": "Key RGB frames selected for multimodal input, exposed only as role/source pairs.",
                "visual_evidence": "Boolean availability summary such as has_rgb, has_depth, has_mock_frames, latest_observation_source, and interpretation_status.",
            },
            "model_inputs": (
                "Up to four encoded real RGB images for planner multimodal input. "
                "Mock, missing, or unreadable images are not attached."
            ),
            "risk_fields": {
                "visual_evidence_missing": "No recent RGB evidence is available.",
                "visual_evidence_is_mock": "A recent frame is mock/synthetic placeholder evidence and should not support confident target approach.",
            },
            "metadata_fields": {
                "selected_frame_metadata": "Full selected-frame audit metadata for debug/reporting.",
                "recent_rgb_frame_metadata": "Full recent RGB audit metadata for debug/reporting.",
            },
            "artifact_refs": "Image artifact paths or memory refs that can be inspected outside the planner context.",
        }


def _recent_rgb_frames(reader: Any, observations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    frames = []
    for observation in observations:
        frame_refs = observation.get("frame_refs", []) if isinstance(observation, dict) else []
        frame_records = reader.read_refs(frame_refs)
        for index, record in enumerate(frame_records):
            if record.get("modality") != "rgb":
                continue
            frame = {
                "id": record.get("id"),
                "path": record.get("path"),
                "observation_id": record.get("observation_id"),
                "observation_source": observation.get("source") if isinstance(observation, dict) else None,
                "artifact_ref": _artifact_ref(record, frame_refs[index] if index < len(frame_refs) else None),
                **dict(record.get("metadata", {})),
            }
            frames.append(frame)
    return frames[-limit:]


def selected_rgb_frames_from_observations(reader: Any, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select key RGB frames from observation history without interpreting image content.

    The fixed v1 policy is deliberately simple: current/latest, initial, previous
    phase-window, and four phase-windows ago. Duplicate frames are removed by
    frame id/path so short runs naturally attach fewer than four images.
    """

    rgb_frames = _recent_rgb_frames(reader, observations, len(observations))
    if not rgb_frames:
        return []
    candidates = [
        ("latest_observation", rgb_frames[-1]),
        ("initial_observation", rgb_frames[0]),
    ]
    if len(rgb_frames) >= 2:
        candidates.append(("previous_observation", rgb_frames[-2]))
    if len(rgb_frames) >= 5:
        candidates.append(("four_windows_ago_observation", rgb_frames[-5]))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role, frame in candidates:
        key = _frame_identity(frame)
        if key in seen:
            continue
        seen.add(key)
        item = dict(frame)
        item["selection_role"] = role
        selected.append(item)
    return selected[:4]


def selected_frame_summaries(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for frame in frames:
        role = frame.get("selection_role")
        if not isinstance(role, str) or not role:
            role = "latest_observation"
        item = {"role": role}
        source = frame.get("observation_source")
        if isinstance(source, str) and source:
            item["source"] = source
        summaries.append(item)
    return summaries


def frame_debug_metadata(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for frame in frames:
        item: dict[str, Any] = {}
        role = frame.get("selection_role")
        if isinstance(role, str) and role:
            item["role"] = role
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


def rgb_model_inputs_from_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for frame in frames[:4]:
        inputs.extend(latest_rgb_model_inputs_from_frame(frame))
    return inputs


def _frame_identity(frame: dict[str, Any]) -> str:
    for key in ("path", "artifact_path", "artifact_ref", "id", "observation_id"):
        value = frame.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return repr(sorted(frame.items()))


def _artifact_ref(record: dict[str, Any], frame_ref: dict[str, Any] | None) -> str | None:
    path = record.get("path")
    if isinstance(path, str) and path:
        return path
    ref = frame_ref.get("ref") if isinstance(frame_ref, dict) else None
    return ref if isinstance(ref, str) and ref else None


def latest_rgb_model_inputs_from_frame(latest_rgb: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Encode one latest real RGB frame for model multimodal input.

    The action planner and post-action phase judge both use this helper so
    unreadable or mock frames follow the same conservative no-image behavior.
    """

    if not latest_rgb or latest_rgb.get("mock") is True:
        return []
    path = _frame_path(latest_rgb)
    if path is None:
        return []
    data_url = _image_data_url(path)
    if data_url is None:
        return []
    mime_type = data_url.split(";", 1)[0].removeprefix("data:")
    role = str(latest_rgb.get("selection_role") or "latest_observation")
    return [
        {
            "type": "image",
            "role": role,
            "mime_type": mime_type,
            "encoding": "data_url",
            "data": data_url,
        }
    ]


def _frame_path(frame: dict[str, Any]) -> Path | None:
    raw_path = frame.get("path") or frame.get("artifact_path") or frame.get("artifact_ref")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return None


def _image_data_url(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    if not mime_type.startswith("image/"):
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
