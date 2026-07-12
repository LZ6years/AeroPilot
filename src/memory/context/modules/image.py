"""Compact RGB context module."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput


CURRENT_IMAGE_USAGE = "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions."
PHASE_START_IMAGE_USAGE = "PAST | PHASE START. Comparison only."
TEMPORAL_EVIDENCE_RULE = (
    "Current-state claims must use NOW only. PAST is only for comparing change. "
    "An object visible only in PAST is not currently visible."
)

_CURRENT_IMAGE_BANNER = "NOW | CURRENT OBSERVATION"
_PHASE_START_IMAGE_BANNER = "PAST | PHASE START"
_CURRENT_IMAGE_BANNER_COLOR = (28, 122, 72)
_PHASE_START_IMAGE_BANNER_COLOR = (190, 117, 20)


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
            selected_rgb_frames_from_observations(
                reader,
                all_observations,
                observation_anchors=getattr(memory, "observation_anchors", {}),
            )
            if reader
            else []
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
        visual_evidence["attached_image_count"] = sum(
            1 for item in model_inputs if item.get("type") == "image"
        )
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
            metadata=frame_context_metadata(observations, selected_rgb_frames, rgb_frames),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Selected RGB frame roles, attached image count, and visual evidence availability.",
            "content_fields": {
                "selected_frames": (
                    "Key RGB frames selected for multimodal input, including roles such as "
                    "current_observation and phase_start_reference, plus input_id, usage, and source identifiers."
                ),
                "visual_evidence": "Boolean availability summary such as has_rgb, has_depth, has_mock_frames, latest_observation_source, and interpretation_status.",
            },
            "model_inputs": (
                "Up to two encoded real RGB images for planner multimodal input, ordered as "
                "a distinct phase_start_reference followed by current_observation. "
                "When both are attached, a concise trailing temporal evidence rule is included. "
                "Mock, missing, unreadable, or duplicate images are not attached."
            ),
            "risk_fields": {
                "visual_evidence_missing": "No recent RGB evidence is available.",
                "visual_evidence_is_mock": "A recent frame is mock/synthetic placeholder evidence and should not support confident target approach.",
            },
            "metadata_fields": {
                "selected_frame_metadata": "Full selected-frame audit metadata for debug/reporting.",
                "recent_rgb_frame_metadata": "Recent RGB audit metadata when it differs from selected frames.",
            },
            "artifact_refs": "Image artifact paths or memory refs that can be inspected outside the planner context.",
        }


def _recent_rgb_frames(reader: Any, observations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    frames = []
    for observation in observations:
        frames.extend(_rgb_frames_from_observation(reader, observation))
    return frames[-limit:]


def selected_rgb_frames_from_observations(
    reader: Any,
    observations: list[dict[str, Any]],
    observation_anchors: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Select key RGB frames from observation history without interpreting image content.

    Prefer a distinct current-phase anchor followed by the latest RGB frame so
    provider-ready multimodal input is chronological and current evidence is
    last. If both roles resolve to one frame, keep it only as current evidence.
    """

    rgb_frames = _recent_rgb_frames(reader, observations, len(observations))
    if not rgb_frames:
        return []
    anchors = observation_anchors if isinstance(observation_anchors, dict) else {}
    phase_start = _rgb_frame_from_anchor(reader, anchors.get("phase_start"))
    legacy_task_initial = _rgb_frame_from_anchor(reader, anchors.get("task_initial"))
    phase_start_frame = phase_start or legacy_task_initial or rgb_frames[0]
    latest_frame = rgb_frames[-1]
    if _frame_identity(phase_start_frame) == _frame_identity(latest_frame):
        candidates = [("current_observation", CURRENT_IMAGE_USAGE, latest_frame)]
    else:
        candidates = [
            ("phase_start_reference", PHASE_START_IMAGE_USAGE, phase_start_frame),
            ("current_observation", CURRENT_IMAGE_USAGE, latest_frame),
        ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role, reason, frame in candidates:
        key = _frame_identity(frame)
        if key in seen:
            continue
        seen.add(key)
        item = dict(frame)
        item["selection_role"] = role
        item["selection_reason"] = reason
        item["input_id"] = _image_input_id(len(selected) + 1, role)
        item["usage"] = reason
        selected.append(item)
    return selected[:2]


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
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "observation_count": len(observations),
        "selected_frame_metadata": frame_debug_metadata(selected_frames),
    }
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


def rgb_model_inputs_from_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded_by_role: dict[str, dict[str, Any]] = {}
    for frame in frames[:2]:
        for item in latest_rgb_model_inputs_from_frame(frame):
            role = item.get("role")
            if isinstance(role, str) and role:
                encoded_by_role[role] = item

    current = encoded_by_role.get("current_observation")
    if current is None:
        return []
    phase_start = encoded_by_role.get("phase_start_reference")
    if phase_start is None:
        return [current]
    return [
        phase_start,
        current,
        {
            "type": "text",
            "role": "temporal_evidence_rule",
            "text": TEMPORAL_EVIDENCE_RULE,
        },
    ]


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


def _artifact_ref(record: dict[str, Any], frame_ref: dict[str, Any] | None) -> str | None:
    path = record.get("path")
    if isinstance(path, str) and path:
        return path
    ref = frame_ref.get("ref") if isinstance(frame_ref, dict) else None
    return ref if isinstance(ref, str) and ref else None


def latest_rgb_model_inputs_from_frame(latest_rgb: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Encode one latest real RGB frame for model multimodal input.

    The coupled planner uses this helper so unreadable or mock frames follow the
    same conservative no-image behavior across action and phase decisions.
    """

    if not latest_rgb or latest_rgb.get("mock") is True:
        return []
    path = _frame_path(latest_rgb)
    if path is None:
        return []
    role = str(latest_rgb.get("selection_role") or "latest")
    input_id = str(latest_rgb.get("input_id") or _image_input_id(1, role))
    usage = str(latest_rgb.get("usage") or latest_rgb.get("selection_reason") or "")
    data_url = _annotated_image_data_url(path, role)
    if data_url is None:
        return []
    return [
        {
            "type": "image",
            "input_id": input_id,
            "role": role,
            "usage": usage,
            "mime_type": "image/png",
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


def _annotated_image_data_url(path: Path, role: str) -> str | None:
    """Return a model-only PNG with an external temporal-role banner."""

    try:
        with Image.open(path) as source:
            scene = source.convert("RGB")
            banner_text, banner_color = _image_banner(role)
            banner_height = _banner_height(scene.width, scene.height)
            annotated = Image.new("RGB", (scene.width, scene.height + banner_height), banner_color)
            annotated.paste(scene, (0, banner_height))
            _draw_banner_text(annotated, banner_text, banner_height)
            output = BytesIO()
            annotated.save(output, format="PNG")
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_banner(role: str) -> tuple[str, tuple[int, int, int]]:
    if role == "phase_start_reference":
        return _PHASE_START_IMAGE_BANNER, _PHASE_START_IMAGE_BANNER_COLOR
    return _CURRENT_IMAGE_BANNER, _CURRENT_IMAGE_BANNER_COLOR


def _banner_height(width: int, height: int) -> int:
    return max(28, min(56, round(min(width, height) * 0.14)))


def _draw_banner_text(image: Image.Image, text: str, banner_height: int) -> None:
    draw = ImageDraw.Draw(image)
    max_font_size = max(10, min(24, banner_height - 12))
    font = _fitted_banner_font(draw, text, image.width - 20, max_font_size)
    text_box = draw.textbbox((0, 0), text, font=font)
    text_height = text_box[3] - text_box[1]
    left = 10
    top = max(0, (banner_height - text_height) // 2 - text_box[1])
    draw.text((left, top), text, fill=(255, 255, 255), font=font)


def _fitted_banner_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, max_size: int) -> ImageFont.ImageFont:
    for font_size in range(max_size, 9, -1):
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:  # Pillow < 10 compatibility for downstream users.
            return ImageFont.load_default()
        text_box = draw.textbbox((0, 0), text, font=font)
        if text_box[2] - text_box[0] <= max_width:
            return font
    return ImageFont.load_default()
