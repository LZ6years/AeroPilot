"""Compact RGB and aligned metric-depth context module."""

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
CURRENT_RGB_DEPTH_OVERLAY_USAGE = (
    "NOW | RGB + METRIC DEPTH OVERLAY. Same scene and pixels as NOW RGB; "
    "false colors encode distance in meters from the UAV camera to visible "
    "scene points, so colors on a visible target give its approximate range."
)
TEMPORAL_EVIDENCE_RULE = (
    "TEMPORAL ROLE RULE: PAST is historical comparison evidence only. "
    "Current visibility, image position, approximate range, and the next action "
    "must be judged from NOW. An object visible only in PAST is not currently visible."
)
CURRENT_STATE_DECISION_REMINDER = (
    "CURRENT-STATE CHECK: The immediately preceding NOW image is the current "
    "observation. Base current visibility, bearing, approximate range, and the "
    "next action on that NOW image."
)

_CURRENT_IMAGE_BANNER = "NOW | CURRENT OBSERVATION"
_PHASE_START_IMAGE_BANNER = "PAST | PHASE START"
_CURRENT_RGB_DEPTH_OVERLAY_BANNER = "NOW | RGB + METRIC DEPTH"
_CURRENT_IMAGE_BANNER_COLOR = (28, 122, 72)
_PHASE_START_IMAGE_BANNER_COLOR = (190, 117, 20)
_CURRENT_RGB_DEPTH_OVERLAY_BANNER_COLOR = (30, 78, 138)
_DEFAULT_DEPTH_DISPLAY_MAX_M = 10.0
_DEPTH_OVERLAY_ALPHA = 0.42
_DEPTH_COLOR_STOPS = (
    (180, 0, 0),
    (255, 110, 0),
    (255, 230, 0),
    (50, 190, 80),
    (0, 170, 220),
    (50, 60, 180),
)


class DefaultImageContextModule(ContextModule):
    name = "image"
    priority = 80
    max_items = 3

    def __init__(
        self,
        *,
        include_phase_start_reference: bool = True,
        include_current_depth: bool = False,
        depth_display_max_m: float = _DEFAULT_DEPTH_DISPLAY_MAX_M,
    ) -> None:
        # The default preserves the shared two-frame temporal contract.  A
        # benchmark composition may disable the reference only when another
        # bounded history representation is successfully attached.
        self.include_phase_start_reference = bool(
            include_phase_start_reference
        )
        self.include_current_depth = bool(include_current_depth)
        self.depth_display_max_m = float(depth_display_max_m)
        if self.depth_display_max_m <= 0.0:
            raise ValueError("depth_display_max_m must be positive")

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
        if not bool(
            getattr(self, "include_phase_start_reference", True)
        ):
            selected_rgb_frames = _current_only_rgb_selection(
                selected_rgb_frames
            )
        latest_rgb = rgb_frames[-1] if rgb_frames else None
        current_rgb = next(
            (
                frame
                for frame in selected_rgb_frames
                if frame.get("selection_role") == "current_observation"
            ),
            None,
        )
        current_depth = (
            _depth_frame_for_current_rgb(reader, all_observations, current_rgb)
            if reader and bool(getattr(self, "include_current_depth", False))
            else None
        )
        artifact_refs = [item["artifact_ref"] for item in rgb_frames if item.get("artifact_ref")]
        if current_depth and current_depth.get("artifact_ref"):
            artifact_refs.append(current_depth["artifact_ref"])
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
        model_inputs = rgb_model_inputs_from_frames(selected_rgb_frames)
        depth_overlay_input = (
            current_rgb_depth_overlay_model_input(
                current_depth,
                current_rgb=current_rgb,
                depth_display_max_m=float(
                    getattr(self, "depth_display_max_m", _DEFAULT_DEPTH_DISPLAY_MAX_M)
                ),
            )
            if current_depth
            else None
        )
        if depth_overlay_input is not None:
            model_inputs = _replace_current_with_depth_overlay_input(
                model_inputs,
                depth_overlay_input,
            )
        selected_frames = selected_frame_summaries(selected_rgb_frames)
        if current_depth is not None:
            selected_frames.extend(selected_frame_summaries([current_depth]))
        visual_evidence = {
            "has_rgb": bool(rgb_frames),
            "has_depth": depth_overlay_input is not None,
            "has_mock_frames": any(
                isinstance(frame, dict) and frame.get("mock") is True
                for frame in rgb_frames
            ),
            "latest_observation_source": latest_rgb.get("observation_source") if latest_rgb else None,
        }
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
            metadata=frame_context_metadata(
                observations,
                selected_rgb_frames,
                rgb_frames,
                current_depth=current_depth,
            ),
        )

    def describe_output(self) -> dict[str, Any]:
        semantic_evidence = [
            "NOW RGB: current visible appearance and image-space relations.",
        ]
        include_phase_start_reference = bool(
            getattr(self, "include_phase_start_reference", True)
        )
        if include_phase_start_reference:
            semantic_evidence.append(
                "PAST RGB when distinct: phase-start comparison only; not current visibility."
            )
        if bool(getattr(self, "include_current_depth", False)):
            semantic_evidence.append(
                "NOW metric depth overlay: approximate UAV-camera-to-scene distance at visible pixels; target pixels provide approximate UAV-to-target range."
            )
        return {
            "summary": "Selected RGB frame roles, attached image count, and visual evidence availability.",
            "semantic_evidence": semantic_evidence,
            "content_fields": {
                "selected_frames": (
                    "Key RGB frames selected for multimodal input, including roles such as "
                    "current_observation and phase_start_reference, plus input_id, usage, and source identifiers."
                ),
                "visual_evidence": "Boolean availability summary such as has_rgb, has_depth, has_mock_frames, latest_observation_source, and interpretation_status.",
            },
            "model_inputs": (
                "One current image, using the pixel-aligned RGB-depth overlay when "
                "configured and valid, otherwise raw RGB"
                + (
                    ", plus a distinct phase-start RGB reference"
                    if include_phase_start_reference
                    else ", without a phase-start reference"
                )
                + ". A temporal rule precedes two distinct roles, and a current-state "
                "reminder immediately follows NOW. Single-image inputs keep only the "
                "NOW role and reminder. Mock, missing, unreadable, or duplicate images "
                "are not attached."
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


def _current_only_rgb_selection(
    selected_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only NOW RGB and normalize its identity to the first image slot."""

    current = next(
        (
            dict(frame)
            for frame in selected_frames
            if frame.get("selection_role") == "current_observation"
        ),
        None,
    )
    if current is None:
        return []
    current["selection_reason"] = CURRENT_IMAGE_USAGE
    current["input_id"] = _image_input_id(1, "current_observation")
    current["usage"] = CURRENT_IMAGE_USAGE
    return [current]


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


def rgb_model_inputs_from_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build provider inputs with temporal instructions adjacent to images.

    Temporal meaning belongs to this module, so the generic provider assembler
    must not infer whether an image is PAST or NOW. For a two-image window the
    rule precedes both images and the current-state reminder immediately follows
    NOW. A first-window/current-only input has no synthetic PAST rule.
    """

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
        return [
            current,
            {
                "type": "text",
                "role": "current_state_decision_reminder",
                "text": CURRENT_STATE_DECISION_REMINDER,
            },
        ]
    return [
        {
            "type": "text",
            "role": "temporal_evidence_rule",
            "text": TEMPORAL_EVIDENCE_RULE,
        },
        phase_start,
        current,
        {
            "type": "text",
            "role": "current_state_decision_reminder",
            "text": CURRENT_STATE_DECISION_REMINDER,
        },
    ]


def _replace_current_with_depth_overlay_input(
    model_inputs: list[dict[str, Any]],
    depth_overlay_input: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace raw current RGB with its aligned RGB-depth representation.

    The overlay retains the current scene pixels, so attaching both copies asks
    the model to compare duplicate current evidence and consumes an unnecessary
    image slot. The phase-start RGB remains a separate historical anchor.
    """

    result: list[dict[str, Any]] = []
    replaced_current = False
    for item in model_inputs:
        if item.get("role") == "current_observation":
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


def current_rgb_depth_overlay_model_input(
    current_depth: dict[str, Any] | None,
    *,
    current_rgb: dict[str, Any] | None,
    depth_display_max_m: float,
) -> dict[str, Any] | None:
    if (
        not current_depth
        or current_depth.get("mock") is True
        or not current_rgb
        or current_rgb.get("mock") is True
    ):
        return None
    depth_path = _frame_path(current_depth)
    rgb_path = _frame_path(current_rgb)
    if depth_path is None or rgb_path is None:
        return None
    data_url = _rgb_depth_overlay_data_url(
        rgb_path,
        depth_path,
        metadata=current_depth,
        depth_display_max_m=depth_display_max_m,
    )
    if data_url is None:
        return None
    return {
        "type": "image",
        "input_id": "image_current_rgb_depth_overlay",
        "role": "current_rgb_depth_overlay",
        "usage": CURRENT_RGB_DEPTH_OVERLAY_USAGE,
        "mime_type": "image/png",
        "encoding": "data_url",
        "data": data_url,
    }


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


def _rgb_depth_overlay_data_url(
    rgb_path: Path,
    depth_path: Path,
    *,
    metadata: dict[str, Any],
    depth_display_max_m: float,
) -> str | None:
    """Blend pixel-aligned RGB and metric depth into one model-only PNG.

    Keeping the RGB structure under a translucent, fixed-scale depth color map
    removes the need for the model to establish object correspondence across
    two separate images. A size mismatch is rejected instead of resized because
    an apparently aligned but geometrically false overlay would be misleading.
    """

    try:
        import numpy as np

        with Image.open(rgb_path) as source:
            rgb = source.convert("RGB")
        depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
        depth = np.squeeze(depth)
        if depth.ndim == 1:
            width = _positive_int(metadata.get("width"))
            height = _positive_int(metadata.get("height"))
            if width is None or height is None or depth.size != width * height:
                return None
            depth = depth.reshape((height, width))
        if depth.ndim != 2 or depth.size == 0:
            return None
        if rgb.size != (int(depth.shape[1]), int(depth.shape[0])):
            return None
        valid = np.isfinite(depth) & (depth > 0.0)
        if not bool(valid.any()):
            return None
        colored = _colorize_metric_depth(depth, valid, depth_display_max_m)
        rgb_array = np.asarray(rgb, dtype=np.float32)
        overlay = (
            (1.0 - _DEPTH_OVERLAY_ALPHA) * rgb_array
            + _DEPTH_OVERLAY_ALPHA * colored.astype(np.float32)
        )
        overlay[~valid] = rgb_array[~valid]
        scene = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
        annotated = _annotated_rgb_depth_overlay_image(scene, depth_display_max_m)
        output = BytesIO()
        annotated.save(output, format="PNG")
    except (OSError, EOFError, UnidentifiedImageError, ValueError, TypeError):
        return None
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _colorize_metric_depth(depth: Any, valid: Any, depth_display_max_m: float) -> Any:
    import numpy as np

    normalized = np.clip(depth / float(depth_display_max_m), 0.0, 1.0)
    stop_positions = np.linspace(0.0, 1.0, len(_DEPTH_COLOR_STOPS), dtype=np.float32)
    colors = np.asarray(_DEPTH_COLOR_STOPS, dtype=np.float32)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = np.interp(
            normalized,
            stop_positions,
            colors[:, channel],
        ).astype(np.uint8)
    output[~valid] = 0
    return output


def _annotated_rgb_depth_overlay_image(
    scene: Image.Image,
    depth_display_max_m: float,
) -> Image.Image:
    banner_height = _banner_height(scene.width, scene.height)
    legend_height = max(42, min(68, round(scene.height * 0.18)))
    annotated = Image.new(
        "RGB",
        (scene.width, scene.height + banner_height + legend_height),
        (20, 24, 30),
    )
    annotated.paste(scene, (0, banner_height))
    banner = Image.new(
        "RGB",
        (scene.width, banner_height),
        _CURRENT_RGB_DEPTH_OVERLAY_BANNER_COLOR,
    )
    annotated.paste(banner, (0, 0))
    _draw_banner_text(annotated, _CURRENT_RGB_DEPTH_OVERLAY_BANNER, banner_height)
    _draw_depth_legend(
        annotated,
        top=banner_height + scene.height,
        height=legend_height,
        depth_display_max_m=depth_display_max_m,
    )
    return annotated


def _draw_depth_legend(
    image: Image.Image,
    *,
    top: int,
    height: int,
    depth_display_max_m: float,
) -> None:
    import numpy as np

    draw = ImageDraw.Draw(image)
    left = max(1, min(round(image.width * 0.04), max(1, image.width // 4)))
    right = max(left + 1, image.width - left)
    bar_top = top + 7
    bar_height = max(9, min(16, height // 3))
    gradient_depth = np.linspace(
        0.0,
        depth_display_max_m,
        max(1, right - left),
        dtype=np.float32,
    )[None, :]
    gradient = _colorize_metric_depth(
        gradient_depth,
        np.ones_like(gradient_depth, dtype=bool),
        depth_display_max_m,
    )
    gradient_image = Image.fromarray(gradient, mode="RGB").resize(
        (right - left, bar_height),
        resample=Image.Resampling.NEAREST,
    )
    image.paste(gradient_image, (left, bar_top))
    draw.rectangle(
        (left, bar_top, right - 1, bar_top + bar_height - 1),
        outline=(235, 235, 235),
        width=1,
    )
    label_top = bar_top + bar_height + 3
    font = _default_font(max(10, min(16, height // 4)))
    labels = (
        (left, "0 m"),
        ((left + right) // 2, f"{depth_display_max_m / 2:g} m"),
        (right, f"{depth_display_max_m:g}+ m"),
    )
    for x, label in labels:
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        if x == left:
            text_x = x
        elif x == right:
            text_x = x - text_width
        else:
            text_x = x - text_width // 2
        draw.text((text_x, label_top), label, fill=(255, 255, 255), font=font)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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


def _default_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
