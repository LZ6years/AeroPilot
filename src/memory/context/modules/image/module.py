"""Checkpoint image selection and fragment assembly."""

from __future__ import annotations

import math
from typing import Any

from src.memory.context.contracts import (
    ContextFragment,
    ContextModuleResult,
    DerivedMediaWriter,
    FragmentAudit,
    HistoryImageCandidate,
    MemorySourceBundle,
    TaskContextInput,
)
from src.memory.context.modules.base import ContextModule
from src.memory.context.modules.image.frames import (
    CURRENT_IMAGE_USAGE,
    _BuildFrameReader,
    _depth_frame_for_current_rgb,
    _recent_rgb_frames,
    _replace_current_with_depth_overlay_input,
    _rgb_media_inputs_from_frames,
    frame_context_metadata,
    selected_frame_summaries,
    selected_rgb_frames_from_observations,
)
from src.memory.context.modules.image.overlay import (
    CURRENT_RGB_DEPTH_OVERLAY_USAGE,
    DEFAULT_DEPTH_DISPLAY_MAX_M,
    current_rgb_depth_overlay_media_input,
)

_DEFAULT_DEPTH_DISPLAY_MAX_M = DEFAULT_DEPTH_DISPLAY_MAX_M
_current_rgb_depth_overlay_media_input = current_rgb_depth_overlay_media_input

class DefaultImageContextModule(ContextModule):
    name = "image"
    priority = 80
    max_items = 3

    def validate_history_policy(self, config: dict[str, Any]) -> None:
        """Validate module-owned selection options, never provider settings."""
        mode = config.get("mode", "all")
        if mode == "all":
            return super().validate_history_policy(config)
        fields = {"mode", "max_observation_images", "keep_recent_observations"}
        if mode != "recency_log" or set(config) != fields:
            raise ValueError("recency_log requires mode and both observation limits")
        budget = config["max_observation_images"]
        recent = config["keep_recent_observations"]
        if type(budget) is not int or type(recent) is not int:
            raise ValueError("history image limits must be integers")
        if recent < 1 or budget < recent + 2:
            raise ValueError("history image budget must be >= recent + 2, recent >= 1")

    def select_history_images(
        self, candidates: tuple[HistoryImageCandidate, ...], config: dict[str, Any]
    ) -> tuple[str, ...]:
        """Retain recent consecutive frames and log-spaced older observations.

        Selection is recomputed from identities only. Clamp each rounded age
        to leave distinct slots for the remainder, including the first frame.
        This is an API-layer slow/fast approximation, not visual token pooling.
        """
        self.validate_history_policy(config)
        total = len(candidates)
        if config.get("mode", "all") == "all" or total <= config["max_observation_images"]:
            return tuple(item.observation_id for item in candidates)
        budget, recent = config["max_observation_images"], config["keep_recent_observations"]
        slots = budget - recent
        ages = list(range(recent))
        for j in range(slots):
            x = math.log(recent + 1) + j / (slots - 1) * (math.log(total) - math.log(recent + 1))
            rounded = math.floor(math.exp(x) - 1 + 0.5)
            ages.append(min(max(rounded, ages[-1] + 1), total - slots + j))
        indices = sorted(total - age - 1 for age in ages)
        return tuple(candidates[index].observation_id for index in indices)

    def __init__(
        self,
        *,
        include_current_depth: bool = False,
        depth_display_max_m: float = _DEFAULT_DEPTH_DISPLAY_MAX_M,
        media_writer: DerivedMediaWriter | None = None,
    ) -> None:
        if type(include_current_depth) is not bool:
            raise ValueError("include_current_depth must be boolean")
        self.include_current_depth = include_current_depth
        self.depth_display_max_m = float(depth_display_max_m)
        self.media_writer = media_writer
        if not math.isfinite(self.depth_display_max_m) or self.depth_display_max_m <= 0.0:
            raise ValueError("depth_display_max_m must be positive")

    def configure(self, options: dict[str, Any]) -> None:
        """Image processing options do not affect sensor acquisition or sampling."""
        if set(options) - {"include_current_depth", "depth_display_max_m"}:
            raise ValueError("Unknown image context option")
        include_depth = options.get("include_current_depth", getattr(self, "include_current_depth", False))
        maximum = options.get("depth_display_max_m", getattr(self, "depth_display_max_m", _DEFAULT_DEPTH_DISPLAY_MAX_M))
        if type(include_depth) is not bool:
            raise ValueError("include_current_depth must be boolean")
        if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("depth_display_max_m must be a positive finite number")
        self.include_current_depth = include_depth
        self.depth_display_max_m = float(maximum)

    def bind_media_writer(self, writer: DerivedMediaWriter) -> None:
        """Inject the Runtime-owned store used only for derived overlays."""

        self.media_writer = writer

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextModuleResult:
        memory = sources.flight_memory
        reader = memory.memory_reader
        all_observations = reader.read_refs(memory.recent_observation_refs) if reader else []
        observations = all_observations[-self.max_items :]
        # Share only this build's successful frame-record batches. Observation
        # records are already independent; the wrapper never escapes this call.
        reader = _BuildFrameReader(reader) if reader else None
        rgb_frames = _recent_rgb_frames(reader, observations, self.max_items) if reader else []
        selected_rgb_frames = (
            selected_rgb_frames_from_observations(reader, all_observations)
            if reader
            else []
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
        artifact_refs = list(
            dict.fromkeys(
                item["artifact_ref"]
                for item in [*rgb_frames, *selected_rgb_frames]
                if item.get("artifact_ref")
            )
        )
        warnings = []
        if not rgb_frames:
            warnings.append("no_recent_image_frames")
        for frame in rgb_frames:
            if isinstance(frame, dict) and frame.get("mock") is True:
                warnings.append("rgb_frame_is_mock")
        try:
            model_inputs = _rgb_media_inputs_from_frames(
                selected_rgb_frames,
                run_dir=getattr(reader, "run_dir", None),
            )
        except ValueError:
            # A bad producer path must not abort a flight or make an older frame
            # look current.  The fragment stays explicitly unavailable and the
            # human audit records the contract violation.
            model_inputs = []
            warnings.append("current_rgb_media_ref_invalid")
        depth_overlay_input = (
            _current_rgb_depth_overlay_media_input(
                current_depth,
                current_rgb=current_rgb,
                depth_display_max_m=float(
                    getattr(self, "depth_display_max_m", _DEFAULT_DEPTH_DISPLAY_MAX_M)
                ),
                media_writer=self.media_writer,
            )
            if current_depth
            else None
        )
        if depth_overlay_input is not None:
            if current_depth and current_depth.get("artifact_ref"):
                artifact_refs.append(current_depth["artifact_ref"])
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
        return ContextModuleResult(
            ContextFragment(
                name=self.name,
                model_content=(
                    {} if model_inputs else {"visual_evidence": "current_rgb_unavailable"}
                ),
                model_inputs=tuple(model_inputs),
            ),
            FragmentAudit(
                name=self.name,
                source=self.__class__.__name__,
                confidence="medium" if latest_rgb else "low",
                warnings=tuple(warnings),
                artifact_refs=tuple(dict.fromkeys(artifact_refs)),
                metadata={
                    "selected_frames": selected_frames,
                    "visual_evidence": visual_evidence,
                    **frame_context_metadata(
                        observations,
                        selected_rgb_frames,
                        rgb_frames,
                        current_depth=current_depth,
                    ),
                },
            ),
        )

    def describe_output(self) -> dict[str, Any]:
        semantic_evidence = [
            "One chronological RGB observation; transcript projection labels it HISTORY or CURRENT.",
        ]
        if bool(getattr(self, "include_current_depth", False)):
            semantic_evidence.append(
                "Metric-depth fusion: qualitative range colors only; explicit meter claims require a grounded-candidate distance result."
            )
        return {
            "summary": "Selected RGB frame roles, attached image count, and visual evidence availability.",
            "semantic_evidence": semantic_evidence,
            "model_fields": {
                "visual_evidence": "Present only when the current RGB is unavailable.",
            },
            "model_inputs": (
                "Exactly one image for this checkpoint, using the pixel-aligned RGB-depth "
                "overlay plus a bottom meter color bar when configured and valid, otherwise "
                "raw RGB. Temporal roles are text labels added by transcript projection."
                if getattr(self, "include_current_depth", False) else
                "Exactly one original RGB image for this checkpoint. No depth colors or color bar. "
                "Temporal roles are text labels added by transcript projection."
            ),
            "audit_fields": {
                "selected_frame_metadata": "Full selected-frame audit metadata for debug/reporting.",
                "recent_rgb_frame_metadata": "Recent RGB audit metadata when it differs from selected frames.",
            },
            "artifact_refs": "Image artifact paths or memory refs that can be inspected outside the planner context.",
        }
