"""Context module implementation for phase-aware image captioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory.context.modules.image import DefaultImageContextModule
from src.memory.context.schemas import ContextFragment, MemorySourceBundle, TaskContextInput
from src.runtime.configs.model_config import ModelConfig

from .client import (
    CaptionClient,
    build_visual_caption_payload,
    call_visual_caption_model,
    caption_image_metadata,
    caption_model_config,
)
from .constants import SCHEMA_VERSION, STRATEGY
from .description import normalize_visual_description, visual_description_text


class PluginContextModule(DefaultImageContextModule):
    """Default image metadata plus phase-aware visual captioning."""

    def __init__(
        self,
        *,
        model_config: ModelConfig | None = None,
        caption_client: CaptionClient | None = None,
        env_path: str | Path = ".env",
    ) -> None:
        self._model_config = model_config
        self._caption_client = caption_client or call_visual_caption_model
        self._env_path = Path(env_path)

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextFragment:
        fragment = super().build(sources, stage_context)
        content = dict(fragment.content)
        metadata = dict(fragment.metadata)
        warnings = list(fragment.warnings)
        latest_rgb = _latest_rgb(metadata)
        path = _usable_rgb_path(latest_rgb)
        model_inputs: list[dict[str, Any]] = []
        visual_description: dict[str, Any] | None = None
        processing_status = "metadata_only_not_interpreted"
        caption_metrics: dict[str, Any] = {}

        if path is not None:
            config = caption_model_config(self._model_config, self._env_path)
            if not config.api_key:
                warnings.append("visual_caption_skipped:no_model_api_key")
                processing_status = "metadata_only_caption_skipped"
            elif not config.model_name:
                warnings.append("visual_caption_skipped:no_model_name")
                processing_status = "metadata_only_caption_skipped"
            else:
                try:
                    payload = build_visual_caption_payload(config, path, stage_context, latest_rgb)
                    caption_result = self._caption_client(config, payload)
                    visual_description = normalize_visual_description(
                        caption_result.get("parsed", {}),
                        path,
                        stage_context,
                    )["visual_description"]
                    model_inputs = [_visual_caption_model_input(visual_description, path, caption_result)]
                    content["visual_description"] = visual_description
                    caption_metrics = _caption_metrics(caption_result)
                    processing_status = "captioned"
                except Exception as exc:  # pragma: no cover - exercised through injected failing clients.
                    warnings.append(_caption_failure_warning(exc))
                    processing_status = "metadata_only_caption_failed"

        visual_evidence = dict(content.get("visual_evidence", {}))
        visual_evidence["interpretation_status"] = processing_status
        content["visual_evidence"] = visual_evidence
        content["visual_processing"] = _visual_processing(
            processing_status,
            latest_rgb,
            model_inputs,
            caption_metrics,
        )

        return ContextFragment(
            name=fragment.name,
            enabled=fragment.enabled,
            source=self.__class__.__name__,
            content=content,
            model_inputs=model_inputs,
            facts=list(fragment.facts),
            inferences=list(fragment.inferences),
            goals=list(fragment.goals),
            risks=list(fragment.risks),
            confidence="medium" if visual_description else fragment.confidence,
            artifact_refs=list(fragment.artifact_refs),
            budget=fragment.budget,
            warnings=warnings,
            metadata=metadata,
        )

    def describe_output(self) -> dict[str, Any]:
        description = dict(super().describe_output())
        description["semantic_evidence"] = [
            "Current image-derived text: visual facts from the latest RGB when processing succeeds; not raw image evidence."
        ]
        content_fields = dict(description.get("content_fields", {}))
        content_fields["visual_processing"] = (
            "Custom image processing status. When this module is configured, it may call "
            "a multimodal model to convert the latest real RGB frame into phase-aware text."
        )
        content_fields["visual_description"] = (
            "Phase-aware visual facts derived from the latest RGB frame. It must not contain "
            "navigation actions or observation recommendations."
        )
        description["content_fields"] = content_fields
        description["model_inputs"] = (
            "Planner-facing text inputs prepared by this module, such as visual_caption JSON. "
            "Image data is not forwarded to the planner by this module."
        )
        return description


def _latest_rgb(metadata: dict[str, Any]) -> dict[str, Any] | None:
    selected = metadata.get("selected_frame_metadata", [])
    if isinstance(selected, list):
        for frame in selected:
            if isinstance(frame, dict) and frame.get("role") in {"current_observation", "latest", "latest_observation"}:
                return _restore_frame_metadata(frame)
        if len(selected) == 1 and isinstance(selected[0], dict) and selected[0].get("role") in {"phase_start_reference", "phase_start"}:
            return _restore_frame_metadata(selected[0])
    return None


def _restore_frame_metadata(frame: dict[str, Any]) -> dict[str, Any]:
    restored = dict(frame)
    if "frame_id" in restored:
        restored["id"] = restored.pop("frame_id")
    if "role" in restored:
        restored["selection_role"] = restored["role"]
    return restored


def _usable_rgb_path(latest_rgb: dict[str, Any] | None) -> Path | None:
    if not latest_rgb or latest_rgb.get("mock") is True:
        return None
    return _frame_path(latest_rgb)


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


def _visual_caption_model_input(
    visual_description: dict[str, Any],
    image_path: Path,
    caption_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "text",
        "source": "image",
        "role": "visual_caption",
        "text": visual_description_text(visual_description),
        "metadata": {
            "source_path": str(image_path),
            "schema_version": visual_description.get("schema_version", SCHEMA_VERSION),
            "strategy": visual_description.get("strategy", STRATEGY),
            "duration_sec": caption_result.get("duration_sec"),
            "usage": dict(caption_result.get("usage", {})) if isinstance(caption_result.get("usage"), dict) else {},
        },
    }


def _visual_processing(
    status: str,
    latest_rgb: dict[str, Any] | None,
    model_inputs: list[dict[str, Any]],
    caption_metrics: dict[str, Any],
) -> dict[str, Any]:
    outputs = [
        {
            "type": item.get("type"),
            "role": item.get("role"),
            "metadata": dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {},
        }
        for item in model_inputs
    ]
    return {
        "strategy": STRATEGY if status == "captioned" else "phase_aware_caption",
        "status": status,
        "input": caption_image_metadata(_frame_path(latest_rgb) if latest_rgb else None, latest_rgb),
        "outputs": outputs,
        "caption_metrics": caption_metrics,
    }


def _caption_metrics(caption_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration_sec": caption_result.get("duration_sec"),
        "usage": dict(caption_result.get("usage", {})) if isinstance(caption_result.get("usage"), dict) else {},
    }


def _caption_failure_warning(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    return f"visual_caption_failed:{message[:240]}"
