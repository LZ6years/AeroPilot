"""Recent RGB metadata context module."""

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
        observations = reader.read_refs(memory.recent_observation_refs[-self.max_items :]) if reader else []
        rgb_frames = _recent_rgb_frames(reader, observations, self.max_items) if reader else []
        latest_rgb = rgb_frames[-1] if rgb_frames else None
        frames = {"rgb": latest_rgb} if latest_rgb else {}
        artifact_refs = [item["artifact_ref"] for item in rgb_frames if item.get("artifact_ref")]
        facts = []
        warnings = []
        risks = []
        if rgb_frames:
            facts.append(
                {
                    "type": "available_modalities",
                    "value": ["rgb"],
                    "source": self.__class__.__name__,
                }
            )
        else:
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
        model_inputs = latest_rgb_model_inputs_from_frame(latest_rgb)
        visual_evidence["interpretation_status"] = (
            "image_attached_not_interpreted" if model_inputs else "metadata_only_not_interpreted"
        )
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "latest_frame_metadata": frames,
                "recent_rgb_frame_metadata": rgb_frames,
                "recent_observation_sources": [item.get("source") for item in observations if isinstance(item, dict)],
                "visual_evidence": visual_evidence,
            },
            model_inputs=model_inputs,
            facts=facts,
            risks=risks,
            artifact_refs=artifact_refs,
            confidence="medium" if frames else "low",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=300, max_items=self.max_items),
            warnings=warnings,
            metadata={"observation_count": len(observations)},
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Recent RGB frame metadata, artifact references, and visual evidence availability.",
            "content_fields": {
                "latest_frame_metadata": "Latest available frame metadata grouped by modality; currently contains latest RGB metadata when present.",
                "recent_rgb_frame_metadata": "Up to three recent RGB frame metadata records, including path/artifact reference and frame metadata.",
                "recent_observation_sources": "Sources of the observations used to build this image context.",
                "visual_evidence": "Boolean availability summary such as has_rgb, has_depth, has_mock_frames, latest_observation_source, and interpretation_status.",
            },
            "model_inputs": (
                "At most one encoded latest real RGB image for planner multimodal input. "
                "Mock, missing, or unreadable images are not attached."
            ),
            "fact_fields": {
                "available_modalities": "Modalities currently available as observed evidence.",
            },
            "risk_fields": {
                "visual_evidence_missing": "No recent RGB evidence is available.",
                "visual_evidence_is_mock": "A recent frame is mock/synthetic placeholder evidence and should not support confident target approach.",
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
    artifact_ref = latest_rgb.get("artifact_ref") if isinstance(latest_rgb.get("artifact_ref"), str) else str(path)
    return [
        {
            "type": "image",
            "source": "image",
            "role": "latest_observation",
            "mime_type": mime_type,
            "encoding": "data_url",
            "data": data_url,
            "metadata": {
                "source_path": str(path),
                "artifact_ref": artifact_ref,
                "modality": "rgb",
                "processing": "select_latest_rgb",
                "mock": False,
            },
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
