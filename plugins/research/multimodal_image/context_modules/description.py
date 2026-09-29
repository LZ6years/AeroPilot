"""Normalize and sanitize phase-aware visual descriptions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.memory.context.contracts import TaskContextInput

from .constants import (
    ALLOWED_POSITIONS,
    ALLOWED_ROLES,
    SCHEMA_VERSION,
    SOFT_ACTION_REPLACEMENTS,
    STRATEGY,
)


def normalize_visual_description(
    raw: dict[str, Any],
    image_path: Path,
    stage_context: TaskContextInput | None = None,
) -> dict[str, Any]:
    data = raw.get("visual_description") if isinstance(raw.get("visual_description"), dict) else raw
    phase = data.get("phase") if isinstance(data.get("phase"), dict) else {}
    stage_phase = {}
    description = {
        "schema_version": str(data.get("schema_version") or SCHEMA_VERSION),
        "strategy": str(data.get("strategy") or STRATEGY),
        "source_image": str(data.get("source_image") or image_path),
        "phase": {
            "name": _text_field(phase.get("name") or stage_phase.get("name")),
            "goal": _text_field(phase.get("goal") or stage_phase.get("goal")),
        },
        "phase_relevance": _text_field(data.get("phase_relevance")),
        "scene_summary": _text_field(data.get("scene_summary")),
        "spatial_overview": _text_field(data.get("spatial_overview") or data.get("spatial_layout")),
        "relevant_objects": _relevant_objects(data.get("relevant_objects") or data.get("visible_objects"))[:4],
        "free_space": _list_of_strings(data.get("free_space"))[:2],
        "landmarks": _list_of_strings(data.get("landmarks"))[:2],
        "phase_relevant_cues": _list_of_strings(
            data.get("phase_relevant_cues")
            or data.get("navigation_relevant_cues")
            or data.get("possible_navigation_cues")
        )[:2],
        "uncertainty": _list_of_strings(data.get("uncertainty"))[:2],
        "confidence": _confidence(data.get("confidence")),
    }
    return {"visual_description": _sanitize_visual_description(description)}


def visual_description_text(visual_description: dict[str, Any]) -> str:
    return json.dumps({"visual_description": visual_description}, ensure_ascii=False, separators=(",", ":"))


def _list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_stringify_list_item(item) for item in value if _stringify_list_item(item)]
    if isinstance(value, str) and value.strip():
        return [_sanitize_soft_action_text(value)]
    return []


def _relevant_objects(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [_relevant_object(item) for item in value if item]


def _relevant_object(item: Any) -> dict[str, str]:
    if isinstance(item, dict):
        name = _text_field(item.get("name") or item.get("class") or item.get("type") or item.get("label") or "unknown")
        position = _position(item.get("position"))
        role = _role(item.get("role"))
        phase_relevance = _text_field(item.get("phase_relevance") or item.get("relevance") or item.get("why_relevant"))
        description = _text_field(item.get("description") or item.get("text") or item.get("value"))
        confidence = _confidence(item.get("confidence"))
    else:
        name = _text_field(item)
        position = "unknown"
        role = "unknown"
        phase_relevance = "模型未提供阶段相关性。"
        description = name
        confidence = "medium"
    return {
        "name": name or "unknown",
        "position": position,
        "role": role,
        "phase_relevance": phase_relevance or "与当前阶段可能相关，但模型未说明原因。",
        "description": description or name or "unknown",
        "confidence": confidence,
    }


def _position(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace("_", "-")
    return text if text in ALLOWED_POSITIONS else "unknown"


def _role(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in ALLOWED_ROLES else "unknown"


def _text_field(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, list):
                item_text = "、".join(str(part) for part in item if str(part).strip())
            else:
                item_text = str(item).strip()
            if item_text:
                parts.append(f"{key}: {item_text}")
        return _sanitize_soft_action_text("；".join(parts))
    if isinstance(value, list):
        return _sanitize_soft_action_text("；".join(str(item).strip() for item in value if str(item).strip()))
    return _sanitize_soft_action_text(str(value or "").strip())


def _stringify_list_item(item: Any) -> str:
    if isinstance(item, dict):
        label = item.get("class") or item.get("cue_type") or item.get("type") or item.get("name")
        description = item.get("description") or item.get("value") or item.get("text")
        if label and description:
            return _sanitize_soft_action_text(f"{label}: {description}")
        if description:
            return _sanitize_soft_action_text(str(description))
        return _sanitize_soft_action_text(json.dumps(item, ensure_ascii=False))
    return _sanitize_soft_action_text(str(item).strip())


def _sanitize_soft_action_text(text: str) -> str:
    sanitized = text
    for old, new in SOFT_ACTION_REPLACEMENTS.items():
        sanitized = sanitized.replace(old, new)
    return sanitized


def _sanitize_visual_description(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_visual_description(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_visual_description(item) for item in value]
    if isinstance(value, str):
        return _sanitize_soft_action_text(value)
    return value


def _confidence(value: Any) -> str:
    text = str(value or "medium").lower()
    return text if text in {"high", "medium", "low"} else "medium"
