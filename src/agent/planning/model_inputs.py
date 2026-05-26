"""Assemble model-facing user content from planner context inputs.

Context modules own data preparation. They may select, fuse, caption, resize,
or encode visual artifacts before exposing the final provider-ready inputs via
ContextFragment.model_inputs. This module only adapts those encoded inputs into
OpenAI-compatible message content and removes bulky binary data from the JSON
context text.
"""

from __future__ import annotations

import copy
import json
from typing import Any


def build_model_user_content(context: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Build an OpenAI-compatible user message from sanitized context and inputs."""

    sanitized_context = _sanitize_context(context)
    text = json.dumps(sanitized_context, ensure_ascii=False)
    parts = _model_input_parts(context)
    if not parts:
        return text
    return [{"type": "text", "text": text}, *parts]


def _model_input_parts(context: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for item in _iter_model_inputs(context):
        input_type = item.get("type")
        if input_type == "image":
            part = _image_part(item)
        elif input_type == "text":
            part = _text_part(item)
        else:
            part = None
        if part is not None:
            parts.append(part)
    return parts


def _iter_model_inputs(context: dict[str, Any]):
    fragments = context.get("fragments", [])
    if not isinstance(fragments, list):
        return
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        model_inputs = fragment.get("model_inputs", [])
        if not isinstance(model_inputs, list):
            continue
        for item in model_inputs:
            if isinstance(item, dict):
                yield item


def _image_part(item: dict[str, Any]) -> dict[str, Any] | None:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("mock") is True:
        return None
    data = item.get("data")
    if item.get("encoding") == "data_url" and isinstance(data, str) and data.startswith("data:image/"):
        return {"type": "image_url", "image_url": {"url": data}}
    return None


def _text_part(item: dict[str, Any]) -> dict[str, Any] | None:
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {"type": "text", "text": text}


def _sanitize_context(context: dict[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(context)
    fragments = sanitized.get("fragments", [])
    if not isinstance(fragments, list):
        return sanitized
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        model_inputs = fragment.get("model_inputs", [])
        if not isinstance(model_inputs, list):
            continue
        for item in model_inputs:
            if isinstance(item, dict) and "data" in item:
                item["data"] = _omitted_data_marker(item.get("data"))
    return sanitized


def _omitted_data_marker(value: Any) -> str:
    if isinstance(value, str):
        return f"<omitted encoded model input data: {len(value)} chars>"
    return "<omitted encoded model input data>"
