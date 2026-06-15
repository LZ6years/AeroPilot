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

    sanitized_context = sanitize_context_for_model_text(context)
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
            if part is not None:
                label = _input_label_part(item)
                if label is not None:
                    parts.append(label)
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


def _input_label_part(item: dict[str, Any]) -> dict[str, Any] | None:
    label = _input_label_text(item)
    if not label:
        return None
    return {"type": "text", "text": label}


def _input_label_text(item: dict[str, Any]) -> str:
    if not any(isinstance(item.get(key), str) and item.get(key, "").strip() for key in ("input_id", "usage")):
        return ""
    fields = []
    for key in ("input_id", "role", "usage"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            fields.append(f"{key}={value.strip()}")
    if not fields:
        return ""
    return "Model input label: " + "; ".join(fields).rstrip(".") + "."


def _text_part(item: dict[str, Any]) -> dict[str, Any] | None:
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {"type": "text", "text": text}


def sanitize_context_for_model_text(context: dict[str, Any]) -> dict[str, Any]:
    """Return the lightweight JSON payload that accompanies model inputs."""

    sanitized = copy.deepcopy(context)
    _remove_debug_only_top_level_fields(sanitized)
    fragments = sanitized.get("fragments", [])
    if not isinstance(fragments, list):
        return sanitized
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        fragment.pop("metadata", None)
        model_inputs = fragment.get("model_inputs", [])
        if not isinstance(model_inputs, list):
            continue
        for item in model_inputs:
            if not isinstance(item, dict):
                continue
            item.pop("metadata", None)
            if "data" in item:
                item["data"] = _omitted_data_marker(item.get("data"))
    return sanitized


def _remove_debug_only_top_level_fields(context: dict[str, Any]) -> None:
    """Keep runtime/safety audit data out of the model-facing text payload.

    Safety constraints and observation policy remain in the recorder/debug
    context. The action planner only needs the tool contract and the rule that
    observation is automatic; the SafetySupervisor enforces live safety limits.
    Post-action phase judgment keeps full audit payloads on disk, while the
    model-facing text sees only semantic progress, executed actions, compact
    observations, measurements, preferences, and attached visual evidence.
    """

    context.pop("safety_constraints", None)
    context.pop("observation_policy", None)
    _remove_phase_judge_debug_fields(context)
    schema = context.get("context_schema")
    if not isinstance(schema, dict):
        return
    fields = schema.get("planner_context_fields")
    if isinstance(fields, dict):
        fields.pop("safety_constraints", None)
        fields.pop("observation_policy", None)


def _remove_phase_judge_debug_fields(context: dict[str, Any]) -> None:
    for key in (
        "planner_context_summary",
        "action_window",
        "action_plan",
        "safety_intervention_summary",
        "yaw_scan_remaining_deg",
        "visual_phase_budget_exhausted",
        "recent_executed_tools",
        "recent_safety_reasons",
        "safety_interventions",
        "recent_history",
        "observation_before",
        "phase_window_count",
        "max_phase_window_count",
    ):
        context.pop(key, None)
    if isinstance(context.get("executed_steps"), list):
        context["executed_steps"] = [_compact_executed_step(item) for item in context["executed_steps"]]
    if isinstance(context.get("observation_after"), dict):
        context["observation_after"] = _compact_observation(context["observation_after"])


def _compact_executed_step(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    step = item.get("step", {}) if isinstance(item.get("step"), dict) else {}
    result = item.get("result", {}) if isinstance(item.get("result"), dict) else {}
    return {
        "phase": item.get("phase"),
        "step_index": item.get("step_index"),
        "tool": step.get("tool"),
        "args": step.get("args"),
        "ok": result.get("ok"),
        "message": result.get("message"),
    }


def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    frames = observation.get("frames") if isinstance(observation.get("frames"), dict) else {}
    return {
        "source": observation.get("source"),
        "timestamp": observation.get("timestamp"),
        "modalities": observation.get("modalities") or list(frames.keys()),
        "pose": observation.get("pose"),
        "has_rgb": "rgb" in frames or "rgb" in (observation.get("modalities") or []),
        "has_depth": "depth" in frames or "depth" in (observation.get("modalities") or []),
    }


def _omitted_data_marker(value: Any) -> str:
    if isinstance(value, str):
        return f"<omitted encoded model input data: {len(value)} chars>"
    return "<omitted encoded model input data>"
