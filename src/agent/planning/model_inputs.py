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
    _remove_model_facing_direction_shorthand(sanitized)
    _normalize_model_facing_phase_completion(sanitized)
    _compact_model_facing_phase_execution_summary(sanitized)
    fragments = sanitized.get("fragments", [])
    if isinstance(fragments, list):
        compact_fragments = []
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            if fragment.get("name") == "skills":
                _clean_selected_skills(fragment)
            compact_fragment = _compact_model_facing_fragment(fragment)
            if compact_fragment is not None:
                compact_fragments.append(compact_fragment)
        sanitized["fragments"] = compact_fragments
    else:
        sanitized.pop("fragments", None)
    _remove_empty_and_debug_top_level_fields(sanitized)
    return _order_model_text_context(sanitized)


def _clean_selected_skills(fragment: dict[str, Any]) -> None:
    content = fragment.get("content")
    if not isinstance(content, dict):
        return
    selected = content.get("selected_skills")
    if not isinstance(selected, list):
        return
    allowed = {"name", "description", "skill_text", "requested_references", "capability"}
    for index, item in enumerate(list(selected)):
        if not isinstance(item, dict):
            continue
        selected[index] = {key: value for key, value in item.items() if key in allowed and value not in (None, "", [], {})}


def _compact_model_facing_fragment(fragment: dict[str, Any]) -> dict[str, Any] | None:
    compact: dict[str, Any] = {}
    name = fragment.get("name")
    if isinstance(name, str) and name.strip():
        compact["name"] = name

    content = fragment.get("content")
    if isinstance(content, dict):
        cleaned_content = _drop_empty_values(content)
        if cleaned_content:
            compact["content"] = cleaned_content

    model_inputs = fragment.get("model_inputs")
    if isinstance(model_inputs, list):
        cleaned_inputs = [_compact_model_input(item) for item in model_inputs if isinstance(item, dict)]
        cleaned_inputs = [item for item in cleaned_inputs if item]
        if cleaned_inputs:
            compact["model_inputs"] = cleaned_inputs

    for key in ("risks", "warnings"):
        value = fragment.get(key)
        if isinstance(value, list) and value:
            compact[key] = value

    return compact or None


def _compact_model_input(item: dict[str, Any]) -> dict[str, Any]:
    compact = dict(item)
    compact.pop("metadata", None)
    compact.pop("path", None)
    compact.pop("artifact_ref", None)
    compact.pop("source_path", None)
    if "data" in compact:
        compact["data"] = _omitted_data_marker(compact.get("data"))
    return _drop_empty_values(compact)


def _drop_empty_values(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            cleaned = _drop_empty_values(value)
            if cleaned:
                result[key] = cleaned
            continue
        if isinstance(value, list):
            cleaned_list = []
            for item in value:
                if isinstance(item, dict):
                    cleaned_item = _drop_empty_values(item)
                    if cleaned_item:
                        cleaned_list.append(cleaned_item)
                elif item not in (None, "", [], {}):
                    cleaned_list.append(item)
            if cleaned_list:
                result[key] = cleaned_list
            continue
        result[key] = value
    return result


def _order_model_text_context(context: dict[str, Any]) -> dict[str, Any]:
    """Put global flight rules before task-specific fragments in model text."""

    preferred_order = (
        "rules",
        "active_instruction",
        "current_phase",
        "semantic_goal",
        "runtime_control",
        "phase_progress_evidence",
        "fragments",
        "allowed_action_constraints",
        "phase_execution_summary",
        "user_preferences",
        "phase_completion_evaluation",
        "latest_observation_summary",
    )
    ordered: dict[str, Any] = {}
    for key in preferred_order:
        if key in context:
            ordered[key] = context[key]
    for key, value in context.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _remove_debug_only_top_level_fields(context: dict[str, Any]) -> None:
    """Keep runtime/safety audit data out of the model-facing text payload.

    Safety constraints and observation policy remain in the recorder/debug
    context. The coupled planner only needs the tool contract and compact
    evidence that supports the next phase decision; the SafetySupervisor
    enforces live safety limits.
    """

    context.pop("safety_constraints", None)
    context.pop("observation_policy", None)
    _remove_phase_judge_debug_fields(context)
    context.pop("context_schema", None)
    context.pop("metadata", None)
    context.pop("disabled_fragments", None)


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
        "args": _model_facing_tool_args(step.get("tool"), step.get("args")),
        "ok": result.get("ok"),
        "message": result.get("message"),
    }


def _remove_model_facing_direction_shorthand(context: dict[str, Any]) -> None:
    """Strip legacy move direction shorthands from model-facing JSON only."""

    constraints = context.get("allowed_action_constraints")
    if isinstance(constraints, dict):
        action_schemas = constraints.get("action_schemas")
        if isinstance(action_schemas, dict):
            move_schema = action_schemas.get("uav.move_body")
            if isinstance(move_schema, dict):
                _sanitize_move_body_schema(move_schema)

    summary = context.get("phase_execution_summary")
    if isinstance(summary, dict):
        _sanitize_phase_execution_summary(summary)

    fragments = context.get("fragments")
    if isinstance(fragments, list):
        for fragment in fragments:
            if isinstance(fragment, dict):
                _sanitize_action_fragment(fragment)


def _sanitize_move_body_schema(schema: dict[str, Any]) -> None:
    args_one_of = schema.get("args_one_of")
    if not isinstance(args_one_of, list):
        return
    numeric_options = [
        item
        for item in args_one_of
        if isinstance(item, dict) and "direction" not in item
    ]
    schema["args_one_of"] = numeric_options or [
        {"vx": "number", "vy": "number", "vz": "number", "duration": "number"}
    ]


def _sanitize_phase_execution_summary(summary: dict[str, Any]) -> None:
    tool_facts = summary.get("tool_facts")
    if not isinstance(tool_facts, dict):
        return
    tool_facts.pop("move_duration_by_direction_success", None)
    tool_facts.pop("move_duration_by_direction_attempted", None)


def _normalize_model_facing_phase_completion(context: dict[str, Any]) -> None:
    evaluation = context.get("phase_completion_evaluation")
    if not isinstance(evaluation, dict):
        return
    measurable_count = int(evaluation.get("measurable_count") or 0)
    metrics_count = int(evaluation.get("metrics_count") or 0)
    if metrics_count == 0 or measurable_count == 0:
        context["phase_completion_evaluation"] = {
            "deterministic_status": "not_measurable",
            "requires_model_judgment": True,
            "reason": "No deterministic phase metrics are available; judge completion from current_phase done_when/continue_when, images, and phase_progress_evidence.",
        }


def _compact_model_facing_phase_execution_summary(context: dict[str, Any]) -> None:
    summary = context.get("phase_execution_summary")
    if not isinstance(summary, dict):
        return
    tool_facts = summary.get("tool_facts") if isinstance(summary.get("tool_facts"), dict) else {}
    compact: dict[str, Any] = {}
    for key in ("phase", "window_count", "max_window_count"):
        if summary.get(key) is not None:
            compact[key] = summary[key]
    if isinstance(tool_facts, dict) and tool_facts.get("tools_executed"):
        compact["tools_executed"] = dict(tool_facts["tools_executed"])
    context["phase_execution_summary"] = compact


def _remove_empty_and_debug_top_level_fields(context: dict[str, Any]) -> None:
    for key in ("phase_metrics",):
        value = context.get(key)
        if value in (None, "", [], {}):
            context.pop(key, None)
    observation = context.get("latest_observation_summary")
    if isinstance(observation, dict):
        context["latest_observation_summary"] = _compact_latest_observation_summary(observation)
    for key in list(context):
        if context.get(key) in (None, "", [], {}):
            context.pop(key, None)


def _compact_latest_observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "source": observation.get("source"),
        "timestamp": observation.get("timestamp"),
        "modalities": observation.get("modalities"),
        "pose": observation.get("pose"),
    }
    frames = observation.get("frames")
    if isinstance(frames, dict):
        compact["has_rgb"] = "rgb" in frames
        compact["has_depth"] = "depth" in frames
    return _drop_empty_values(compact)


def _sanitize_action_fragment(fragment: dict[str, Any]) -> None:
    if fragment.get("name") != "action":
        return
    content = fragment.get("content")
    if not isinstance(content, dict):
        return
    recent_actions = content.get("recent_actions")
    if isinstance(recent_actions, list):
        for item in recent_actions:
            if not isinstance(item, dict):
                continue
            item["args"] = _model_facing_tool_args(item.get("tool"), item.get("args"))


def _model_facing_tool_args(tool: Any, args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    compact = dict(args)
    if tool == "uav.move_body" and "direction" in compact:
        compact.pop("direction", None)
        compact["legacy_named_motion"] = True
    return compact


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
