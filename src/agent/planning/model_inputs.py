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


def build_model_user_content(
    context: dict[str, Any],
    *,
    image_mode: str = "all",
    include_memory_documents: bool = True,
) -> str | list[dict[str, Any]]:
    """Build an OpenAI-compatible user message from sanitized context and inputs.

    ``current_only`` is a provider-recovery mode. It omits phase-start media
    and the two-image temporal rule while preserving the normal model context.
    """

    _validate_image_mode(image_mode)

    sanitized_context = sanitize_context_for_model_text(
        context,
        include_memory_documents=include_memory_documents,
    )
    text = json.dumps(sanitized_context, ensure_ascii=False)
    input_parts = _model_input_parts(
        context,
        image_mode=image_mode,
    )
    if not input_parts:
        return text
    # Context modules deliberately interleave semantic instructions and media.
    # Preserve that order exactly and append the structured JSON only after the
    # module-owned sequence. The assembler does not interpret temporal roles.
    return [*input_parts, {"type": "text", "text": text}]


def _model_input_parts(
    context: dict[str, Any],
    *,
    image_mode: str,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    model_inputs = list(_iter_model_inputs(context))
    selected_images = {
        id(item) for item in select_model_image_inputs(
            context,
            image_mode=image_mode,
        )
    }
    for item in model_inputs:
        input_type = item.get("type")
        if input_type == "image":
            if id(item) not in selected_images:
                continue
            part = _image_part(item)
            if part is not None:
                label = _input_label_part(item)
                if label is not None:
                    parts.append(label)
                parts.append(part)
        elif input_type == "text":
            if image_mode == "current_only" and item.get("role") == "temporal_evidence_rule":
                continue
            part = _text_part(item)
            if part is not None:
                parts.append(part)
    return parts


def select_model_image_inputs(
    context: dict[str, Any],
    *,
    image_mode: str = "all",
) -> list[dict[str, Any]]:
    """Return exactly the valid image inputs sent for one request mode.

    The action planner uses this selector immediately before provider-message
    assembly.  Human review reuses it to materialize the same successful
    request images without inferring them from observation history.
    """

    _validate_image_mode(image_mode)
    model_inputs = list(_iter_model_inputs(context))
    current_recovery_role = _current_recovery_role(model_inputs)
    return [
        item
        for item in model_inputs
        if item.get("type") == "image"
        and (
            image_mode == "all"
            or item.get("role") == current_recovery_role
        )
        and _image_part(item) is not None
    ]


def _validate_image_mode(image_mode: str) -> None:
    if image_mode not in {"all", "current_only"}:
        raise ValueError(f"unsupported image_mode: {image_mode}")


def _current_recovery_role(model_inputs: list[dict[str, Any]]) -> str:
    """Choose one current image role for provider recovery.

    A valid overlay contains the current RGB scene as well as aligned distance
    cues, so it is the most informative single-image retry. Raw current RGB is
    retained as the fallback for profiles without an overlay.
    """

    if any(
        item.get("type") == "image"
        and item.get("role") == "current_rgb_depth_overlay"
        for item in model_inputs
    ):
        return "current_rgb_depth_overlay"
    return "current_observation"


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
    usage = item.get("usage")
    if isinstance(usage, str) and usage.strip():
        return usage.strip()
    role = item.get("role")
    if isinstance(role, str) and role.strip():
        return f"Image role: {role.strip()}."
    return ""


def _text_part(item: dict[str, Any]) -> dict[str, Any] | None:
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {"type": "text", "text": text}


def sanitize_context_for_model_text(
    context: dict[str, Any],
    *,
    include_memory_documents: bool = True,
) -> dict[str, Any]:
    """Return the lightweight JSON payload that accompanies model inputs."""

    sanitized = copy.deepcopy(context)
    _remove_debug_only_top_level_fields(sanitized)
    _remove_legacy_semantic_metric_fields(sanitized)
    if not include_memory_documents:
        # ModelActionPlanner promotes these stable documents to separate system
        # messages. Keeping duplicate copies in JSON weakens their role and
        # wastes context, while standalone callers retain the previous default.
        sanitized.pop("rules", None)
        sanitized.pop("user_preferences", None)
    _remove_model_facing_direction_shorthand(sanitized)
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
    # Image roles are already bound to provider-ready media by adjacent labels.
    # Repeating selection/debug metadata in JSON adds no evidence and can make
    # the model treat "not interpreted" audit state as a completion blocker.
    if name == "image":
        return None
    if isinstance(name, str) and name.strip():
        compact["name"] = name

    content = fragment.get("content")
    if isinstance(content, dict):
        model_content = copy.deepcopy(content)
        if name == "pose":
            model_content = _compact_model_facing_pose(model_content)
        for key in (
            "visual_evidence",
            "selected_frames",
            "interpretation_status",
            "attached_image_count",
        ):
            model_content.pop(key, None)
        cleaned_content = _drop_empty_values(model_content)
        if cleaned_content:
            compact["content"] = cleaned_content

    for key in ("risks", "warnings"):
        value = fragment.get(key)
        if isinstance(value, list) and value:
            compact[key] = value

    # A calibrated AGL value is the pose module's only model-facing evidence.
    # Do not retain a name-only pose fragment when a simulator cannot provide
    # that optional value.
    if name == "pose" and "content" not in compact and not any(
        key in compact for key in ("risks", "warnings")
    ):
        return None
    return compact or None


def _compact_model_facing_pose(content: dict[str, Any]) -> dict[str, Any]:
    """Defensively keep only optional AGL from any pose context producer.

    Custom or legacy modules may still publish absolute world pose fields.
    Those values are useful in the full planner/debug payload but must not
    regain model visibility through a compatibility path.
    """

    if "agl_m" not in content:
        return {}
    compact = {"agl_m": content["agl_m"]}
    semantics = content.get("field_semantics")
    if isinstance(semantics, dict) and "agl_m" in semantics:
        compact["field_semantics"] = {"agl_m": semantics["agl_m"]}
    return compact


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
        "active_instruction",
        "current_phase",
        "phase_strategy",
        "phase_neighborhood",
        "semantic_goal",
        "runtime_control",
        "fragments",
        "allowed_action_constraints",
        "rules",
        "user_preferences",
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
    context.pop("phase_progress_evidence", None)
    context.pop("phase_completion_evaluation", None)
    context.pop("phase_metrics", None)
    context.pop("phase_execution_summary", None)
    context.pop("latest_observation_summary", None)
    # Backend capability details belong to Runtime lifecycle handling and its
    # state artifact. The model receives only the already-filtered canonical
    # action contract. Keeping this defensive removal prevents a custom or
    # legacy context source from reintroducing the obsolete duplicate field.
    context.pop("simulator_contract", None)


def _remove_legacy_semantic_metric_fields(context: dict[str, Any]) -> None:
    """Prevent stale custom contexts from reviving the removed metric schema."""

    current_phase = context.get("current_phase")
    if isinstance(current_phase, dict):
        current_phase.pop("metrics", None)


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


def _remove_empty_and_debug_top_level_fields(context: dict[str, Any]) -> None:
    for key in list(context):
        if context.get(key) in (None, "", [], {}):
            context.pop(key, None)


def _sanitize_action_fragment(fragment: dict[str, Any]) -> None:
    if fragment.get("name") != "action":
        return
    content = fragment.get("content")
    if not isinstance(content, dict):
        return
    content.pop("recent_actions", None)
    content.pop("action_trend", None)
    summary = content.get("motion_history_summary")
    if not isinstance(summary, dict):
        return
    recent_execution = summary.get("recent_execution")
    if isinstance(recent_execution, list):
        for item in recent_execution:
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
