"""Render portable benchmark review pages from structured audit data.

The renderer owns simulator-neutral audit views, planner-window diagnostics,
portable links, and shared CSS. Benchmarks contribute metrics and typed review
sections; they never need to parse Runtime memory or inject arbitrary HTML.
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.benchmarks.core.audit import (
    BenchmarkAuditLog,
    ResolvedFrameAsset,
)
from src.agent.planning.model_inputs import build_model_user_content


@dataclass(frozen=True)
class ReviewSection:
    """One benchmark-owned structured extension section."""

    title: str
    kind: str
    payload: Any

    def __post_init__(self) -> None:
        if self.kind not in {"text", "key_value", "table", "image_group"}:
            raise ValueError(f"Unsupported review section kind: {self.kind}")


@dataclass(frozen=True)
class BenchmarkCaseReview:
    """Complete input for one common case review page."""

    case_id: str
    benchmark_name: str
    protocol_variant: str
    instruction: str
    outcome: dict[str, Any]
    metrics: dict[str, Any]
    audit: BenchmarkAuditLog
    frame_assets: tuple[ResolvedFrameAsset, ...]
    # Human galleries default to the camera image used for visual review.
    # Other recorded modalities remain available in audit.json and may be
    # enabled explicitly by a benchmark without changing Runtime capture.
    gallery_modalities: tuple[str, ...] = ("rgb",)
    sections: tuple[ReviewSection, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSummaryReview:
    """Complete input for one evaluation-level review index."""

    benchmark_name: str
    protocol_variant: str
    metrics: dict[str, Any]
    cases: tuple[dict[str, Any], ...]
    failure_counts: dict[str, Any] = field(default_factory=dict)


class BenchmarkReviewRenderer:
    """Write shared review CSS plus evaluation and case HTML pages."""

    def write(
        self,
        eval_dir: str | Path,
        summary: BenchmarkSummaryReview,
        cases: tuple[BenchmarkCaseReview, ...],
    ) -> dict[str, str]:
        root = Path(eval_dir)
        style_path = root / "review_assets" / "benchmark_review.css"
        style_path.parent.mkdir(parents=True, exist_ok=True)
        style_path.write_text(_STYLE, encoding="utf-8")
        for case in cases:
            case_dir = root / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "review.html").write_text(
                self._case_html(case, case_dir, style_path),
                encoding="utf-8",
            )
        index = root / "review.html"
        index.write_text(
            self._summary_html(summary, root, style_path),
            encoding="utf-8",
        )
        return {
            "html_review": str(index),
            "review_css": str(style_path),
        }

    def _summary_html(
        self,
        summary: BenchmarkSummaryReview,
        root: Path,
        style_path: Path,
    ) -> str:
        rows = []
        for case in summary.cases:
            case_id = str(case.get("case_id") or "")
            href = f"cases/{_url_part(case_id)}/review.html"
            rows.append(
                "<tr>"
                f'<td><a href="{href}"><code>{_e(case_id)}</code></a></td>'
                f"<td>{_e(case.get('status'))}</td>"
                f"<td>{_metrics_inline(case.get('metrics'))}</td>"
                "</tr>"
            )
        return _document(
            title=f"{summary.benchmark_name} Benchmark Review",
            style_href=_relative_href(root, style_path),
            body=(
                f"<h1>{_e(summary.benchmark_name)} Benchmark Review</h1>"
                '<p class="muted">Protocol: '
                f"<code>{_e(summary.protocol_variant)}</code></p>"
                "<section><h2>Aggregate Metrics</h2>"
                f"{_key_value_table(summary.metrics)}</section>"
                "<section><h2>Failure Counts</h2>"
                f"{_key_value_table(summary.failure_counts)}</section>"
                "<section><h2>Cases</h2>"
                "<table><thead><tr><th>Case</th><th>Status</th>"
                "<th>Metrics</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table></section>"
            ),
        )

    def _case_html(
        self,
        case: BenchmarkCaseReview,
        case_dir: Path,
        style_path: Path,
    ) -> str:
        gallery = _frame_gallery(
            case.audit,
            case.frame_assets,
            case_dir,
            gallery_modalities=case.gallery_modalities,
        )
        diagnostics = _json_details(
            case.audit.diagnostics,
            "Diagnostics",
            empty_message="No audit diagnostics were recorded.",
        )
        return _document(
            title=f"{case.benchmark_name}: {case.case_id}",
            style_href=_relative_href(case_dir, style_path),
            body=(
                '<p><a href="../../review.html">← Evaluation index</a></p>'
                f"<h1>{_e(case.case_id)}</h1>"
                f'<p class="instruction">{_e(case.instruction)}</p>'
                '<p class="muted">Protocol: '
                f"<code>{_e(case.protocol_variant)}</code></p>"
                "<section><h2>Outcome</h2>"
                f"{_key_value_table(case.outcome)}</section>"
                "<section><h2>Metrics</h2>"
                f"{_key_value_table(case.metrics)}</section>"
                "<section><h2>Observation Gallery</h2>"
                f"{gallery}</section>"
                "<section><h2>Action Timeline</h2>"
                + _record_table(
                    case.audit.actions,
                    ("id", "timestamp", "tool", "args", "result"),
                )
                + "</section>"
                "<section><h2>Agent Messages</h2>"
                + _record_table(
                    case.audit.agent_messages,
                    ("timestamp", "source", "phase", "message"),
                    empty_message=(
                        "No agent messages were emitted for this episode."
                    ),
                )
                + "</section>"
                "<section><h2>Planner Windows</h2>"
                f"{_planner_review(case.audit)}</section>"
                "<section><h2>Runtime Applied Phase State</h2>"
                f"{_phase_event_table(case.audit.events)}</section>"
                + "".join(
                    _section_html(section, case_dir)
                    for section in case.sections
                )
                + "<section><h2>Runtime Events</h2>"
                f"{_json_details(case.audit.events, 'All bounded events')}"
                "</section>"
                "<section><h2>Audit Diagnostics</h2>"
                f"{diagnostics}</section>"
            ),
        )


def _section_html(section: ReviewSection, case_dir: Path) -> str:
    if section.kind == "text":
        content = f"<p>{_e(section.payload)}</p>"
    elif section.kind == "key_value":
        content = _key_value_table(section.payload)
    elif section.kind == "table":
        rows = section.payload if isinstance(section.payload, list) else []
        keys = (
            tuple(rows[0].keys())
            if rows and isinstance(rows[0], dict)
            else ()
        )
        content = _record_table(tuple(rows), keys)
    else:
        content = _image_group(section.payload, case_dir)
    return f"<section><h2>{_e(section.title)}</h2>{content}</section>"


def _image_group(payload: Any, case_dir: Path) -> str:
    values = payload if isinstance(payload, list) else []
    cards = []
    for value in values:
        if isinstance(value, dict):
            path = Path(str(value.get("path") or ""))
            label = value.get("label") or path.name
            description = value.get("description")
        else:
            path = Path(str(value))
            label = path.name
            description = None
        if not path.is_file():
            continue
        caption = f"<strong>{_e(label)}</strong>"
        if description not in (None, ""):
            caption += f"<br><span>{_e(description)}</span>"
        cards.append(
            '<figure class="frame-card">'
            f'<a href="{_relative_href(case_dir, path)}">'
            f'<img loading="lazy" src="{_relative_href(case_dir, path)}" '
            f'alt="{_e(label)}"></a>'
            f"<figcaption>{caption}</figcaption></figure>"
        )
    if not cards:
        return '<p class="empty">No images available.</p>'
    return '<div class="gallery">' + "".join(cards) + "</div>"


def _frame_gallery(
    audit: BenchmarkAuditLog,
    assets: tuple[ResolvedFrameAsset, ...],
    case_dir: Path,
    *,
    gallery_modalities: tuple[str, ...],
) -> str:
    visible_modalities = set(gallery_modalities)
    gallery_assets = tuple(
        asset for asset in assets if asset.modality in visible_modalities
    )
    observations = {
        str(item.get("id")): item
        for item in audit.observations
        if item.get("id") not in (None, "")
    }
    anchor_roles: dict[str, list[str]] = {}
    for role, anchor in audit.observation_anchors.items():
        observation_id = anchor.get("id")
        if observation_id not in (None, ""):
            anchor_roles.setdefault(str(observation_id), []).append(role)

    key_cards = []
    all_cards = []
    for asset in gallery_assets:
        if not asset.display_path:
            continue
        observation = observations.get(str(asset.observation_id), {})
        roles = anchor_roles.get(str(asset.observation_id), [])
        association = _frame_association(
            observation,
            asset.timestamp,
            audit.actions,
        )
        card = _frame_card(
            asset,
            case_dir,
            roles=roles,
            observation_source=observation.get("source"),
            association=association,
        )
        all_cards.append(card)
        if roles or association:
            key_cards.append(card)

    unavailable = [
        item
        for item in gallery_assets
        if item.unavailable_reason is not None
    ]
    if not all_cards:
        body = (
            '<p class="empty">'
            + _empty_gallery_message(gallery_modalities)
            + "</p>"
        )
    else:
        selected = key_cards or all_cards[:1]
        body = (
            "<h3>Key Frames</h3><div class=\"gallery key\">"
            + "".join(selected)
            + "</div>"
            + "<details><summary>All observation frames "
            f"({len(all_cards)})</summary><div class=\"gallery\">"
            + "".join(all_cards)
            + "</div></details>"
        )
    if unavailable:
        body += (
            "<details><summary>Unavailable frame assets "
            f"({len(unavailable)})</summary>"
            + _unavailable_frame_table(unavailable, case_dir)
            + "</details>"
        )
    return body


def _empty_gallery_message(modalities: tuple[str, ...]) -> str:
    if modalities == ("rgb",):
        return "No RGB frame assets available."
    labels = ", ".join(item.upper() for item in modalities)
    if labels:
        return f"No {labels} frame assets available."
    return "No frame modalities are configured for this gallery."


def _frame_card(
    asset: ResolvedFrameAsset,
    case_dir: Path,
    *,
    roles: list[str],
    observation_source: Any,
    association: str | None,
) -> str:
    path = Path(str(asset.display_path))
    labels = []
    if roles:
        labels.append(", ".join(roles))
    if observation_source not in (None, ""):
        labels.append(str(observation_source))
    if association:
        labels.append(association)
    caption = (
        f"<code>{_e(asset.frame_id)}</code> · {_e(asset.modality)} · "
        f"{_e(asset.source)}"
    )
    if labels:
        caption += "<br><strong>" + _e(" · ".join(labels)) + "</strong>"
    raw_link = ""
    if asset.raw_path and asset.raw_path != asset.display_path:
        raw_link = (
            "<br><a href=\""
            + _relative_href(case_dir, Path(asset.raw_path))
            + "\">download raw asset</a>"
        )
    return (
        '<figure class="frame-card">'
        f'<a href="{_relative_href(case_dir, path)}">'
        f'<img loading="lazy" src="{_relative_href(case_dir, path)}" '
        f'alt="{_e(asset.frame_id)}"></a>'
        f"<figcaption>{caption}{raw_link}</figcaption></figure>"
    )


def _unavailable_frame_table(
    assets: list[ResolvedFrameAsset],
    case_dir: Path,
) -> str:
    rows = []
    for asset in assets:
        raw_link = "unavailable"
        if asset.raw_path and Path(asset.raw_path).is_file():
            raw_link = (
                f'<a href="{_relative_href(case_dir, Path(asset.raw_path))}">'
                "download raw asset</a>"
            )
        rows.append(
            "<tr>"
            f"<td><code>{_e(asset.frame_id)}</code></td>"
            f"<td>{_e(asset.modality)}</td>"
            f"<td>{_e(asset.unavailable_reason)}</td>"
            f"<td>{raw_link}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Frame</th><th>Modality</th>"
        "<th>Reason</th><th>Raw asset</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _frame_association(
    observation: dict[str, Any],
    timestamp: str | None,
    actions: tuple[dict[str, Any], ...],
) -> str | None:
    explicit_id = observation.get("action_id")
    if explicit_id not in (None, ""):
        action = next(
            (
                item
                for item in actions
                if str(item.get("id")) == str(explicit_id)
            ),
            None,
        )
        tool = action.get("tool") if action else None
        return f"after {tool or explicit_id}"

    if not timestamp:
        return None
    preceding = [
        item
        for item in actions
        if item.get("timestamp") not in (None, "")
        and str(item["timestamp"]) <= timestamp
    ]
    if not preceding:
        return None
    action = max(preceding, key=lambda item: str(item["timestamp"]))
    return f"after {action.get('tool') or action.get('id')} (time-associated)"


def _planner_review(audit: BenchmarkAuditLog) -> str:
    records = list(audit.planner_debug)
    if not records:
        return (
            '<p class="empty">Action planner debug was not enabled or no '
            "planner windows were recorded.</p>"
        )

    stats = _planner_stats(records, audit.actions)
    summary = _key_value_table(stats)
    timeline = _planner_timeline(records)
    details = "".join(_planner_window(record) for record in records)
    return (
        "<h3>Decision Summary</h3>"
        + summary
        + "<h3>Decision Timeline</h3>"
        + timeline
        + "<h3>Window Details</h3>"
        + details
    )


def _planner_stats(
    records: list[dict[str, Any]],
    actions: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    counts = {
        "continue_current_phase": 0,
        "advance_to_next_phase": 0,
        "complete_task": 0,
    }
    no_tool_transitions = 0
    for record in records:
        decision = _decision(record)
        name = str(decision.get("decision") or "")
        if name in counts:
            counts[name] += 1
        if name in {"advance_to_next_phase", "complete_task"} and not (
            _plan_tools(_dict(record.get("safe_plan")))
        ):
            no_tool_transitions += 1
    return {
        "window_count": len(records),
        "continue_count": counts["continue_current_phase"],
        "advance_count": counts["advance_to_next_phase"],
        "complete_count": counts["complete_task"],
        "no_tool_transition_count": no_tool_transitions,
        "behavior_flags": _behavior_flags(actions, records),
    }


def _planner_timeline(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        raw_tools = _plan_tools(_dict(record.get("raw_plan")))
        safe_tools = _plan_tools(_dict(record.get("safe_plan")))
        decision = _decision(record)
        rows.append(
            {
                "window": record.get("window_index"),
                "phase": record.get("phase"),
                "decision": decision.get("decision"),
                "tool_execution": _tool_execution_label(
                    decision,
                    safe_tools,
                ),
                "raw_tools": raw_tools,
                "safe_tools": safe_tools,
                "reason": (
                    decision.get("guard_reason")
                    or decision.get("reason")
                ),
            }
        )
    return _record_table(
        rows,
        (
            "window",
            "phase",
            "decision",
            "tool_execution",
            "raw_tools",
            "safe_tools",
            "reason",
        ),
    )


def _planner_window(record: dict[str, Any]) -> str:
    model_context = _dict(
        record.get("model_context") or record.get("planner_context")
    )
    current_phase = _dict(model_context.get("current_phase"))
    decision = _decision(record)
    raw_plan = _dict(record.get("raw_plan"))
    safe_plan = _dict(record.get("safe_plan"))
    safe_tools = _plan_tools(safe_plan)
    coupled_plan = _dict(record.get("coupled_plan"))
    completion = _dict(coupled_plan.get("completion_assessment"))
    completion_result = _dict(record.get("_phase_completion_result"))
    if not completion:
        completion = _dict(completion_result.get("assessment"))
    structured = bool(completion)
    title = (
        f"Window {record.get('window_index')} · {record.get('phase')} · "
        f"{decision.get('decision')} · "
        f"{_tool_execution_label(decision, safe_tools)}"
    )
    summary = {
        "instruction": model_context.get("active_instruction"),
        "current_phase": current_phase.get("name") or record.get("phase"),
        (
            "derived_phase_transition"
            if structured
            else "legacy_phase_decision"
        ): decision.get("decision"),
        "tool_execution": _tool_execution_label(decision, safe_tools),
        "transition_reason": (
            decision.get("guard_reason") or decision.get("reason")
        ),
        "phase_metrics": model_context.get("phase_metrics"),
        "phase_progress_evidence": _compact_progress(
            model_context.get("phase_progress_evidence")
        ),
        "raw_tools": _plan_tools(raw_plan),
        "safe_tools": safe_tools,
    }
    note = (
        "The model assessed done_when evidence and planner code derived the "
        "phase transition."
        if structured
        else "Legacy planner record: no structured completion assessment "
        "was recorded."
    )
    return (
        '<details class="planner-window"><summary>'
        + _e(title)
        + "</summary>"
        + f'<p class="muted">{_e(note)}</p>'
        + _key_value_table(summary)
        + _completion_assessment(
            completion,
            completion_result,
            current_phase,
            decision,
        )
        + "<h4>Model-Facing Context</h4><pre>"
        + _e(_model_facing_context_text(model_context))
        + "</pre><h4>Action Plan</h4>"
        + _json_details(_compact_plan(raw_plan), "Raw model plan")
        + _json_details(_compact_plan(safe_plan), "Safety-reviewed plan")
        + "<h4>Safety</h4>"
        + _json_details(record.get("safety_decision"), "Safety decision")
        + "<h4>Raw Planner Record</h4>"
        + _json_details(record, "Complete debug record")
        + "</details>"
    )


def _completion_assessment(
    assessment: dict[str, Any],
    result: dict[str, Any],
    current_phase: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    if not assessment:
        return ""
    done_when = current_phase.get("done_when")
    done_when = done_when if isinstance(done_when, list) else []
    rows = []
    statuses = []
    items = assessment.get("done_when")
    items = items if isinstance(items, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        status = str(item.get("status") or "")
        statuses.append(status)
        condition = (
            done_when[index]
            if isinstance(index, int) and 0 <= index < len(done_when)
            else "Condition text unavailable"
        )
        rows.append(
            {
                "index": index,
                "done_when": condition,
                "status": status,
                "evidence": item.get("evidence"),
            }
        )
    semantic_complete = result.get("semantic_complete")
    if semantic_complete is None and statuses:
        semantic_complete = all(item == "supported" for item in statuses)
    completion_summary = {
        "semantic_complete": semantic_complete,
        "metric_complete": result.get("metric_complete", "not recorded"),
        "derived_decision": (
            result.get("derived_decision") or decision.get("decision")
        ),
        "fallback": result.get("fallback", False),
        "fallback_reasons": result.get("fallback_reasons", []),
    }
    return (
        "<h4>Completion Assessment</h4>"
        + _record_table(
            rows,
            ("index", "done_when", "status", "evidence"),
        )
        + _key_value_table(completion_summary)
    )


def _phase_event_table(events: tuple[dict[str, Any], ...]) -> str:
    names = {
        "coupled_planner_phase_decided",
        "phase_transition_applied",
        "phase_transition_continued",
        "phase_completion_assessed",
    }
    rows = []
    for item in events:
        event = item.get("event") or item.get("kind")
        if event not in names:
            continue
        payload = _dict(item.get("payload"))
        rows.append(
            {
                "timestamp": item.get("timestamp"),
                "event": event,
                "phase": payload.get("phase"),
                "decision": (
                    payload.get("applied_decision")
                    or payload.get("derived_decision")
                    or payload.get("decision")
                ),
                "source": payload.get("transition_source"),
                "reason": (
                    payload.get("guard_reason") or payload.get("reason")
                ),
            }
        )
    return _record_table(
        rows,
        ("timestamp", "event", "phase", "decision", "source", "reason"),
    )


def _model_facing_context_text(context: dict[str, Any]) -> str:
    """Return only the text JSON sent to the action planner model."""

    if not context:
        return "missing planner context"
    try:
        user_content = build_model_user_content(context)
    except Exception as exc:
        # Review generation must remain available for partial or legacy debug
        # records even when they no longer satisfy the live model-input
        # builder's schema.
        return json.dumps(
            {
                "unavailable_reason": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "saved_context": _redact_binary(context),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    if isinstance(user_content, str):
        return _pretty_json_text(user_content)
    text_parts = [
        str(part.get("text") or "")
        for part in user_content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and str(part.get("text") or "").strip()
    ]
    return (
        "\n\n--- text model input ---\n\n".join(
            _pretty_json_text(value) for value in text_parts
        )
        or "missing text payload"
    )


def _pretty_json_text(value: str) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _compact_progress(value: Any) -> dict[str, Any]:
    progress = _dict(value)
    motion = _dict(progress.get("motion_progress"))
    return {
        "has_translation": motion.get("has_translation"),
        "yaw_only_so_far": motion.get("yaw_only_so_far"),
        "forward_m": motion.get("forward_m"),
        "right_m": motion.get("right_m"),
        "up_m": motion.get("up_m"),
        "planar_m": motion.get("planar_m"),
        "yaw_delta_deg": motion.get("yaw_delta_deg"),
        "tools_executed": motion.get("tools_executed"),
    }


def _compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not plan:
        return {}
    return {
        "action_window": plan.get("action_window"),
        "agent_messages": plan.get("agent_messages"),
        "steps": plan.get("steps"),
    }


def _plan_tools(plan: dict[str, Any]) -> list[str]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return []
    return [
        str(step.get("tool"))
        for step in steps
        if isinstance(step, dict) and step.get("tool")
    ]


def _decision(record: dict[str, Any]) -> dict[str, Any]:
    completion = _dict(record.get("_phase_completion_result"))
    coupled = _dict(record.get("coupled_phase_decision"))
    if completion.get("derived_decision"):
        return {
            **coupled,
            "decision": completion.get("derived_decision"),
            "reason": (
                completion.get("reason")
                or coupled.get("guard_reason")
                or coupled.get("reason")
            ),
        }
    return coupled


def _tool_execution_label(
    decision: dict[str, Any],
    safe_tools: list[str],
) -> str:
    name = decision.get("decision")
    if name in {"advance_to_next_phase", "complete_task"} and not safe_tools:
        return "No UAV tool executed"
    if safe_tools:
        return "Executed " + ", ".join(safe_tools)
    if name == "continue_current_phase":
        return "No UAV tool planned"
    return ""


def _behavior_flags(
    actions: tuple[dict[str, Any], ...],
    records: list[dict[str, Any]],
) -> list[str]:
    flags = []
    yaws = [
        _float(_dict(item.get("args")).get("yaw_deg"))
        for item in actions
        if item.get("tool") == "uav.turn_relative"
    ]
    yaws = [item for item in yaws if item is not None]
    if _has_repeated_large_yaw(yaws):
        flags.append("repeated large yaw")
    if _has_alternating_yaw(yaws, records):
        flags.append("alternating yaw")
    if any(
        _compact_progress(
            _dict(
                record.get("model_context")
                or record.get("planner_context")
            ).get("phase_progress_evidence")
        ).get("yaw_only_so_far")
        for record in records
    ):
        flags.append("yaw-only so far")
    if any(
        _decision(record).get("decision")
        in {"advance_to_next_phase", "complete_task"}
        and not _plan_tools(_dict(record.get("safe_plan")))
        for record in records
    ):
        flags.append("transition without tool")
    return flags


def _has_repeated_large_yaw(yaws: list[float]) -> bool:
    streak_sign = None
    streak = 0
    for yaw in yaws:
        if abs(yaw) < 25:
            streak_sign = None
            streak = 0
            continue
        sign = 1 if yaw > 0 else -1
        streak = streak + 1 if sign == streak_sign else 1
        streak_sign = sign
        if streak >= 3:
            return True
    return False


def _has_alternating_yaw(
    yaws: list[float],
    records: list[dict[str, Any]],
) -> bool:
    for record in records:
        context = _dict(
            record.get("model_context") or record.get("planner_context")
        )
        fragments = context.get("fragments")
        fragments = fragments if isinstance(fragments, list) else []
        for fragment in fragments:
            if not isinstance(fragment, dict) or (
                fragment.get("name") != "action"
            ):
                continue
            content = _dict(fragment.get("content"))
            trend = _dict(content.get("action_trend"))
            pattern = _dict(trend.get("motion_pattern"))
            if pattern.get("type") == "alternating_yaw":
                return True
    signs = [1 if yaw > 0 else -1 for yaw in yaws if abs(yaw) >= 1]
    return len(signs) >= 4 and all(
        signs[index] != signs[index - 1]
        for index in range(1, min(len(signs), 6))
    )


def _record_table(
    records: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    empty_message: str = "No records.",
) -> str:
    if not records or not keys:
        return f'<p class="empty">{_e(empty_message)}</p>'
    head = "".join(f"<th>{_e(key)}</th>" for key in keys)
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{_value_html(record.get(key))}</td>" for key in keys
            )
            + "</tr>"
        )
    return (
        f"<table><thead><tr>{head}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _key_value_table(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return '<p class="empty">No values.</p>'
    return (
        "<table><tbody>"
        + "".join(
            f"<tr><th>{_e(key)}</th><td>{_value_html(item)}</td></tr>"
            for key, item in value.items()
        )
        + "</tbody></table>"
    )


def _json_details(
    records: Any,
    label: str,
    *,
    empty_message: str = "No records.",
) -> str:
    if not records:
        return f'<p class="empty">{_e(empty_message)}</p>'
    payload = json.dumps(
        _redact_binary(records),
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    return (
        f"<details><summary>{_e(label)}</summary><pre>"
        f"{_e(payload)}"
        "</pre></details>"
    )


def _metrics_inline(value: Any) -> str:
    if not isinstance(value, dict):
        return _e(value)
    return " · ".join(
        f"<code>{_e(key)}</code>={_e(item)}"
        for key, item in value.items()
        if not isinstance(item, (dict, list))
    )


def _value_html(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return (
            "<code>"
            + _e(json.dumps(value, ensure_ascii=False, default=str))
            + "</code>"
        )
    return _e(value)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _redact_binary(value: Any) -> Any:
    """Remove inline binary payloads while preserving debug structure."""

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"base64", "image_base64", "image_url"} and (
                isinstance(item, str)
            ):
                result[key] = "<binary payload omitted>"
            else:
                result[key] = _redact_binary(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact_binary(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return "<binary image payload omitted>"
    return value


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _document(*, title: str, style_href: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title>"
        f'<link rel="stylesheet" href="{style_href}"></head>'
        f"<body><main>{body}</main></body></html>"
    )


def _relative_href(parent: Path, target: Path) -> str:
    return Path(
        os.path.relpath(target.resolve(), parent.resolve())
    ).as_posix()


def _url_part(value: str) -> str:
    return value.replace("\\", "_").replace("/", "_")


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


_STYLE = """
:root { color-scheme: light; font-family: Inter, Segoe UI, sans-serif; }
body { margin: 0; background: #f5f7fb; color: #172033; }
main { max-width: 1440px; margin: auto; padding: 28px; }
h1, h2, h3, h4 { color: #102a43; }
section { background: white; border: 1px solid #d9e2ec; border-radius: 12px;
  margin: 18px 0; padding: 18px; box-shadow: 0 3px 12px #102a430d; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border-bottom: 1px solid #e8edf3; padding: 8px; text-align: left;
  vertical-align: top; }
th { color: #486581; }
code, pre { font-family: Consolas, monospace; white-space: pre-wrap;
  overflow-wrap: anywhere; }
.gallery { display: grid;
  grid-template-columns: repeat(auto-fill,minmax(260px,1fr)); gap: 14px; }
.gallery.key { grid-template-columns: repeat(auto-fill,minmax(320px,1fr)); }
.frame-card { margin: 0; border: 1px solid #d9e2ec; border-radius: 9px;
  overflow: hidden; background: #fff; }
.frame-card img { display: block; width: 100%; height: 220px;
  object-fit: contain; background: #111827; }
.frame-card figcaption { padding: 8px; font-size: 12px; color: #52667a; }
.instruction { font-size: 18px; }
.muted, .empty { color: #627d98; }
details { margin-top: 12px; }
details.planner-window { border: 1px solid #d9e2ec; border-radius: 8px;
  padding: 10px 12px; }
details > summary { cursor: pointer; color: #243b53; font-weight: 600; }
a { color: #1367d1; }
"""
