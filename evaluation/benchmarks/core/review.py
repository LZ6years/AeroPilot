"""Render portable benchmark review pages from structured audit data.

The renderer owns simulator-neutral audit views, planner-window diagnostics,
portable links, and shared CSS. Benchmarks contribute metrics and typed review
sections; they never need to parse Runtime memory or inject arbitrary HTML.
"""

from __future__ import annotations

import base64
import binascii
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
from evaluation.benchmarks.core.report_io import atomic_write_text
from src.agent.planning.model_inputs import (
    build_model_user_content,
    select_model_image_inputs,
)


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
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkReviewRenderer:
    """Write shared review CSS plus evaluation and case HTML pages."""

    def write_case(
        self,
        eval_dir: str | Path,
        case: BenchmarkCaseReview,
    ) -> dict[str, str]:
        """Write one finalized case without requiring the batch to finish."""

        root = Path(eval_dir)
        style_path = self._write_style(root)
        case_dir = root / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_html = case_dir / "review.html"
        atomic_write_text(
            case_html,
            self._case_html(case, case_dir, style_path),
        )
        return {
            "case_html": str(case_html),
            "review_css": str(style_path),
        }

    def write_summary(
        self,
        eval_dir: str | Path,
        summary: BenchmarkSummaryReview,
    ) -> dict[str, str]:
        """Write the evaluation index independently from case rendering."""

        root = Path(eval_dir)
        style_path = self._write_style(root)
        index = root / "review.html"
        atomic_write_text(
            index,
            self._summary_html(summary, root, style_path),
        )
        return {
            "html_review": str(index),
            "review_css": str(style_path),
        }

    def write(
        self,
        eval_dir: str | Path,
        summary: BenchmarkSummaryReview,
        cases: tuple[BenchmarkCaseReview, ...],
    ) -> dict[str, str]:
        for case in cases:
            self.write_case(eval_dir, case)
        return self.write_summary(eval_dir, summary)

    @staticmethod
    def _write_style(root: Path) -> Path:
        style_path = root / "review_assets" / "benchmark_review.css"
        atomic_write_text(style_path, _STYLE)
        return style_path

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
        state = ""
        if summary.metadata:
            run_status = str(
                summary.metadata.get("run_status") or "unknown"
            ).lower()
            state = (
                f'<section class="run-state run-state-{_css_token(run_status)}">'
                f"<h2>Evaluation State: {_e(run_status.upper())}</h2>"
                f"{_key_value_table(summary.metadata)}</section>"
            )
        return _document(
            title=f"{summary.benchmark_name} Benchmark Review",
            style_href=_relative_href(root, style_path),
            body=(
                f"<h1>{_e(summary.benchmark_name)} Benchmark Review</h1>"
                '<p class="muted">Protocol: '
                f"<code>{_e(summary.protocol_variant)}</code></p>"
                f"{state}"
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
        planner_input_assets = _materialize_planner_input_assets(
            case.audit,
            case_dir,
        )
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
                "<section><h2>Agent Tool Transcript</h2>"
                f"{_agent_tool_transcript(case.audit, case.frame_assets, case_dir)}</section>"
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
                f"{_planner_review(case.audit, case_dir, planner_input_assets)}</section>"
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


def _agent_tool_transcript(
    audit: BenchmarkAuditLog,
    assets: tuple[ResolvedFrameAsset, ...],
    case_dir: Path,
) -> str:
    """Render the canonical model/tool exchange without inventing reasoning.

    ``public_intent`` is the model's explicit explanation. The report pairs
    calls and results by provider ``tool_call_id`` and keeps physical telemetry
    in the separate Action Timeline below.
    """

    records = list(audit.agent_transcript)
    if not records:
        return (
            '<p class="empty">No agent tool transcript was recorded for this '
            "episode.</p>"
        )

    calls_by_id, results_by_id = _transcript_indexes(records)
    diagnostics = _transcript_diagnostics(calls_by_id, results_by_id)
    diagnostic_html = ""
    if diagnostics:
        diagnostic_html = (
            '<div class="transcript-diagnostics"><strong>Transcript audit '
            "issues</strong><ul>"
            + "".join(f"<li>{_e(item)}</li>" for item in diagnostics)
            + "</ul></div>"
        )

    cards = [
        _transcript_record_card(
            record,
            calls_by_id=calls_by_id,
            results_by_id=results_by_id,
            assets=assets,
            case_dir=case_dir,
        )
        for record in records
    ]
    return diagnostic_html + '<div class="transcript">' + "".join(cards) + "</div>"


def _transcript_indexes(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, list[tuple[int, dict[str, Any]]]],
    dict[str, list[tuple[int, dict[str, Any]]]],
]:
    calls: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    results: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        payload = _dict(record.get("payload"))
        if record.get("kind") == "assistant_turn":
            raw_calls = payload.get("tool_calls")
            if not isinstance(raw_calls, list):
                continue
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict):
                    continue
                call_id = str(raw_call.get("tool_call_id") or "")
                if call_id:
                    calls.setdefault(call_id, []).append((index, raw_call))
        elif record.get("kind") == "tool_result":
            call_id = str(payload.get("tool_call_id") or "")
            if call_id:
                results.setdefault(call_id, []).append((index, payload))
    return calls, results


def _transcript_diagnostics(
    calls_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    results_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
) -> list[str]:
    issues: list[str] = []
    for call_id, calls in calls_by_id.items():
        results = results_by_id.get(call_id, [])
        if len(calls) > 1:
            issues.append(f"duplicate ToolCall id: {call_id}")
        if not results:
            issues.append(f"ToolCall has no terminal ToolResult: {call_id}")
        elif len(results) > 1:
            issues.append(f"ToolCall has multiple ToolResults: {call_id}")
        elif len(calls) == 1:
            call_tool = str(calls[0][1].get("tool_name") or "")
            result_tool = str(results[0][1].get("tool_name") or "")
            if call_tool and result_tool and call_tool != result_tool:
                issues.append(
                    f"ToolCall/ToolResult tool mismatch: {call_id} "
                    f"({call_tool} != {result_tool})"
                )
    for call_id in results_by_id:
        if call_id not in calls_by_id:
            issues.append(f"orphan ToolResult: {call_id}")
    return issues


def _transcript_record_card(
    record: dict[str, Any],
    *,
    calls_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    results_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    assets: tuple[ResolvedFrameAsset, ...],
    case_dir: Path,
) -> str:
    kind = str(record.get("kind") or "unknown")
    payload = _dict(record.get("payload"))
    heading = kind.replace("_", " ").title()
    metadata = " · ".join(
        item
        for item in (
            str(record.get("timestamp") or ""),
            str(record.get("id") or ""),
        )
        if item
    )
    body = _transcript_record_body(
        kind,
        payload,
        calls_by_id=calls_by_id,
        results_by_id=results_by_id,
        assets=assets,
        case_dir=case_dir,
    )
    return (
        f'<article class="transcript-card transcript-{_css_token(kind)}">'
        f"<header><h3>{_e(heading)}</h3>"
        f'<span class="muted">{_e(metadata)}</span></header>'
        f"{body}</article>"
    )


def _transcript_record_body(
    kind: str,
    payload: dict[str, Any],
    *,
    calls_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    results_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    assets: tuple[ResolvedFrameAsset, ...],
    case_dir: Path,
) -> str:
    if kind == "user_instruction":
        return f'<p class="transcript-intent">{_e(payload.get("instruction"))}</p>'
    if kind == "state_checkpoint":
        context = _dict(payload.get("context"))
        phase = _dict(context.get("current_phase"))
        observation_id = str(payload.get("observation_id") or "")
        summary = {
            "observation_id": observation_id or None,
            "current_phase": phase.get("name"),
            "completed": context.get("completed"),
            "stopped": context.get("stopped"),
        }
        checkpoint_assets = tuple(
            asset
            for asset in assets
            if observation_id
            and str(asset.observation_id or "") == observation_id
            and asset.modality == "rgb"
            and asset.display_path
        )
        media = ""
        if checkpoint_assets:
            media = '<div class="gallery transcript-media">' + "".join(
                _frame_card(
                    asset,
                    case_dir,
                    roles=["transcript checkpoint"],
                    observation_source=None,
                    association=None,
                )
                for asset in checkpoint_assets
            ) + "</div>"
        return (
            _key_value_table(summary)
            + media
            + _json_details(context, "Full checkpoint context")
        )
    if kind == "assistant_turn":
        raw_calls = payload.get("tool_calls")
        raw_calls = raw_calls if isinstance(raw_calls, list) else []
        rows = []
        for raw_call in raw_calls:
            call = raw_call if isinstance(raw_call, dict) else {}
            call_id = str(call.get("tool_call_id") or "")
            rows.append(
                {
                    "tool_call_id": call_id,
                    "tool": call.get("tool_name"),
                    "args": call.get("args"),
                    "result_status": _paired_result_status(
                        call_id, calls_by_id, results_by_id
                    ),
                    "pairing": _transcript_pairing(
                        call_id, calls_by_id, results_by_id
                    ),
                }
            )
        intent = (
            '<p class="transcript-intent"><strong>Public intent:</strong> '
            f'{_e(payload.get("public_intent"))}</p>'
        )
        return intent + _record_table(
            rows,
            ("tool_call_id", "tool", "args", "result_status", "pairing"),
            empty_message="This assistant turn did not contain ToolCalls.",
        )
    if kind == "tool_result":
        call_id = str(payload.get("tool_call_id") or "")
        summary = {
            "tool_call_id": call_id or None,
            "tool": payload.get("tool_name"),
            "status": payload.get("status"),
            "message": payload.get("message"),
            "pairing": _transcript_pairing(
                call_id, calls_by_id, results_by_id
            ),
            "requested_args": payload.get("requested_args"),
            "executed_args": payload.get("executed_args"),
            "safety_interventions": payload.get("safety_interventions"),
            "result_ref": payload.get("result_ref"),
        }
        return (
            _key_value_table(summary)
            + _transcript_artifact_refs(payload, assets, case_dir)
            + _json_details(
                payload.get("data"),
                "Result data",
                empty_message="No structured result data.",
            )
        )
    if kind == "protocol_feedback":
        return (
            '<p class="transcript-feedback">'
            f'{_e(payload.get("message"))}</p>'
        )
    return _json_details(payload, "Raw transcript payload")


def _transcript_pairing(
    call_id: str,
    calls_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    results_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
) -> str:
    if not call_id:
        return "invalid: missing tool_call_id"
    call_count = len(calls_by_id.get(call_id, []))
    result_count = len(results_by_id.get(call_id, []))
    if call_count == 0:
        return "invalid: orphan result"
    if call_count > 1:
        return "invalid: duplicate call id"
    if result_count == 0:
        return "open: missing ToolResult"
    if result_count > 1:
        return "invalid: multiple ToolResults"
    call_tool = str(calls_by_id[call_id][0][1].get("tool_name") or "")
    result_tool = str(results_by_id[call_id][0][1].get("tool_name") or "")
    if call_tool and result_tool and call_tool != result_tool:
        return "invalid: tool mismatch"
    return "paired"


def _paired_result_status(
    call_id: str,
    calls_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    results_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
) -> Any:
    if _transcript_pairing(call_id, calls_by_id, results_by_id) != "paired":
        return None
    return results_by_id[call_id][0][1].get("status")


def _transcript_artifact_refs(
    payload: dict[str, Any],
    assets: tuple[ResolvedFrameAsset, ...],
    case_dir: Path,
) -> str:
    refs = payload.get("artifact_refs")
    if not isinstance(refs, list) or not refs:
        return ""
    safe_targets: dict[str, Path] = {}
    for asset in assets:
        for value in (asset.frame_id, asset.display_path, asset.raw_path):
            if value not in (None, ""):
                target = asset.display_path or asset.raw_path
                if target:
                    safe_targets[str(value)] = Path(target)
    rendered = []
    for value in refs:
        ref = str(value)
        target = safe_targets.get(ref)
        if target is not None and target.is_file():
            rendered.append(
                f'<a href="{_relative_href(case_dir, target)}"><code>{_e(ref)}</code></a>'
            )
        else:
            rendered.append(f"<code>{_e(ref)}</code>")
    return "<h4>Artifact references</h4><ul>" + "".join(
        f"<li>{item}</li>" for item in rendered
    ) + "</ul>"


def _materialize_planner_input_assets(
    audit: BenchmarkAuditLog,
    case_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Copy the exact successful-request images into portable review assets.

    Planner contexts retain provider-ready data URLs for debug reproducibility.
    This renderer converts only the image inputs selected for the successful
    attempt into local files. It deliberately does not substitute a similar
    observation frame when a historical record lacks the original input.
    """

    destination = case_dir / "review_assets" / "planner_inputs"
    assets_by_record: dict[str, list[dict[str, Any]]] = {}
    for record in audit.planner_debug:
        record_key = _planner_record_key(record)
        context = _dict(record.get("planner_context"))
        image_mode, attempt = _successful_image_request(record)
        if not context:
            assets_by_record[record_key] = [
                {"unavailable_reason": "planner_context_not_saved"}
            ]
            continue
        try:
            images = select_model_image_inputs(
                context,
                image_mode=image_mode,
            )
        except ValueError as exc:
            assets_by_record[record_key] = [
                {"unavailable_reason": f"image_selection_error:{exc}"}
            ]
            continue
        if not images:
            assets_by_record[record_key] = [
                {"unavailable_reason": "no_valid_model_images"}
            ]
            continue

        assets: list[dict[str, Any]] = []
        for order, image in enumerate(images, start=1):
            payload = _decode_model_image_data_url(image.get("data"))
            input_id = str(image.get("input_id") or f"image_{order}")
            if payload is None:
                assets.append(
                    {
                        "order": order,
                        "input_id": input_id,
                        "role": image.get("role"),
                        "usage": image.get("usage"),
                        "image_mode": image_mode,
                        "attempt": attempt,
                        "unavailable_reason": "invalid_model_image_data_url",
                    }
                )
                continue
            bytes_value, suffix = payload
            safe_input_id = _safe_asset_name(input_id)
            target = destination / (
                f"window_{int(record.get('window_index') or 0):03d}_"
                f"{order:02d}_{safe_input_id}{suffix}"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes_value)
            assets.append(
                {
                    "path": str(target),
                    "order": order,
                    "input_id": input_id,
                    "role": image.get("role"),
                    "usage": image.get("usage"),
                    "image_mode": image_mode,
                    "attempt": attempt,
                }
            )
        assets_by_record[record_key] = assets
    return assets_by_record


def _successful_image_request(record: dict[str, Any]) -> tuple[str, int | None]:
    """Return the image mode from the final successful model request attempt."""

    recovery = _dict(record.get("model_request_recovery"))
    attempts = recovery.get("attempts")
    if isinstance(attempts, list):
        for item in reversed(attempts):
            if not isinstance(item, dict) or item.get("status") != "ok":
                continue
            image_mode = item.get("image_mode")
            if image_mode in {"all", "current_only"}:
                attempt = item.get("attempt")
                return image_mode, attempt if isinstance(attempt, int) else None
    # Legacy planner debug did not persist request recovery. Its historical
    # behavior was the ordinary all-images request mode.
    return "all", None


def _decode_model_image_data_url(value: Any) -> tuple[bytes, str] | None:
    if not isinstance(value, str) or not value.startswith("data:image/"):
        return None
    try:
        header, encoded = value.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header.lower():
        return None
    mime_type = header[5:].split(";", 1)[0].lower()
    suffix = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }.get(mime_type)
    if suffix is None:
        return None
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    return (payload, suffix) if payload else None


def _safe_asset_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    ).strip("_") or "image"


def _planner_record_key(record: dict[str, Any]) -> str:
    return str(
        record.get("path")
        or f"{record.get('phase_key') or record.get('phase')}:{record.get('window_index')}"
    )


def _planner_review(
    audit: BenchmarkAuditLog,
    case_dir: Path,
    planner_input_assets: dict[str, list[dict[str, Any]]],
) -> str:
    records = list(audit.planner_debug)
    if not records:
        return (
            '<p class="empty">Action planner debug was not enabled or no '
            "planner windows were recorded.</p>"
        )

    stats = _planner_stats(records, audit.actions)
    summary = _key_value_table(stats)
    timeline = _planner_timeline(records)
    details = "".join(
        _planner_window(
            record,
            case_dir,
            planner_input_assets.get(_planner_record_key(record), []),
        )
        for record in records
    )
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


def _planner_window(
    record: dict[str, Any],
    case_dir: Path,
    model_input_assets: list[dict[str, Any]],
) -> str:
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
    execution = _dict(record.get("_phase_window_execution"))
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
        + _model_input_images(model_input_assets, case_dir)
        + _steps_comparison(raw_plan, safe_plan, execution)
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


def _model_input_images(
    assets: list[dict[str, Any]],
    case_dir: Path,
) -> str:
    """Render the exact images sent in this window's successful request."""

    if not assets:
        return (
            "<h4>Model Input Images</h4>"
            '<p class="empty">No saved model input image record.</p>'
        )
    available = [item for item in assets if item.get("path")]
    unavailable = [item for item in assets if item.get("unavailable_reason")]
    content = ""
    if available:
        cards = []
        for asset in available:
            role = asset.get("role") or "image"
            label = f"#{asset.get('order')} · {role}"
            metadata = [
                f"input_id={asset.get('input_id')}",
                f"image_mode={asset.get('image_mode')}",
            ]
            if asset.get("attempt") is not None:
                metadata.append(f"attempt={asset.get('attempt')}")
            if asset.get("usage"):
                metadata.append(str(asset["usage"]))
            cards.append(
                {
                    "path": asset["path"],
                    "label": label,
                    "description": " · ".join(metadata),
                }
            )
        content += _image_group(cards, case_dir)
    if unavailable:
        reasons = "; ".join(
            f"#{item.get('order', '?')} {item.get('role', 'image')}: "
            f"{item.get('unavailable_reason')}"
            for item in unavailable
        )
        content += f'<p class="empty">Unavailable: {_e(reasons)}</p>'
    return "<h4>Model Input Images</h4>" + content


def _steps_comparison(
    raw_plan: dict[str, Any],
    safe_plan: dict[str, Any],
    execution: dict[str, Any],
) -> str:
    """Render proposal, safety result, and Runtime outcome by step index."""

    raw_steps = _plan_steps(raw_plan)
    safe_steps = _plan_steps(safe_plan)
    executed_steps = _executed_steps_by_index(execution)
    row_count = max(
        len(raw_steps),
        len(safe_steps),
        (max(executed_steps) + 1) if executed_steps else 0,
    )
    if row_count == 0:
        return (
            "<h4>Steps</h4>"
            '<p class="empty">No tool steps were planned for this window.</p>'
        )
    rows = []
    for index in range(row_count):
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{_step_cell(raw_steps[index] if index < len(raw_steps) else None)}</td>"
            f"<td>{_step_cell(safe_steps[index] if index < len(safe_steps) else None)}</td>"
            f"<td>{_execution_cell(executed_steps.get(index))}</td>"
            "</tr>"
        )
    return (
        "<h4>Steps</h4>"
        '<table class="step-comparison"><thead><tr>'
        "<th>index</th><th>model output</th><th>safety-reviewed</th>"
        "<th>actual execution</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _plan_steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    steps = plan.get("steps")
    return [item for item in steps if isinstance(item, dict)] if isinstance(steps, list) else []


def _executed_steps_by_index(execution: dict[str, Any]) -> dict[int, dict[str, Any]]:
    values = execution.get("executed_steps")
    if not isinstance(values, list):
        return {}
    indexed: dict[int, dict[str, Any]] = {}
    for fallback_index, item in enumerate(values):
        if not isinstance(item, dict):
            continue
        index = item.get("step_index")
        indexed[index if isinstance(index, int) and index >= 0 else fallback_index] = item
    return indexed


def _step_cell(step: dict[str, Any] | None) -> str:
    if not step:
        return '<span class="muted">not present</span>'
    values = [f"<code>{_e(step.get('tool') or step.get('kind') or 'step')}</code>"]
    args = step.get("args")
    if isinstance(args, dict):
        values.append(f"<br><code>{_e(json.dumps(args, ensure_ascii=False))}</code>")
    if step.get("reason"):
        values.append(f"<br>{_e(step['reason'])}")
    modalities = step.get("modalities")
    if isinstance(modalities, list) and modalities:
        values.append(f"<br><span class=\"muted\">modalities: {_e(', '.join(map(str, modalities)))}</span>")
    return "".join(values)


def _execution_cell(executed: dict[str, Any] | None) -> str:
    if not executed:
        return '<span class="muted">execution record unavailable</span>'
    actual_step = _dict(executed.get("step"))
    result = _dict(executed.get("result"))
    ok = result.get("ok")
    if ok is True:
        status = "success"
    elif ok is False:
        status = "failed"
    else:
        status = "unknown"
    values = []
    if actual_step:
        values.append(_step_cell(actual_step) + "<br>")
    values.append(f"<strong>{_e(status)}</strong>")
    if result.get("message"):
        values.append(f"<br>{_e(result['message'])}")
    stabilization = _dict(executed.get("stabilization"))
    if stabilization:
        label = stabilization.get("status") or "recorded"
        reason = stabilization.get("reason")
        values.append(
            "<br><span class=\"muted\">stabilization: "
            + _e(label)
            + (f" ({_e(reason)})" if reason else "")
            + "</span>"
        )
    return "".join(values)


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


def _css_token(value: str) -> str:
    """Keep report-state CSS classes bounded to safe identifier characters."""

    token = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value
    ).strip("-")
    return token or "unknown"


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


_STYLE = """
:root { color-scheme: light; font-family: Inter, Segoe UI, sans-serif; }
body { margin: 0; background: #f5f7fb; color: #172033; }
main { max-width: 1440px; margin: auto; padding: 28px; }
h1, h2, h3, h4 { color: #102a43; }
section { background: white; border: 1px solid #d9e2ec; border-radius: 12px;
  margin: 18px 0; padding: 18px; box-shadow: 0 3px 12px #102a430d; }
.run-state { border-width: 2px; }
.run-state-interrupted, .run-state-failed {
  border-color: #d97706; background: #fffbeb; }
.run-state-running { border-color: #2563eb; background: #eff6ff; }
.run-state-completed { border-color: #059669; background: #ecfdf5; }
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
.transcript { display: grid; gap: 12px; }
.transcript-card { border: 1px solid #d9e2ec; border-left: 5px solid #829ab1;
  border-radius: 8px; padding: 12px 14px; background: #fbfdff; }
.transcript-card header { display: flex; align-items: baseline;
  justify-content: space-between; gap: 16px; }
.transcript-card header h3 { margin: 0 0 8px; }
.transcript-assistant-turn { border-left-color: #2563eb; }
.transcript-tool-result { border-left-color: #059669; }
.transcript-protocol-feedback { border-left-color: #d97706; }
.transcript-state-checkpoint { border-left-color: #7c3aed; }
.transcript-intent, .transcript-feedback { font-size: 15px; }
.transcript-diagnostics { border: 1px solid #d97706; border-radius: 8px;
  background: #fffbeb; color: #92400e; margin-bottom: 12px; padding: 12px; }
.transcript-diagnostics ul { margin-bottom: 0; }
.transcript-media { margin-top: 12px; }
.step-comparison td { min-width: 200px; }
details > summary { cursor: pointer; color: #243b53; font-weight: 600; }
a { color: #1367d1; }
"""
