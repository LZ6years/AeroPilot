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
from evaluation.benchmarks.core.trace import PHYSICAL_WINDOW_EVENTS
from evaluation.benchmarks.core.report_io import atomic_write_text
from src.agent.context.request_assembler import RequestAssembler


@dataclass(frozen=True)
class ReviewSection:
    """One benchmark-owned structured extension section."""

    title: str
    kind: str
    payload: Any
    primary: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"text", "key_value", "table", "image_group", "trajectory_ned"}:
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

    def __init__(self, *, detailed_audit: bool = False) -> None:
        # Full captured requests are opt-in: collapsed JSON still costs bytes.
        self.detailed_audit = detailed_audit

    def write_case(
        self,
        eval_dir: str | Path,
        case: BenchmarkCaseReview,
        *,
        directory_name: str | None = None,
    ) -> dict[str, str]:
        """Write one finalized case without requiring the batch to finish."""

        root = Path(eval_dir)
        style_path = self._write_style(root)
        # Display identity stays authoritative while exports can use short paths.
        name = case.case_id if directory_name is None else directory_name
        if not name or name in {".", ".."} or any(c in name for c in '/\\:'):
            raise ValueError("Case directory must be one relative path component")
        case_dir = root / "cases" / name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_html = case_dir / ("audit_review.html" if self.detailed_audit else "review.html")
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
        if not self.detailed_audit:
            links = ' · '.join(f'<a href="{name}">{name}</a>' for name in
                               ("audit.json", "trace.json", "metrics.json", "case_result.json")
                               if (case_dir / name).is_file())
            detail = case_dir / "audit_review.html"
            if detail.is_file():
                links += ' · <a href="audit_review.html">Detailed audit</a>'
            return _document(
                title=f"{case.benchmark_name}: {case.case_id}",
                style_href=_relative_href(case_dir, style_path),
                body=('<p><a href="../../review.html">← Evaluation index</a></p>'
                      f'<h1>{_e(case.case_id)}</h1><p class="instruction">{_e(case.instruction)}</p>'
                      + _compact_outcome(case)
                      + _review_alerts(case, case_dir)
                      + _uncommitted_results_review(case.audit, case_dir)
                      + ''.join(_section_html(s, case_dir) for s in case.sections if s.primary or s.kind == "trajectory_ned")
                      + _initial_plan_review(case.audit)
                      + _phase_decision_review(case.audit, case.frame_assets, case_dir, planner_input_assets)
                      + f'<footer>Raw audit files: {links}</footer>'),
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
                + _compact_outcome(case)
                + _uncommitted_results_review(case.audit, case_dir)
                + _initial_plan_review(case.audit)
                + _phase_decision_review(case.audit, case.frame_assets, case_dir, planner_input_assets)
                + '<details class="advanced-audit"><summary>Advanced audit · 完整记录与实际请求</summary>'
                "<section><h2>Observation Gallery</h2>"
                f"{gallery}</section>"
                "<section><h2>Agent Tool Transcript</h2>"
                f"{_agent_tool_transcript(case.audit, case.frame_assets, case_dir, planner_input_assets)}</section>"
                "<section><h2>Action Timeline</h2>"
                + _record_table(
                    case.audit.actions,
                    ("id", "timestamp", "tool", "args", "result"),
                )
                + "</section>"
                "<section><h2>Decision / Window Details</h2>"
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
                "</details>"
            ),
        )


def _uncommitted_results_review(audit, case_dir):
    """Show received-but-uncommitted facts without repairing task history."""
    import hashlib
    import shutil

    receipts = audit.uncommitted_tool_completions()
    if not receipts:
        return ""
    cards = []
    run_dir = Path(audit.run_dir).resolve()
    for receipt in receipts:
        images = []
        data = _dict(_dict(receipt.get("result")).get("data"))
        media = data.get("_flyagent_model_inputs", [])
        for item in media if isinstance(media, list) else []:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            source = item.get("path") or item.get("artifact_ref")
            if not source:
                continue
            path = Path(str(source))
            path = (path if path.is_absolute() else run_dir / path).resolve()
            # Capability artifacts belong to this run. Do not export arbitrary
            # paths from a corrupt or externally supplied completion record.
            if not path.is_relative_to(run_dir) or not path.is_file():
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            target = case_dir / "review_assets" / "completions" / (digest + path.suffix.lower())
            if len(str(target.resolve())) > 240:
                raise ValueError("Completion review asset path exceeds 240 characters")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(path, target)
            images.append({"path": str(target), "label": item.get("label", "Tool evidence")})
        cards.append(_json_details(receipt, "Completed result / not committed")
                     + _image_group(images, case_dir))
    return ('<section class="completion-results"><h2>Interrupted batch · 无最终评分</h2>'
            '<p>Results below were saved before ordered submission. They are not '
            'canonical task evidence or a final navigation score.</p>'
            + ''.join(cards) + '</section>')


def _review_alerts(case: BenchmarkCaseReview, case_dir: Path) -> str:
    """Keep failures visible without re-embedding full tracking/debug records."""
    messages = [str(item) for item in case.audit.diagnostics]
    tracking = _dict(case.metrics.get("backend_action_tracking"))
    gate_path = case_dir / "manual_gate.json"
    gate = {}
    if gate_path.is_file():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            tracking = _dict(gate.get("tracking")) or tracking
        except (ValueError, OSError) as exc:
            messages.append(f"Gate artifact unreadable: {exc}")
    if gate.get("passed") is False:
        messages.append("Per-case audit gate FAILED; see manual_gate.json")
    if case.outcome.get("cleanup_errors"):
        messages.append(f"Cleanup errors: {case.outcome['cleanup_errors']}")
    if tracking:
        for key in ("trace_valid", "tracking_valid", "all_observed_within_tolerance"):
            if tracking.get(key) is False:
                messages.append(f"{key}=false; tracking requires review")
    status = _key_value_table({k: tracking.get(k) for k in
        ("trace_valid", "tracking_valid", "all_observed_within_tolerance",
         "max_position_error_m", "max_abs_yaw_error_deg")}) if tracking else ""
    return (('<section class="review-alerts"><h2>Audit / Tracking</h2>' + status
             + ''.join(f'<p class="transcript-feedback">{_e(m)}</p>' for m in messages)
             + '</section>') if messages or tracking else '')


def _compact_outcome(case: BenchmarkCaseReview) -> str:
    """Keep result categories distinct without a screen of repeated key tables."""
    def line(values):
        return ' · '.join(f'<strong>{_e(k)}</strong>: {_e(v)}' for k, v in values.items()
                          if v is not None and not isinstance(v, (list, dict)))
    outcome_keys = ("status", "completed", "stopped", "stopped_reason", "error")
    metric_keys = ("success", "oracle_success", "ndtw", "sdtw", "distance_to_goal")
    notice = ('<p class="transcript-feedback">Scripted offline replay — no Provider request; '
              'navigation success is not assessed. Images are recorded fixtures.</p>'
              if case.metadata.get("scripted") is True else '')
    return (notice + '<section><h2>Outcome / Metrics</h2><p>Agent / Runtime self-report: '
            + line({k: case.outcome.get(k) for k in outcome_keys}) + '</p>'
            + '<p>Benchmark metrics (independent of self-report): '
            + line({k: case.metrics.get(k) for k in metric_keys}) + '</p>'
            + '</section>')


def _initial_plan_review(audit: BenchmarkAuditLog) -> str:
    """Only the pre-execution event is authoritative for the initial full plan."""
    if any(r.get("loop_schema") == "unified_tool_loop_v2" for r in getattr(audit, "planner_debug", ())):
        return ('<section><h2>Optional task plan</h2><p>No fixed Phase plan is required. '
                'Each decision shows the plan available at that time; updates are tool results, '
                'not proof of task completion.</p></section>')
    snapshots = [
        _dict(event.get("payload")) for event in audit.events
        if event.get("event") == "initial_semantic_plan"
    ]
    if len(snapshots) != 1:
        return ('<section><h2>Initial task plan</h2><p class="empty">'
                'Initial plan snapshot unavailable or ambiguous. Historical checkpoints '
                'below are observed phases, not a reconstructed complete initial plan.</p></section>')
    snapshot = snapshots[0]
    plan = _dict(_dict(snapshot.get("parse_result")).get("plan"))
    cards = []
    for phase in plan.get("phases", []):
        phase = _dict(phase)
        cards.append(f'<li><strong>Phase {_e(phase.get("index"))}: {_e(phase.get("name"))}</strong>'
                     f'<p>{_e(phase.get("goal"))}</p>'
                     + _key_value_table({"done_when": phase.get("done_when")}) + '</li>')
    return ('<section><h2>Initial task plan</h2><ol>' + ''.join(cards) + '</ol>'
            + _json_details(snapshot.get("protocol_context"), "Recorded episode interpretation rules",
                            empty_message="No additional episode interpretation rules recorded.")
            + '</section>')


def _phase_decision_review(audit, frame_assets, case_dir, input_assets) -> str:
    """Join primary review by explicit phase/decision/call/observation IDs only.

    The detailed legacy view remains in Advanced audit. Missing IDs are visible,
    never inferred from timestamps, and request images are never regenerated.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    ungrouped = []
    ids = [r.get("decision_id") for r in audit.planner_debug]
    for record in audit.planner_debug:
        unified = record.get("loop_schema") == "unified_tool_loop_v2"
        if (not unified and not record.get("phase_key")) or not record.get("decision_id") or ids.count(record.get("decision_id")) != 1:
            ungrouped.append(record)
            continue
        groups.setdefault("__instruction__" if unified else str(record["phase_key"]), []).append(record)
    _, results = _transcript_indexes(list(audit.agent_transcript))
    heading = "Task execution timeline" if "__instruction__" in groups else "Phase execution timeline"
    parts = [f'<section class="phase-timeline"><h2>{heading}</h2>']
    for phase_key, records in groups.items():
        phase = _dict(_dict(records[0].get("model_context")).get("current_phase"))
        if phase_key == "__instruction__":
            parts.append('<section class="phase-group"><h3>Instruction decisions</h3>')
        else:
            parts.append(f'<section class="phase-group"><h3>Phase {_e(phase_key)}</h3><p>{_e(phase.get("goal"))}</p>')
            parts.append(_json_details(phase.get("done_when"), "Observed completion conditions at phase entry"))
        for record in records:
            parts.append(_compact_decision(record, audit, frame_assets, case_dir,
                                           input_assets.get(_planner_record_key(record), []), results))
        parts.append('</section>')
    if not groups:
        parts.append('<p class="empty">No ID-linked decisions available; see full audit.</p>')
    if ungrouped:
        parts.append(_json_details([{"decision_id": r.get("decision_id"), "phase_key": r.get("phase_key"),
                                    "path": r.get("path")} for r in ungrouped],
                                   "Audit anomaly: ungrouped or duplicate-ID decisions; see audit.json"))
    linked_calls = {str(c.get("tool_call_id")) for r in audit.planner_debug
                    for model_round in r.get("model_rounds", [])
                    for c in _dict(model_round.get("canonical_turn")).get("tool_calls", [])}
    unlinked = sorted(set(results) - linked_calls)
    if unlinked:
        parts.append(_json_details(unlinked, "Audit anomaly: ToolResults without captured decision association"))
    parts.append('</section>')
    return ''.join(parts)


def _compact_decision(record, audit, frame_assets, case_dir, assets, results) -> str:
    """One current frame, short public intent, nested corrections and applied state."""
    decision_id = record["decision_id"]
    obs = record.get("checkpoint_observation_id")
    rounds = [r for r in record.get("model_rounds", []) if isinstance(r, dict)]
    parts = [f'<article class="transcript-card" id="decision-{_e(decision_id)}">'
             f'<h4>Decision {_e(record.get("decision_index"))} · CURRENT {_e(obs)}</h4>']
    current = next((a for a in assets if a.get("observation_id") == obs
                    and a.get("timeline_role") != "TOOL_EVIDENCE" and a.get("path")), None)
    if current:
        parts.append(_image_group([{"path": current["path"], "label": f"Captured CURRENT {obs}",
                                    "description": current.get("image_mode") or "Actual request image"}], case_dir))
    else:
        parts.append(f'<p class="empty">Captured CURRENT image unavailable: {_e(obs)}. No substitute.</p>')
    rgb = [a for a in frame_assets if a.modality == "rgb" and a.observation_id == obs and a.display_path]
    if rgb:
        if not current:
            parts.append(_image_group([{"path": rgb[0].display_path,
                "label": "Sensor RGB — not a captured Provider input",
                "description": f"Recorded observation {obs}; actual request image unavailable."}], case_dir))
        parts.append(f'<p><a href="{_relative_href(case_dir, Path(rgb[0].display_path))}">Original RGB · sensor audit</a></p>')
    context = _dict(record.get("model_context"))
    prior = _dict(context.get("phase_strategy"))
    def stage_label(strategy):
        index = strategy.get("active_stage")
        stages = strategy.get("stages", [])
        if isinstance(index, int) and 0 <= index < len(stages):
            return f"{index}: {stages[index]}"
        return str(index) if strategy else "尚未初始化"
    unified = record.get("loop_schema") == "unified_tool_loop_v2"
    if unified:
        plan = _dict(context.get("task_plan"))
        parts.append('<p>Plan revision at checkpoint: ' + _e(plan.get("revision", "No plan created")) + '</p>')
        if plan:
            parts.append(_record_table(plan.get("steps", []), ("id", "description", "status")))
    else:
        parts.append('<p>Strategy at checkpoint: ' + _e(stage_label(prior)) + '</p>')
    applied = [_dict(e.get("payload")) for e in audit.events if e.get("event") == "phase_strategy_updated"
               and _dict(e.get("payload")).get("decision_id") == decision_id]
    if applied:
        strategy = _dict(_dict(record.get("coupled_plan")).get("phase_strategy"))
        parts.append('<p>Recorded active stage: ' + _e(stage_label(strategy)) + '</p>')
        kind = applied[-1].get("update_kind")
        stages = strategy.get("stages") or []
        active = strategy.get("active_stage", 0)
        active = active if isinstance(active, int) and 0 <= active <= len(stages) else 0
        parts.append(f'<p>Strategy update: {_e(kind)} · version {_e(applied[-1].get("revision"))} (not a replan count)</p>')
        if kind in {"initialize", "replan"}:
            parts.append('<div class="strategy-change">' + _key_value_table({
                "before": prior.get("stages"), "after": stages,
                "reason": strategy.get("reason")}) + '</div>')
        else:
            parts.append('<p class="muted">Completed prefix: '
                         + _e(' → '.join(str(s) for s in stages[:active] if isinstance(active, int))) + '</p>')
    for index, model_round in enumerate(rounds):
        canonical = _dict(model_round.get("canonical_turn"))
        outcome = _dict(model_round.get("runtime_outcome"))
        intent = str(canonical.get("public_intent") or "No public intent captured")
        # Non-final rounds (including corrections) stay inside this decision.
        if index < len(rounds) - 1:
            parts.append(f'<details><summary>Round {_e(model_round.get("round_index"))} · {_e(outcome.get("status"))}</summary>')
        parts.append(f'<p>{_e(intent[:180])}{"…" if len(intent) > 180 else ""}</p>')
        parts.append('<details class="intent-detail"><summary>Full public intent · '
                     + _e(canonical.get("public_intent_source")) + '</summary><p class="intent-text">'
                     + _e(intent) + '</p></details>')
        if outcome.get("feedback"):
            parts.append(_json_details(outcome["feedback"], "Original protocol feedback"))
        for call in canonical.get("tool_calls", []):
            call_id = call.get("tool_call_id")
            matches = results.get(str(call_id), [])
            result = matches[0][1] if len(matches) == 1 else {}
            args = _dict(call.get("args"))
            parts.append(f'<p><code>{_e(call.get("tool_name"))}</code> · {_e(call_id)} · '
                         f'{_e(result.get("status") or "Audit anomaly: missing/duplicate terminal result")}</p>')
            if call.get("tool_name") == "runtime.commit_phase_turn":
                parts.append(_record_table(_dict(args.get("completion_assessment")).get("done_when") or [],
                                           ("index", "status", "evidence")))
                if result.get("status") != "ok" and args.get("phase_strategy"):
                    parts.append(_json_details(args["phase_strategy"], "Proposed strategy · 未应用"))
            else:
                parts.append(_key_value_table(args))
            diagnosis = _dict(_dict(result.get("data")).get("rejection"))
            explanation = {
                "missing_commit": "缺少阶段决策提交；本轮动作未执行。",
                "multiple_commits": "本轮有多个阶段决策提交；动作未执行。",
                "commit_not_first": "阶段决策提交不在首位；本轮动作未执行。",
            }.get(diagnosis.get("code"))
            if explanation:
                parts.append(f'<p class="transcript-feedback">{_e(explanation)}</p>')
            elif result.get("status") == "collision":
                parts.append(
                    '<p class="collision-feedback"><strong>Collision:</strong> '
                    + _e(str(result.get("message") or "")[:240])
                    + "</p>"
                )
            elif result.get("status") in {"error", "rejected"}:
                parts.append(f'<p class="transcript-feedback">{_e(str(result.get("message") or "")[:240])}</p>')
            if result.get("safety_interventions"):
                parts.append('<strong>Safety modified execution</strong>' + _key_value_table({
                    "requested": result.get("requested_args"), "executed": result.get("executed_args"),
                    "interventions": result.get("safety_interventions")}))
            data = _dict(result.get("data"))
            if unified and call.get("tool_name") == "knowledge.load":
                parts.append(_key_value_table({key: data.get(key) for key in
                    ("name", "version", "body_included", "reference_paths")}))
                parts.append(_json_details(data, "Knowledge receipt · complete audit"))
            if unified and call.get("tool_name") == "runtime.update_plan":
                label = "Applied plan update" if result.get("status") == "ok" else "Plan update not applied"
                parts.append(_json_details(data.get("plan") if result.get("status") == "ok" else args, label))
            if "candidates" in data:
                parts.append(_record_table(data.get("candidates") or [],
                                           ("candidate_id", "label", "distance_m", "distance_status")))
            if result.get("message") and result.get("status") == "ok":
                parts.append(f'<p class="muted">{_e(str(result["message"])[:180])}</p>')
        call_ids = {c.get("tool_call_id") for c in canonical.get("tool_calls", [])}
        evidence = [a for a in assets if a.get("timeline_role") == "TOOL_EVIDENCE"
                    and a.get("tool_call_id") in call_ids]
        if evidence:
            parts.append('<details><summary>Captured tool candidate images</summary>'
                         + _model_input_images(evidence, case_dir) + '</details>')
        if index < len(rounds) - 1:
            parts.append('</details>')
    windows = [_dict(e.get("payload")) for e in audit.events
               if (e.get("event") or e.get("kind")) in PHYSICAL_WINDOW_EVENTS
               and _dict(e.get("payload")).get("decision_id") == decision_id]
    for window in windows:
        after = window.get("observation_after_id")
        parts.append(f'<p>After physical batch: {_e(after)}</p>')
        following = [r for r in audit.planner_debug if r.get("checkpoint_observation_id") == after and r.get("decision_id")]
        if following:
            parts.append(f'<a href="#decision-{_e(following[0]["decision_id"])}">Next observation / decision</a>')
    if not unified:
        parts.append(_key_value_table({"phase outcome": _decision(record).get("decision"),
                                       "reason": _decision(record).get("reason")}))
    parts.append('</article>')
    return ''.join(parts)


def _section_html(section: ReviewSection, case_dir: Path) -> str:
    if section.kind == "trajectory_ned":
        from evaluation.benchmarks.core.trajectory_plots import write_ned_comparison
        plots = write_ned_comparison(case_dir, section.payload.get("actual", []), section.payload.get("reference", []))
        if plots["status"] == "ok":
            content = _image_group([{"path": plots["path_2d"], "label": "Actual / reference · 2D · metres"}], case_dir)
            content += '<details><summary>3D trajectory · positive-up display</summary>' + _image_group(
                [{"path": plots["path_3d"], "label": "Actual / reference · 3D · metres"}], case_dir) + '</details>'
            if not plots["has_reference"]:
                content += '<p class="empty">Reference trajectory unavailable.</p>'
        else:
            content = f'<p class="empty">Trajectory unavailable: {_e(plots["reason"])}</p>'
    elif section.kind == "text":
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
    planner_input_assets: dict[str, list[dict[str, Any]]],
) -> str:
    """Render the canonical model/tool exchange without inventing reasoning.

    ``public_intent_source`` distinguishes provider-authored text from the
    deterministic summary reconstructed from ToolCall arguments when a
    provider emits ``content=null``. The report pairs canonical calls and
    results by provider ``tool_call_id`` and keeps physical telemetry in the
    separate Action Timeline below.
    """

    records = list(audit.agent_transcript)
    if not records:
        return (
            '<p class="empty">No agent tool transcript was recorded for this '
            "episode.</p>"
        )

    calls_by_id, results_by_id = _transcript_indexes(records)
    fused_images: dict[str, dict[str, Any]] = {}
    for request_assets in planner_input_assets.values():
        for asset in request_assets:
            if asset.get("timeline_role") == "TOOL_EVIDENCE":
                # Candidate confirmation is not the observation image, even
                # when it refers to the same observation ID.
                continue
            observation_id = str(asset.get("observation_id") or "")
            if observation_id and asset.get("path") and observation_id not in fused_images:
                fused_images[observation_id] = asset
    diagnostics = _transcript_diagnostics(calls_by_id, results_by_id)
    diagnostic_html = ""
    if diagnostics:
        diagnostic_html = (
            '<div class="transcript-diagnostics"><strong>Transcript audit '
            "issues</strong><ul>"
            + "".join(f"<li>{_e(item)}</li>" for item in diagnostics)
            + "</ul></div>"
        )

    last_record_by_observation: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.get("kind") != "state_checkpoint":
            continue
        observation_id = str(_dict(record.get("payload")).get("observation_id") or "")
        if observation_id:
            last_record_by_observation[observation_id] = index
    latest_observation_index = max(last_record_by_observation.values(), default=-1)
    cards = [
        _transcript_record_card(
            record,
            timeline_role=(
                "CURRENT" if index == latest_observation_index else "HISTORY"
            ),
            # Review links may repeat; this does not change provider projection.
            show_checkpoint_media=True,
            calls_by_id=calls_by_id,
            results_by_id=results_by_id,
            assets=assets,
            case_dir=case_dir,
            fused_images=fused_images,
        )
        for index, record in enumerate(records)
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
    timeline_role: str,
    show_checkpoint_media: bool,
    calls_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    results_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    assets: tuple[ResolvedFrameAsset, ...],
    case_dir: Path,
    fused_images: dict[str, dict[str, Any]],
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
        timeline_role=timeline_role,
        show_checkpoint_media=show_checkpoint_media,
        calls_by_id=calls_by_id,
        results_by_id=results_by_id,
        assets=assets,
        case_dir=case_dir,
        fused_images=fused_images,
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
    timeline_role: str,
    show_checkpoint_media: bool,
    calls_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    results_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
    assets: tuple[ResolvedFrameAsset, ...],
    case_dir: Path,
    fused_images: dict[str, dict[str, Any]],
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
        temporal_roles = (
            _checkpoint_temporal_roles(context, observation_id, timeline_role)
            if show_checkpoint_media
            else []
        )
        media = ""
        frame_cards = []
        for temporal in temporal_roles:
            role_observation_id = str(temporal.get("observation_id") or "")
            frame_id = str(temporal.get("frame_id") or "")
            fused = fused_images.get(role_observation_id)
            if fused is not None and Path(str(fused.get("path") or "")).is_file():
                frame_cards.append(
                    _image_group(
                        [
                            {
                                "path": fused["path"],
                                "label": str(temporal.get("label") or "image"),
                                "description": (
                                    f"observation_id={role_observation_id} · "
                                    f"frame_id={frame_id or fused.get('frame_id') or 'unknown'} · "
                                    "actual provider observation image (RGB/depth fusion when available)"
                                ),
                            }
                        ],
                        case_dir,
                    )
                )
            matching = [
                asset
                for asset in assets
                if role_observation_id
                and str(asset.observation_id or "") == role_observation_id
                and asset.modality == "rgb"
                and asset.display_path
                and (not frame_id or asset.frame_id == frame_id)
            ]
            for asset in matching:
                frame_cards.append(
                    _frame_card(
                        asset,
                        case_dir,
                        roles=[str(temporal.get("label") or temporal.get("role")), "Original RGB (not necessarily sent separately)"],
                        observation_source=None,
                        association=_checkpoint_action_association(
                            str(temporal.get("role") or ""),
                            role_observation_id,
                            results_by_id,
                        ),
                    )
                )
            if not matching:
                frame_cards.append(
                    '<p class="empty">Original RGB unavailable in review assets: '
                    f'observation_id={_e(role_observation_id)} frame_id={_e(frame_id)}</p>'
                )
            if fused is None or not Path(str(fused.get("path") or "")).is_file():
                frame_cards.append(
                    '<p class="empty">Exact provider image unavailable for this checkpoint; '
                    f'original RGB above is audit evidence only. observation_id={_e(role_observation_id)}</p>'
                )
        if frame_cards:
            media = (
                '<div class="gallery transcript-media">'
                + "".join(frame_cards)
                + "</div>"
            )
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
        intent_source = str(
            payload.get("public_intent_source") or "provider_content"
        )
        intent_label = (
            "Intent reconstructed from Tool arguments"
            if intent_source == "tool_arguments_fallback"
            else "Provider public intent"
        )
        intent = (
            '<p class="transcript-intent"><strong>'
            f'{_e(intent_label)}:</strong> {_e(payload.get("public_intent"))}</p>'
        )
        return intent + _record_table(
            rows,
            ("tool_call_id", "tool", "args", "result_status", "pairing"),
            empty_message="This assistant turn did not contain ToolCalls.",
        )
    if kind == "provider_turn_rejected":
        raw_calls = payload.get("tool_calls")
        raw_calls = raw_calls if isinstance(raw_calls, list) else []
        rows = []
        for raw_call in raw_calls:
            call = raw_call if isinstance(raw_call, dict) else {}
            rows.append(
                {
                    "tool_call_id": call.get("tool_call_id"),
                    "wire_name": call.get("wire_name"),
                    "arguments": call.get("arguments"),
                    "arguments_status": call.get("arguments_status"),
                }
            )
        summary = {
            "error": payload.get("error"),
            "finish_reason": payload.get("finish_reason"),
            "content_present": payload.get("content_present"),
            "content": payload.get("content"),
        }
        return (
            '<p class="transcript-feedback"><strong>Not executed; audit only.</strong> '
            "This provider response could not be represented as a canonical "
            "AssistantTurn.</p>"
            + _key_value_table(summary)
            + _record_table(
                rows,
                ("tool_call_id", "wire_name", "arguments", "arguments_status"),
                empty_message="The rejected provider response contained no ToolCalls.",
            )
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
        result_data = _dict(payload.get("data"))
        if payload.get("status") == "rejected":
            diagnosis = _dict(result_data.get("rejection"))
            summary["rejection_diagnosis"] = diagnosis or "Structured attribution unavailable in this record."
            summary["execution_disposition"] = {
                "source": "This call caused rejection; not executed.",
                "affected": "Not executed because another call rejected the turn.",
                "turn_member": "Not executed because the turn combination was invalid.",
            }.get(diagnosis.get("role"), "Rejected; see original message.")
        if payload.get("tool_name") == "skill.visual_grounding.measure_visible_depth":
            candidates = result_data.get("candidates")
            candidate_rows = candidates if isinstance(candidates, list) else []
            result_body = _key_value_table(
                {
                    "observation_id": result_data.get("observation_id"),
                    "query": result_data.get("query"),
                    "include_distance": result_data.get("include_distance"),
                }
            ) + _record_table(
                [item for item in candidate_rows if isinstance(item, dict)],
                (
                    "candidate_id",
                    "label",
                    "confidence",
                    "box_normalized",
                    "distance_m",
                    "distance_status",
                ),
                empty_message="No grounded candidates were returned.",
            ) + _json_details(
                result_data.get("audit_details"),
                "Internal grounding and depth audit",
                empty_message="No internal grounding audit.",
            )
            if payload.get("status") == "ok":
                result_body += (
                    '<p class="transcript-feedback"><strong>Manual candidate review required.</strong> '
                    "Detector boxes, masks, and metric depth are auditable, but evaluation "
                    "has not verified that a candidate is the semantic target intended by the instruction.</p>"
                )
        elif payload.get("tool_name") == "skill.metric_depth.sample_region":
            result_body = _depth_roi_preview(
                payload,
                result_data,
                fused_images,
                assets,
                case_dir,
            ) + _key_value_table(
                {
                    "observation_id": result_data.get("observation_id"),
                    "distance_m": result_data.get("distance_m"),
                }
            ) + _json_details(
                result_data.get("audit_details"),
                "Internal depth calculation audit",
                empty_message="No internal depth calculation audit.",
            )
            if payload.get("status") == "ok":
                result_body += (
                    '<p class="transcript-feedback"><strong>Manual ROI review required.</strong> '
                    "The metric depth is valid for the submitted pixels, but "
                    "evaluation has not verified that the ROI belongs to the "
                    "semantic target intended by the model.</p>"
                )
        else:
            result_body = _json_details(
                result_data,
                "Result data",
                empty_message="No structured result data.",
            )
        return (
            _key_value_table(summary)
            + _transcript_artifact_refs(payload, assets, case_dir)
            + result_body
        )
    if kind == "protocol_feedback":
        return (
            '<p class="transcript-feedback">'
            f'{_e(payload.get("message"))}</p>'
        )
    if kind == "task_terminal":
        return (
            '<p class="transcript-feedback"><strong>Instruction stopped incomplete.</strong> '
            "Runtime closed the declared blocker and retained the scene for the next case.</p>"
            + _key_value_table(
                {
                    "completed": payload.get("completed"),
                    "stopped": payload.get("stopped"),
                    "stopped_reason": payload.get("stopped_reason"),
                    "stop_source": payload.get("stop_source"),
                    "blocking_condition": payload.get("blocking_condition"),
                    "evidence_refs": payload.get("evidence_refs"),
                    "suggested_user_action": payload.get("suggested_user_action"),
                    "observation_after_id": payload.get("observation_after_id"),
                }
            )
            + _json_details(payload, "Complete terminal record")
        )
    return _json_details(payload, "Raw transcript payload")


def _depth_roi_preview(
    payload: dict[str, Any],
    result_data: dict[str, Any],
    fused_images: dict[str, dict[str, Any]],
    assets: tuple[ResolvedFrameAsset, ...],
    case_dir: Path,
) -> str:
    """Overlay the requested display-normalized ROI on the safe fused image."""

    args = _dict(payload.get("requested_args"))
    observation_id = str(
        result_data.get("observation_id") or args.get("observation_id") or ""
    )
    asset = fused_images.get(observation_id)
    if isinstance(asset, dict):
        path = Path(str(asset.get("path") or ""))
    else:
        fallback = next(
            (
                item
                for item in assets
                if item.modality == "rgb"
                and str(item.observation_id or "") == observation_id
                and item.display_path
            ),
            None,
        )
        path = Path(fallback.display_path) if fallback is not None else Path()
    if not path.is_file():
        return ""
    try:
        x_min = float(args["x_min"])
        y_min = float(args["y_min"])
        x_max = float(args["x_max"])
        y_max = float(args["y_max"])
    except (KeyError, TypeError, ValueError):
        return ""
    if not (
        0.0 <= x_min < x_max <= 1.0
        and 0.0 <= y_min < y_max <= 1.0
    ):
        return ""
    style = (
        f"left:{x_min * 100:.4f}%;top:{y_min * 100:.4f}%;"
        f"width:{(x_max - x_min) * 100:.4f}%;"
        f"height:{(y_max - y_min) * 100:.4f}%"
    )
    href = _relative_href(case_dir, path)
    return (
        '<figure class="frame-card depth-roi-preview">'
        '<div class="depth-roi-image">'
        f'<img loading="lazy" src="{href}" alt="{_e(observation_id)} depth ROI">'
        f'<span class="depth-roi-box" style="{style}"></span>'
        "</div>"
        "<figcaption><strong>Requested metric-depth ROI</strong><br>"
        f"<span>{_e(observation_id)} · normalized against displayed image</span>"
        "</figcaption></figure>"
    )


def _checkpoint_temporal_roles(
    context: dict[str, Any], current_observation_id: str, timeline_role: str
) -> list[dict[str, Any]]:
    """Read this checkpoint's sole chronological image reference."""

    frame_id = None
    fragments = context.get("fragments")
    if isinstance(fragments, list):
        for fragment in fragments:
            fragment = _dict(fragment)
            if fragment.get("name") != "image":
                continue
            selected = _dict(fragment.get("content")).get("selected_frames")
            if not isinstance(selected, list):
                break
            for item in selected:
                item = _dict(item)
                if str(item.get("observation_id") or "") == current_observation_id:
                    frame_id = item.get("frame_id")
                    break
            break
    return [
        {
            "role": "current_observation",
            "label": timeline_role,
            "observation_id": current_observation_id,
            "frame_id": frame_id,
        }
    ]


def _checkpoint_action_association(
    role: str,
    observation_id: str,
    results_by_id: dict[str, list[tuple[int, dict[str, Any]]]],
) -> str | None:
    if not observation_id:
        return None
    field = "observation_after_id" if role == "current_observation" else None
    if field is None:
        return None
    for call_id, results in results_by_id.items():
        for _, result in results:
            data = _dict(result.get("data"))
            if str(data.get(field) or "") == observation_id:
                prefix = "before" if field == "observation_before_id" else "after"
                return f"{prefix} committed action window {call_id}"
    return None


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
        model_rounds = record.get("model_rounds")
        if isinstance(model_rounds, list):
            assets_by_record[record_key] = _materialize_provider_round_assets(
                audit,
                case_dir,
                record,
                model_rounds,
            )
            continue
        # V2 review never regenerates media from a checkpoint. Exact provider
        # occurrences come only from the captured request-media audit above.
        assets_by_record[record_key] = [
            {"unavailable_reason": "provider_request_media_not_captured"}
        ]
    return assets_by_record


def _materialize_provider_round_assets(
    audit: BenchmarkAuditLog,
    case_dir: Path,
    record: dict[str, Any],
    model_rounds: list[Any],
) -> list[dict[str, Any]]:
    """Copy content-addressed request media while preserving occurrence order."""

    run_root = Path(audit.run_dir).resolve()
    destination = case_dir / "review_assets" / "provider_inputs"
    assets: list[dict[str, Any]] = []
    for round_value in model_rounds:
        if not isinstance(round_value, dict):
            continue
        round_index = round_value.get("round_index")
        media = round_value.get("request_media")
        if not isinstance(media, list):
            continue
        for order, value in enumerate(media, start=1):
            item = value if isinstance(value, dict) else {}
            asset = {
                "round_index": round_index,
                "order": order,
                "sha256": item.get("sha256"),
                "mime_type": item.get("mime_type"),
                "request_path": item.get("request_path"),
                "timeline_role": item.get("timeline_role"),
                "observation_id": item.get("observation_id"),
                "frame_id": item.get("frame_id"),
                "tool_call_id": item.get("tool_call_id"),
                "scene_size": item.get("scene_size"),
                "display_size": item.get("display_size"),
                "scene_bounds_in_display_pixels": item.get(
                    "scene_bounds_in_display_pixels"
                ),
                "label_text": item.get("label"),
            }
            ref = item.get("asset_ref")
            if not isinstance(ref, str) or not ref:
                asset["unavailable_reason"] = (
                    item.get("unavailable_reason") or "provider_media_asset_not_saved"
                )
                assets.append(asset)
                continue
            source = (run_root / ref).resolve()
            if run_root != source and run_root not in source.parents:
                asset["unavailable_reason"] = "provider_media_ref_outside_run_dir"
                assets.append(asset)
                continue
            if not source.is_file():
                asset["unavailable_reason"] = "provider_media_asset_missing"
                assets.append(asset)
                continue
            suffix = source.suffix.lower() or ".bin"
            digest = _safe_asset_name(str(item.get("sha256") or "image"))
            target = destination / f"{digest}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                target.write_bytes(source.read_bytes())
            asset["path"] = str(target)
            assets.append(asset)
    return assets


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
    has_provider_exchange = any(
        record.get("schema_version") == "agent_decision_debug_v2"
        and isinstance(record.get("model_rounds"), list)
        for record in records
    )
    has_checkpoint_transcript = any(
        item.get("kind") == "state_checkpoint" for item in audit.agent_transcript
    )
    has_model_transcript = any(
        item.get("kind") in {"assistant_turn", "provider_turn_rejected"}
        for item in audit.agent_transcript
    )
    if (
        not has_provider_exchange
        and has_checkpoint_transcript
        and has_model_transcript
    ):
        return _legacy_decision_review(
            audit,
            case_dir,
            records,
            planner_input_assets,
        )
    if not records:
        return (
            '<p class="empty">Action planner debug was not enabled or no '
            "planner windows were recorded.</p>"
        )

    stats = _planner_stats(records, audit.actions)
    summary = _key_value_table(stats)
    timeline = _planner_timeline(records)
    details = "".join(
        (
            _decision_window(
                record,
                case_dir,
                planner_input_assets.get(_planner_record_key(record), []),
                strategy_events=[
                    _dict(event.get("payload")) for event in audit.events
                    if event.get("event") == "phase_strategy_updated"
                    and _dict(event.get("payload")).get("decision_id") == record.get("decision_id")
                ] if record.get("decision_id") else [],
            )
            if record.get("schema_version") == "agent_decision_debug_v2"
            else _planner_window(
                record,
                case_dir,
                planner_input_assets.get(_planner_record_key(record), []),
            )
        )
        for record in records
    )
    return (
        "<h3>Decision Summary</h3>"
        + summary
        + "<h3>Decision Timeline</h3>"
        + timeline
        + "<h3>Decision / Window Details</h3>"
        + details
    )


def _decision_window(
    record: dict[str, Any],
    case_dir: Path,
    model_input_assets: list[dict[str, Any]],
    *,
    strategy_events: list[dict[str, Any]] | None = None,
) -> str:
    """Render one v2 state decision and all provider rounds within it."""

    model_context = _dict(record.get("model_context"))
    current_phase = _dict(model_context.get("current_phase"))
    decision = _decision(record)
    raw_plan = _dict(record.get("raw_plan"))
    safe_plan = _dict(record.get("safe_plan"))
    execution = _dict(record.get("_phase_window_execution"))
    rounds = [item for item in record.get("model_rounds", []) if isinstance(item, dict)]
    physical = record.get("physical_window_index")
    physical_label = (
        f"physical window {physical}"
        if isinstance(physical, int)
        else "no physical window"
    )
    transition = decision.get("decision") or "provider_failure"
    title = (
        f"Decision {record.get('decision_index')} · "
        f"{record.get('checkpoint_observation_id') or 'unknown observation'} · "
        f"{len(rounds)} model round{'s' if len(rounds) != 1 else ''} · "
        f"{transition} · {physical_label}"
    )
    summary = {
        "decision_id": record.get("decision_id"),
        "checkpoint_observation_id": record.get("checkpoint_observation_id"),
        "phase": current_phase.get("name") or record.get("phase"),
        "completed_window_count_at_checkpoint": record.get("completed_window_count"),
        "physical_window_index": physical,
        "transition": transition,
        "actual_execution_tools": _execution_tools(execution),
    }
    round_html = "".join(
        _provider_round(
            item,
            case_dir,
            [
                asset
                for asset in model_input_assets
                if asset.get("round_index") == item.get("round_index")
            ],
            applied_strategy=(
                {"events": strategy_events, "snapshot": _dict(record.get("coupled_plan")).get("phase_strategy")}
                if strategy_events and _dict(item.get("runtime_outcome")).get("status") == "commit_accepted"
                else None
            ),
        )
        for item in rounds
    )
    if not rounds:
        round_html = (
            '<p class="empty">Exact provider exchange was not captured for this decision.</p>'
        )
    return (
        '<details class="planner-window"><summary>'
        + _e(title)
        + "</summary>"
        + _key_value_table(summary)
        + "<h4>Provider Model Rounds</h4>"
        + round_html
        + _steps_comparison(raw_plan, safe_plan, execution)
        + "<h4>Derived Runtime Artifacts</h4>"
        + _json_details(
            _compact_plan(raw_plan),
            "Runtime plan reconstructed from accepted ToolCalls",
        )
        + _json_details(_compact_plan(safe_plan), "Safety-reviewed execution plan")
        + _json_details(record.get("safety_decision"), "Safety decision")
        + _json_details(record, "Complete decision debug record")
        + "</details>"
    )


def _provider_round(
    model_round: dict[str, Any],
    case_dir: Path,
    assets: list[dict[str, Any]],
    *,
    applied_strategy: dict[str, Any] | None = None,
) -> str:
    """Show the public response and Runtime handling before folded raw data."""

    response = _dict(model_round.get("response"))
    message = _dict(response.get("message"))
    outcome = _dict(model_round.get("runtime_outcome"))
    request = _dict(model_round.get("request"))
    canonical = _dict(model_round.get("canonical_turn"))
    content = message.get("content") if "content" in message else None
    visible = {
        "finish_reason": response.get("finish_reason"),
        "content": content,
        "runtime_status": outcome.get("status"),
        "runtime_feedback": outcome.get("feedback"),
        "public_intent": canonical.get("public_intent"),
        "public_intent_source": canonical.get("public_intent_source"),
    }
    tools = request.get("tools")
    request_messages = request.get("messages")
    if not request:
        availability = (
            '<p class="empty">Exact provider request unavailable for this model round.</p>'
        )
    else:
        availability = _model_input_images(assets, case_dir)
    return (
        '<details class="model-round" open><summary>'
        f"Model round {_e(model_round.get('round_index'))} · "
        f"{_e(outcome.get('status') or 'unknown outcome')}</summary>"
        + availability
        + _key_value_table(visible)
        + _round_strategy_review(request_messages, canonical, outcome, applied_strategy)
        + "<h5>Raw provider ToolCalls</h5>"
        + _json_details(message.get("tool_calls"), "Wire ToolCalls")
        + "<h5>Runtime handling</h5>"
        + _json_details(outcome, "Protocol feedback and ToolResults")
        + _json_details(canonical, "Canonical parsed turn (may be rejected by Runtime)")
        + "<h5>Actual Provider Request</h5>"
        + history_projection_review(model_round.get("history_projection"))
        + _json_details(request_messages, "Complete request messages")
        + _readable_request_messages(request_messages)
        + _json_details(tools, "Exposed tools schema")
        + _json_details(response, "Public provider response")
        + "</details>"
    )


def history_projection_review(audit: Any, *, preview: bool = False) -> str:
    """Keep simulated selections separate from captured provider input facts."""
    if not isinstance(audit, dict) or not audit:
        return ""
    title = "Offline sampling preview - NOT sent to model" if preview else "Captured request image selection"
    return (
        "<h5>" + _e(title) + "</h5>"
        + _key_value_table({
            "observations in history": audit.get("observation_count"),
            "observation images attached": audit.get("observation_image_count"),
            "additional tool images": audit.get("tool_image_count"),
        })
        + _json_details(audit, "Selection policy, observation IDs and missing images")
    )


def _request_contexts(messages: Any) -> list[dict[str, Any]]:
    """Read only context JSON actually present in user messages, never future state."""
    contexts = []
    for message in messages if isinstance(messages, list) else []:
        message = _dict(message)
        if message.get("role") != "user":
            continue
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks:
            block = _dict(block)
            if block.get("type") != "text":
                continue
            parsed = _parsed_json_object(block.get("text"))
            if parsed is not None and ("current_phase" in parsed or "fragments" in parsed):
                contexts.append(parsed)
    return contexts


def _parsed_json_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _round_strategy_review(messages, canonical, outcome, applied) -> str:
    """Separate prior input, proposed strategy, and event-confirmed application."""
    contexts = _request_contexts(messages)
    prior = contexts[-1].get("phase_strategy") if contexts else None
    submitted = [
        {"tool_call_id": call.get("tool_call_id"), "phase_strategy": _dict(call.get("args")).get("phase_strategy")}
        for item in canonical.get("tool_calls", [])
        if (call := _dict(item)).get("tool_name") == "runtime.commit_phase_turn"
    ]
    return (
        "<h5>Phase strategy: input / submission / application</h5>"
        + _json_details(prior, "Strategy in latest request checkpoint", empty_message="No strategy in latest request checkpoint; no future state substituted.")
        + _json_details(submitted, "Strategy submitted this round", empty_message="No canonical commit strategy submitted.")
        + _key_value_table({"runtime_status": outcome.get("status"), "application": "Recorded strategy update (not proof of successful motion)" if applied else "No strategy application event for this round"})
        + _json_details(applied, "Recorded applied strategy and change", empty_message="No recorded application; a submitted strategy is not an applied strategy.")
    )


def _readable_request_messages(messages: Any) -> str:
    """Unpack known JSON text and arguments for review without changing requests."""
    cards = []
    for index, raw in enumerate(messages if isinstance(messages, list) else []):
        message = _dict(raw)
        parts = []
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks:
            block = _dict(block)
            if block.get("type") == "text":
                text = block.get("text")
                parsed = _parsed_json_object(text)
                if parsed is not None and ("current_phase" in parsed or "fragments" in parsed or message.get("role") == "tool"):
                    parts.append(_json_details(parsed, "Decoded message JSON"))
                else:
                    parts.append(f"<pre>{_e(text)}</pre>")
            else:
                parts.append('<p>Media block: see the exact provider image gallery and raw request reference.</p>')
        for raw_call in message.get("tool_calls", []):
            call = _dict(raw_call)
            function = _dict(call.get("function"))
            parsed = _parsed_json_object(function.get("arguments"))
            parts.append(_json_details({"id": call.get("id"), "name": function.get("name"), "arguments": parsed if parsed is not None else function.get("arguments")}, "ToolCall arguments"))
        cards.append(f'<details><summary>Message {index} · {_e(message.get("role"))} · {_e(message.get("tool_call_id") or "")}</summary>' + "".join(parts) + "</details>")
    return '<details><summary>Readable request messages (same captured request)</summary>' + "".join(cards) + "</details>"


def _execution_tools(execution: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    steps = execution.get("executed_steps")
    if not isinstance(steps, list):
        return tools
    for item in steps:
        step = _dict(_dict(item).get("step"))
        if step.get("tool"):
            tools.append(str(step["tool"]))
    return tools


def _legacy_decision_review(
    audit: BenchmarkAuditLog,
    case_dir: Path,
    planner_records: list[dict[str, Any]],
    planner_input_assets: dict[str, list[dict[str, Any]]],
) -> str:
    """Reconstruct state decisions from canonical transcript boundaries.

    Historical runs did not save the exact provider payload. The canonical
    transcript can still prove chronology, rejection feedback, ToolResults and
    the distinction between terminal decisions and physical windows. It must
    never be relabelled as an exact request snapshot.
    """

    cycles: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for item in audit.agent_transcript:
        kind = item.get("kind")
        if kind == "state_checkpoint":
            if active is not None:
                cycles.append(active)
            payload = _dict(item.get("payload"))
            active = {
                "observation_id": payload.get("observation_id"),
                "context": _dict(payload.get("context")),
                "items": [],
            }
            continue
        if active is not None:
            active["items"].append(dict(item))
    if active is not None:
        cycles.append(active)

    all_assets = [
        asset
        for values in planner_input_assets.values()
        for asset in values
    ]
    physical_index = 0
    cards: list[str] = []
    for decision_index, cycle in enumerate(cycles, start=1):
        model_items = [
            item
            for item in cycle["items"]
            if item.get("kind") in {"assistant_turn", "provider_turn_rejected"}
        ]
        accepted = [
            _dict(item.get("payload"))
            for item in model_items
            if item.get("kind") == "assistant_turn"
        ]
        final_turn = accepted[-1] if accepted else {}
        calls = final_turn.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        physical_calls = [
            call
            for call in calls
            if str(_dict(call).get("tool_name") or "").startswith("uav.")
        ]
        physical = None
        if physical_calls:
            physical_index += 1
            physical = physical_index
        transition = _legacy_transition_from_calls(calls)
        observation_id = cycle.get("observation_id")
        assets = [
            asset
            for asset in all_assets
            if asset.get("observation_id") == observation_id
            and "current" in str(asset.get("role") or "")
        ]
        cards.append(
            _legacy_decision_card(
                decision_index,
                cycle,
                model_items,
                transition,
                physical,
                assets,
                case_dir,
            )
        )

    return (
        '<p class="empty">Exact provider request unavailable: run predates '
        "provider exchange capture. Canonical transcript records are shown "
        "without reconstructing missing system messages, tool schemas, or images.</p>"
        + "<h3>Decision Summary</h3>"
        + _key_value_table(
            {
                "decision_count": len(cycles),
                "physical_window_count": physical_index,
                "capture_mode": "legacy canonical transcript",
            }
        )
        + "<h3>Decision / Window Details</h3>"
        + "".join(cards)
    )


def _legacy_transition_from_calls(calls: list[Any]) -> str:
    commit = next(
        (
            _dict(call)
            for call in calls
            if _dict(call).get("tool_name") == "runtime.commit_phase_turn"
        ),
        {},
    )
    args = _dict(commit.get("args"))
    assessment = _dict(args.get("completion_assessment"))
    done_when = assessment.get("done_when")
    if isinstance(done_when, list) and done_when and all(
        _dict(item).get("status") == "supported" for item in done_when
    ):
        return "complete_task_or_advance_phase"
    return "continue_current_phase" if commit else "no_commit"


def _legacy_decision_card(
    decision_index: int,
    cycle: dict[str, Any],
    model_items: list[dict[str, Any]],
    transition: str,
    physical_window_index: int | None,
    assets: list[dict[str, Any]],
    case_dir: Path,
) -> str:
    observation_id = cycle.get("observation_id") or "unknown observation"
    physical = (
        f"physical window {physical_window_index}"
        if physical_window_index is not None
        else "no physical window"
    )
    round_blocks = []
    items = cycle.get("items") if isinstance(cycle.get("items"), list) else []
    for round_index, model_item in enumerate(model_items, start=1):
        try:
            start = items.index(model_item)
        except ValueError:
            start = 0
        later_model_positions = [
            index
            for index in range(start + 1, len(items))
            if items[index].get("kind") in {"assistant_turn", "provider_turn_rejected"}
        ]
        end = later_model_positions[0] if later_model_positions else len(items)
        handling = items[start + 1 : end]
        payload = _dict(model_item.get("payload"))
        round_blocks.append(
            '<details class="model-round" open><summary>'
            f"Model round {round_index} · {_e(model_item.get('kind'))}</summary>"
            + _json_details(payload, "Canonical model record")
            + _json_details(handling, "Canonical ToolResults / protocol feedback")
            + "</details>"
        )
    image_html = _model_input_images(assets, case_dir) if assets else (
        "<h4>Model Input Images</h4>"
        '<p class="empty">Exact image unavailable for this checkpoint; no substitute was used.</p>'
    )
    return (
        '<details class="planner-window"><summary>'
        f"Decision {decision_index} · {_e(observation_id)} · "
        f"{len(model_items)} model round{'s' if len(model_items) != 1 else ''} · "
        f"{_e(transition)} · {_e(physical)}"
        "</summary>"
        + image_html
        + "".join(round_blocks)
        + _json_details(cycle.get("context"), "Saved checkpoint context (not exact provider request)")
        + "</details>"
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
        "decision_count": len(records),
        "physical_window_count": sum(
            1
            for record in records
            if isinstance(record.get("physical_window_index"), int)
            or (
                record.get("schema_version") != "agent_decision_debug_v2"
                and isinstance(record.get("window_index"), int)
            )
        ),
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
        + "<h4>Saved Planner Context (legacy; not an exact provider request)</h4><pre>"
        + _e(_model_facing_context_text(model_context))
        + "</pre><h4>Action Plan</h4>"
        + _json_details(
            _compact_plan(raw_plan),
            "Runtime plan reconstructed from accepted ToolCalls",
        )
        + _json_details(_compact_plan(safe_plan), "Safety-reviewed execution plan")
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
            role = (
                asset.get("timeline_role")
                or asset.get("role")
                or "provider request image"
            )
            label = f"#{asset.get('order')} · {role}"
            metadata = []
            if asset.get("input_id"):
                metadata.append(f"input_id={asset.get('input_id')}")
            if asset.get("image_mode"):
                metadata.append(f"image_mode={asset.get('image_mode')}")
            if asset.get("sha256"):
                metadata.append(f"sha256={asset.get('sha256')}")
            if asset.get("request_path"):
                metadata.append(
                    "request_path="
                    + json.dumps(asset.get("request_path"), ensure_ascii=False)
                )
            if asset.get("observation_id"):
                metadata.append(f"observation_id={asset.get('observation_id')}")
            if asset.get("frame_id"):
                metadata.append(f"frame_id={asset.get('frame_id')}")
            if asset.get("scene_size"):
                metadata.append(f"scene_size={asset.get('scene_size')}")
            if asset.get("display_size"):
                metadata.append(f"display_size={asset.get('display_size')}")
            if asset.get("scene_bounds_in_display_pixels"):
                metadata.append(
                    "scene_bounds="
                    f"{asset.get('scene_bounds_in_display_pixels')}"
                )
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
        "<th>index</th><th>accepted ToolCall</th><th>safety-reviewed call</th>"
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
    execution_state = _dict(result.get("execution_state"))
    tracking = _dict(executed.get("tracking"))
    if execution_state.get("outcome") == "collision" or tracking.get("status") == "collision":
        status = "collision"
    elif result.get("status") == "skipped":
        status = "skipped"
    elif ok is True:
        status = "success"
    elif ok is False:
        status = "failed"
    else:
        status = "unknown"
    values = []
    if actual_step:
        values.append(_step_cell(actual_step) + "<br>")
    status_class = "collision-label" if status == "collision" else ""
    values.append(f'<strong class="{status_class}">{_e(status)}</strong>')
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
                "decision_id": payload.get("decision_id"),
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
        (
            "decision_id",
            "timestamp",
            "event",
            "phase",
            "decision",
            "source",
            "reason",
        ),
    )


def _model_facing_context_text(context: dict[str, Any]) -> str:
    """Return only the text JSON sent to the action planner model."""

    if not context:
        return "missing planner context"
    try:
        user_content = RequestAssembler().checkpoint_content(
            context, model_inputs=[], timeline_role="CURRENT"
        )
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
.phase-group { border-left: 5px solid #346a9a; padding: 16px; margin: 28px 0; background: #f2f6fa; }
.phase-group > h3 { position: sticky; top: 0; background: #e5edf5; padding: 12px; z-index: 1; }
.phase-group article { margin: 18px 0 18px 12px; background: white; padding: 16px; }
.phase-group details { border-left: 3px solid #c5d3e0; padding: 8px 12px; background: #f7f9fc; }
.phase-group details[open] { border-left-color: #567b9c; }
summary { cursor: pointer; font-weight: 600; }
.intent-text { white-space: pre-wrap; }
.strategy-change { background: #fff6dd; padding: 10px; }
footer { padding: 20px; border-top: 1px solid #ccd; }
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
.depth-roi-preview { max-width: 360px; margin: 12px 0; }
.depth-roi-image { position: relative; background: #111827; }
.depth-roi-image img { display: block; width: 100%; height: auto; }
.depth-roi-box { position: absolute; box-sizing: border-box; border: 2px solid #ff2d55;
  background: #ff2d5526; box-shadow: 0 0 0 1px #fff; pointer-events: none; }
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
.transcript-task-terminal { border-left-color: #b91c1c; background: #fff7f7; }
.transcript-intent, .transcript-feedback { font-size: 15px; }
.collision-feedback { font-size: 15px; border-left: 4px solid #d97706;
  background: #fffbeb; color: #92400e; padding: 8px 10px; }
.collision-label { display: inline-block; border-radius: 999px; padding: 2px 8px;
  background: #fef3c7; color: #92400e; }
.transcript-diagnostics { border: 1px solid #d97706; border-radius: 8px;
  background: #fffbeb; color: #92400e; margin-bottom: 12px; padding: 12px; }
.transcript-diagnostics ul { margin-bottom: 0; }
.transcript-media { margin-top: 12px; }
.step-comparison td { min-width: 200px; }
details > summary { cursor: pointer; color: #243b53; font-weight: 600; }
a { color: #1367d1; }
"""
