"""Reproducibility manifests and offline campaign aggregation for UAV-Flow.

The live runner owns writing one small, redacted manifest before a scene is
started.  Campaign aggregation is deliberately offline: it reads finalized
evaluation artifacts, never imports a simulator backend, and never calls a
model provider.  This separation lets an interrupted 273-case campaign resume
only genuinely missing cases without rewriting evidence from earlier segments.
"""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from evaluation.benchmarks.core.report_io import atomic_write_text
from evaluation.benchmarks.uav_flow.cases import (
    UavFlowCase,
    load_cases,
    load_classification,
)
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.metrics import summarize_metrics
from evaluation.benchmarks.uav_flow.reports import failure_counts


RUN_MANIFEST_SCHEMA = "uav_flow_run_manifest_v1"
CAMPAIGN_AUDIT_SCHEMA = "uav_flow_campaign_audit_v1"
_CONFIG_SELECTOR_FIELDS = {"case_ids", "classes", "limit_per_class", "max_cases"}
_CONFIG_LOCATION_FIELDS = {"output_dir", "run_root", "env_path"}


def write_run_manifest(
    eval_dir: str | Path,
    *,
    config: UavFlowEvalConfig,
    model_config: Any,
    cases: Iterable[UavFlowCase],
) -> str:
    """Freeze the effective, secret-free inputs for one evaluation segment."""

    root = Path(eval_dir)
    selected = tuple(cases)
    git_commit, git_dirty = _git_state(Path(__file__).resolve().parents[3])
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_id": root.name,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "resolved_config": _redacted_config(config),
        "model": {
            "provider": str(getattr(model_config, "provider", "")),
            "name": str(getattr(model_config, "model_name", "")),
            "max_tool_rounds": int(
                getattr(model_config, "max_tool_rounds", 0) or 0
            ),
            "max_consecutive_protocol_rejections": int(
                getattr(
                    model_config,
                    "max_consecutive_protocol_rejections",
                    0,
                )
                or 0
            ),
        },
        "selected_case_ids": [case.case_id for case in selected],
        "class_counts": dict(
            sorted(Counter(str(case.class_name or "unclassified") for case in selected).items())
        ),
        "dataset_sha256": dataset_sha256(config),
    }
    return atomic_write_text(
        root / "run_manifest.json",
        json.dumps(payload, indent=2, ensure_ascii=False),
    )


def dataset_sha256(config: UavFlowEvalConfig) -> str:
    """Hash the classified roster and every referenced task JSON in order."""

    classification = (
        load_classification(config.classified_instr_path)
        if config.classified_instr_path.is_file()
        else {"unclassified": [path.name for path in sorted(config.test_jsons_path.glob("*.json"))]}
    )
    digest = hashlib.sha256()
    for class_name, file_names in classification.items():
        digest.update(class_name.encode("utf-8"))
        digest.update(b"\0")
        for file_name in file_names:
            path = config.test_jsons_path / file_name
            digest.update(file_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def audit_campaign(
    config_path: str | Path,
    eval_dirs: Iterable[str | Path],
    *,
    expected_loop_guard_windows: int,
    segment_auditor: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Audit one or more main/continuation segments as one dataset campaign."""

    config = UavFlowEvalConfig.from_file(config_path)
    roster = load_cases(
        config.uav_flow_repo_path,
        classes=None,
        limit_per_class=None,
        instruction_field=config.instruction_field,
        test_jsons_path=config.test_jsons_path,
        classified_instr_path=config.classified_instr_path,
    )
    expected_ids = [case.case_id for case in roster]
    expected_classes = {case.case_id: case.class_name for case in roster}
    expected_hash = dataset_sha256(config)
    expected_config = config.to_dict()
    expected_config["runtime_loop_guard_windows"] = int(
        expected_loop_guard_windows
    )
    expected_comparable = _comparable_config(expected_config)

    violations: list[dict[str, Any]] = []
    segment_reports: list[dict[str, Any]] = []
    finalized_by_id: dict[str, dict[str, Any]] = {}
    owner_by_id: dict[str, str] = {}
    aggregate_coverage: Counter[str] = Counter()
    reference_model: dict[str, Any] | None = None
    reference_git_commit: str | None = None

    roots = [Path(item).resolve() for item in eval_dirs]
    if not roots:
        _violate(violations, "segments_missing", "at least one --eval-dir is required")

    for root in roots:
        segment_name = root.name
        manifest = _read_object(root / "run_manifest.json", violations, segment_name)
        summary = _read_object(root / "summary.json", violations, segment_name)
        session = _read_object(root / "scene_session.json", violations, segment_name)
        finalized = _load_finalized_cases(root, violations)
        finalized_ids = set(finalized)

        if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
            _violate(
                violations,
                "run_manifest_schema_invalid",
                f"expected {RUN_MANIFEST_SCHEMA!r}",
                segment=segment_name,
            )
        if manifest.get("dataset_sha256") != expected_hash:
            _violate(
                violations,
                "dataset_hash_mismatch",
                "segment dataset does not match the configured 273-case roster",
                segment=segment_name,
            )
        actual_config = manifest.get("resolved_config")
        if not isinstance(actual_config, dict) or _comparable_config(actual_config) != expected_comparable:
            _violate(
                violations,
                "resolved_config_mismatch",
                "segment effective config differs from the requested campaign config",
                segment=segment_name,
            )
        model = manifest.get("model")
        if not isinstance(model, dict):
            _violate(violations, "model_manifest_invalid", "model must be an object", segment=segment_name)
        elif reference_model is None:
            reference_model = dict(model)
        elif model != reference_model:
            _violate(violations, "model_mismatch", "model configuration differs between segments", segment=segment_name)
        commit = manifest.get("git_commit")
        if manifest.get("git_dirty") is not False:
            _violate(violations, "git_worktree_dirty", "segment was not started from a clean worktree", segment=segment_name)
        if not isinstance(commit, str) or not commit:
            _violate(violations, "git_commit_missing", "segment git commit is missing", segment=segment_name)
        elif reference_git_commit is None:
            reference_git_commit = commit
        elif commit != reference_git_commit:
            _violate(violations, "git_commit_mismatch", "git commit differs between segments", segment=segment_name)

        selected_ids = manifest.get("selected_case_ids")
        if not isinstance(selected_ids, list) or any(not isinstance(item, str) for item in selected_ids):
            _violate(violations, "selected_case_ids_invalid", "selected_case_ids must be a string list", segment=segment_name)
            selected_ids = []
        unknown_selected = sorted(set(selected_ids) - set(expected_ids))
        if unknown_selected:
            _violate(violations, "unknown_selected_cases", f"unknown selected IDs: {unknown_selected}", segment=segment_name)
        unselected_finalized = sorted(finalized_ids - set(selected_ids))
        if unselected_finalized:
            _violate(violations, "unselected_finalized_cases", f"finalized cases absent from run manifest: {unselected_finalized}", segment=segment_name)

        run_status = summary.get("run_status")
        if run_status not in {"running", "completed", "interrupted", "failed"}:
            _violate(violations, "run_status_invalid", f"unsupported run_status {run_status!r}", segment=segment_name)
        lifecycle_clean = _audit_segment_lifecycle(session, run_status, violations, segment_name)

        native = segment_auditor(root)
        native_cases = {
            str(item.get("case_id")): item
            for item in native.get("cases", [])
            if isinstance(item, dict) and str(item.get("case_id")) in finalized_ids
        }
        native_case_violations = [
            item
            for item in native.get("violations", [])
            if isinstance(item, dict) and str(item.get("case_id")) in finalized_ids
        ]
        for item in native_case_violations:
            violations.append(
                {
                    "scope": "native_transcript",
                    "segment": segment_name,
                    **item,
                }
            )
        for item in native_cases.values():
            coverage = item.get("coverage")
            if isinstance(coverage, dict):
                for key, value in coverage.items():
                    if isinstance(value, int):
                        aggregate_coverage[key] += value

        for case_id, case_payload in finalized.items():
            case = case_payload.get("case") if isinstance(case_payload.get("case"), dict) else {}
            actual_case_id = str(case.get("case_id") or "")
            if actual_case_id != case_id:
                _violate(violations, "case_id_mismatch", f"case payload has ID {actual_case_id!r}", segment=segment_name, case_id=case_id)
            if case_id not in expected_classes:
                _violate(violations, "unknown_finalized_case", "case is absent from dataset roster", segment=segment_name, case_id=case_id)
            elif case.get("class_name") != expected_classes[case_id]:
                _violate(violations, "case_class_mismatch", f"expected {expected_classes[case_id]!r}, got {case.get('class_name')!r}", segment=segment_name, case_id=case_id)
            if case_id in finalized_by_id:
                _violate(violations, "duplicate_finalized_case", f"already finalized by {owner_by_id[case_id]}", segment=segment_name, case_id=case_id)
                continue
            finalized_by_id[case_id] = _case_result(case_payload, root / "cases" / case_id)
            owner_by_id[case_id] = segment_name

        segment_reports.append(
            {
                "eval_dir": str(root),
                "segment": segment_name,
                "run_status": run_status,
                "selected_case_count": len(selected_ids),
                "finalized_case_count": len(finalized),
                "lifecycle_clean": lifecycle_clean,
                "review_html": str(root / "review.html"),
            }
        )

    finalized_ids = set(finalized_by_id)
    missing_ids = [case_id for case_id in expected_ids if case_id not in finalized_ids]
    unknown_finalized = sorted(finalized_ids - set(expected_ids))
    ordered_results = [
        finalized_by_id[case_id]
        for case_id in expected_ids
        if case_id in finalized_by_id
    ]
    config_codes = {
        "run_manifest_invalid",
        "run_manifest_missing",
        "run_manifest_schema_invalid",
        "dataset_hash_mismatch",
        "resolved_config_mismatch",
        "model_manifest_invalid",
        "model_mismatch",
        "git_worktree_dirty",
        "git_commit_missing",
        "git_commit_mismatch",
        "scene_session_invalid",
        "scene_session_missing",
        "scene_lifecycle_mismatch",
    }
    protocol_valid = not any(
        item.get("scope") == "native_transcript" for item in violations
    )
    trace_valid = bool(ordered_results) and all(
        item.get("trace_valid") is True for item in ordered_results
    )
    tracking_valid = bool(ordered_results) and all(
        item.get("tracking_valid") is True for item in ordered_results
    )
    infrastructure_clean = (
        not any(item.get("code") in config_codes for item in violations)
        and all(item.get("infrastructure_clean") is True for item in ordered_results)
    )
    duplicate_present = any(
        item.get("code") == "duplicate_finalized_case" for item in violations
    )
    coverage_complete = (
        not missing_ids
        and not unknown_finalized
        and not duplicate_present
        and len(finalized_by_id) == len(expected_ids)
    )
    status_counts = Counter(str(item.get("status") or "unknown") for item in ordered_results)
    stopped_reason_counts = Counter(
        str(item.get("stopped_reason"))
        for item in ordered_results
        if item.get("stopped_reason")
    )
    behavior = _behavior_summary(ordered_results)
    return {
        "schema_version": CAMPAIGN_AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "dataset_sha256": expected_hash,
        "expected_case_count": len(expected_ids),
        "expected_class_counts": dict(sorted(Counter(str(case.class_name) for case in roster).items())),
        "finalized_case_count": len(finalized_by_id),
        "coverage_complete": coverage_complete,
        "protocol_valid": protocol_valid,
        "trace_valid": trace_valid,
        "tracking_valid": tracking_valid,
        "infrastructure_clean": infrastructure_clean,
        "status": (
            "ok"
            if coverage_complete
            and protocol_valid
            and trace_valid
            and tracking_valid
            and infrastructure_clean
            and not violations
            else "incomplete_or_invalid"
        ),
        "missing_case_ids": missing_ids,
        "unknown_finalized_case_ids": unknown_finalized,
        "segments": segment_reports,
        "metrics": summarize_metrics(ordered_results),
        "failure_counts": failure_counts(ordered_results),
        "status_counts": dict(sorted(status_counts.items())),
        "stopped_reason_counts": dict(sorted(stopped_reason_counts.items())),
        "transcript_coverage": dict(sorted(aggregate_coverage.items())),
        "agent_behavior": behavior,
        "cases": [
            {
                "case_id": item["case_id"],
                "class_name": item.get("class_name"),
                "status": item.get("status"),
                "completed": bool(item.get("completed")),
                "stopped_reason": item.get("stopped_reason"),
                "ndtw": (item.get("ndtw") or {}).get("ndtw"),
                "action_count": ((item.get("trace") or {}).get("action_count", 0)),
                "trace_valid": item.get("trace_valid"),
                "tracking_valid": item.get("tracking_valid"),
                "infrastructure_clean": item.get("infrastructure_clean"),
                "segment": owner_by_id.get(item["case_id"]),
                "review_html": item.get("review_html"),
            }
            for item in ordered_results
        ],
        "violations": violations,
    }


def write_campaign_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write deterministic JSON, Markdown, HTML, and continuation selectors."""

    root = Path(output_dir)
    missing = [str(item) for item in report.get("missing_case_ids", [])]
    paths = {
        "campaign_audit": atomic_write_text(
            root / "campaign_audit.json",
            json.dumps(report, indent=2, ensure_ascii=False),
        ),
        "campaign_summary": atomic_write_text(root / "campaign_summary.md", _campaign_markdown(report)),
        "campaign_review": atomic_write_text(root / "campaign_review.html", _campaign_html(report)),
        "remaining_case_ids_txt": atomic_write_text(root / "remaining_case_ids.txt", ",".join(missing) + ("\n" if missing else "")),
        "remaining_case_ids_json": atomic_write_text(root / "remaining_case_ids.json", json.dumps(missing, indent=2, ensure_ascii=False)),
    }
    return paths


def _redacted_config(config: UavFlowEvalConfig) -> dict[str, Any]:
    # UavFlowEvalConfig contains paths and switches but no credential values.
    # Keep the env file path for reproducibility without reading its contents.
    return config.to_dict()


def _git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit or None, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in _CONFIG_SELECTOR_FIELDS | _CONFIG_LOCATION_FIELDS
    }


def _read_object(path: Path, violations: list[dict[str, Any]], segment: str) -> dict[str, Any]:
    if not path.is_file():
        _violate(violations, f"{path.stem}_missing", f"missing {path.name}", segment=segment)
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _violate(violations, f"{path.stem}_invalid", f"{type(exc).__name__}: {exc}", segment=segment)
        return {}
    if not isinstance(value, dict):
        _violate(violations, f"{path.stem}_invalid", "expected a JSON object", segment=segment)
        return {}
    return value


def _load_finalized_cases(root: Path, violations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cases_root = root / "cases"
    if not cases_root.is_dir():
        _violate(violations, "cases_directory_missing", f"missing {cases_root}", segment=root.name)
        return {}
    finalized: dict[str, dict[str, Any]] = {}
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_path = case_dir / "case.json"
        audit_path = case_dir / "audit.json"
        if not case_path.is_file() or not audit_path.is_file():
            continue
        try:
            case_payload = json.loads(case_path.read_text(encoding="utf-8"))
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _violate(violations, "finalized_case_invalid", f"{type(exc).__name__}: {exc}", segment=root.name, case_id=case_dir.name)
            continue
        if not isinstance(case_payload, dict) or not isinstance(audit_payload, dict):
            _violate(violations, "finalized_case_invalid", "case.json and audit.json must be objects", segment=root.name, case_id=case_dir.name)
            continue
        finalized[case_dir.name] = case_payload
    return finalized


def _audit_segment_lifecycle(
    session: dict[str, Any],
    run_status: Any,
    violations: list[dict[str, Any]],
    segment: str,
) -> bool:
    expected_stop = 0 if run_status == "running" else 1
    expected = {"start_count": 1, "restart_count": 0, "stop_count": expected_stop}
    clean = True
    for key, value in expected.items():
        if session.get(key) != value:
            clean = False
            _violate(violations, "scene_lifecycle_mismatch", f"{key} expected {value}, got {session.get(key)!r}", segment=segment)
    return clean


def _case_result(payload: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    case = payload.get("case") if isinstance(payload.get("case"), dict) else {}
    return {
        "case_id": str(case.get("case_id") or case_dir.name),
        "class_name": case.get("class_name"),
        "status": payload.get("status"),
        "completed": payload.get("completed"),
        "stopped_reason": payload.get("stopped_reason"),
        "error": payload.get("error"),
        "ndtw": payload.get("ndtw") if isinstance(payload.get("ndtw"), dict) else {},
        "sr_judge": payload.get("sr_judge") if isinstance(payload.get("sr_judge"), dict) else {},
        "trace": payload.get("trace") if isinstance(payload.get("trace"), dict) else {},
        "backend_action_tracking": payload.get("backend_action_tracking") if isinstance(payload.get("backend_action_tracking"), dict) else {},
        "trace_valid": payload.get("trace_valid"),
        "tracking_valid": payload.get("tracking_valid"),
        "infrastructure_clean": payload.get("infrastructure_clean"),
        "review_html": str(case_dir / "review.html"),
    }


def _behavior_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    zero_motion = 0
    tracking_out_of_tolerance = 0
    action_count = 0
    for item in results:
        trace = item.get("trace") or {}
        tracking = item.get("backend_action_tracking") or {}
        action_count += int(trace.get("physical_call_count") or 0)
        zero_motion += int(tracking.get("zero_motion_completion") is True)
        tracking_out_of_tolerance += int(tracking.get("all_observed_within_tolerance") is False)
    return {
        "completed_count": sum(bool(item.get("completed")) for item in results),
        "loop_guard_count": sum(item.get("stopped_reason") == "runtime_loop_guard_reached" for item in results),
        "zero_motion_completion_count": zero_motion,
        "tracking_out_of_tolerance_count": tracking_out_of_tolerance,
        "physical_action_count": action_count,
    }


def _campaign_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    lines = [
        "# UAV-Flow Campaign Summary",
        "",
        f"- Status: {report.get('status')}",
        f"- Coverage complete: {report.get('coverage_complete')}",
        f"- Protocol valid: {report.get('protocol_valid')}",
        f"- Trace valid: {report.get('trace_valid')}",
        f"- Tracking valid: {report.get('tracking_valid')}",
        f"- Infrastructure clean: {report.get('infrastructure_clean')}",
        f"- Finalized: {report.get('finalized_case_count')}/{report.get('expected_case_count')}",
        f"- Overall mean nDTW: {metrics.get('overall_mean_ndtw')}",
        f"- Missing cases: {len(report.get('missing_case_ids') or [])}",
        f"- Violations: {len(report.get('violations') or [])}",
        "",
        "## By Class",
        "",
        "| Class | Evaluated | Mean nDTW |",
        "|---|---:|---:|",
    ]
    for class_name, item in (metrics.get("by_class") or {}).items():
        lines.append(f"| {class_name} | {item.get('evaluated_count')} | {item.get('mean_ndtw')} |")
    return "\n".join(lines) + "\n"


def _campaign_html(report: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    segment_rows = "".join(
        "<tr>"
        f"<td>{esc(item.get('segment'))}</td>"
        f"<td>{esc(item.get('run_status'))}</td>"
        f"<td>{esc(item.get('finalized_case_count'))}/{esc(item.get('selected_case_count'))}</td>"
        f"<td><a href=\"{esc(Path(str(item.get('review_html'))).resolve().as_uri())}\">review</a></td>"
        "</tr>"
        for item in report.get("segments", [])
    )
    case_rows = "".join(
        "<tr>"
        f"<td>{esc(item.get('case_id'))}</td>"
        f"<td>{esc(item.get('class_name'))}</td>"
        f"<td>{esc(item.get('status'))}</td>"
        f"<td>{esc(item.get('ndtw'))}</td>"
        f"<td><a href=\"{esc(Path(str(item.get('review_html'))).resolve().as_uri())}\">transcript</a></td>"
        "</tr>"
        for item in report.get("cases", [])
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>UAV-Flow Campaign</title>"
        "<style>body{font:14px system-ui;margin:2rem;color:#172033}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #d7deea;padding:.45rem;text-align:left}th{background:#f3f6fa}.bad{color:#a11}</style>"
        "</head><body>"
        "<h1>UAV-Flow Campaign</h1>"
        f"<p>Status: <strong>{esc(report.get('status'))}</strong>; finalized {esc(report.get('finalized_case_count'))}/{esc(report.get('expected_case_count'))}; violations {esc(len(report.get('violations') or []))}.</p>"
        f"<p>Protocol valid: {esc(report.get('protocol_valid'))}; trace valid: {esc(report.get('trace_valid'))}; tracking valid: {esc(report.get('tracking_valid'))}; infrastructure clean: {esc(report.get('infrastructure_clean'))}.</p>"
        "<h2>Segments</h2><table><thead><tr><th>Segment</th><th>Status</th><th>Cases</th><th>Review</th></tr></thead>"
        f"<tbody>{segment_rows}</tbody></table>"
        "<h2>Cases</h2><table><thead><tr><th>Case</th><th>Class</th><th>Status</th><th>nDTW</th><th>Review</th></tr></thead>"
        f"<tbody>{case_rows}</tbody></table>"
        "</body></html>"
    )


def _violate(
    violations: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    segment: str | None = None,
    case_id: str | None = None,
) -> None:
    violations.append(
        {
            "scope": "campaign",
            "code": code,
            "message": message,
            "segment": segment,
            "case_id": case_id,
        }
    )
