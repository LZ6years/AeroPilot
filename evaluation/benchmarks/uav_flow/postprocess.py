"""Rebuild UAV-Flow reviews from finalized, already persisted case outputs.

This module is deliberately offline.  It does not load MapPackages, create a
Runtime, connect to UnrealCV, or call model services.  Its only inputs are the
human/debug artifacts that ``UavFlowBenchmarkRunner.run_case`` has already
committed beneath an evaluation directory.

The same summary writer is also used by the live runner.  Keeping live and
recovered reports on one construction path prevents a repaired interrupted
batch from having subtly different aggregate metrics or HTML fields.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation.benchmarks.core import (
    BenchmarkAuditLog,
    BenchmarkCaseReview,
    BenchmarkReviewRenderer,
    BenchmarkSummaryReview,
    ResolvedFrameAsset,
)
from evaluation.benchmarks.core.report_io import atomic_write_text
from evaluation.benchmarks.uav_flow.adapter import UAV_FLOW_PROTOCOL
from evaluation.benchmarks.uav_flow.reports import write_eval_summary
from evaluation.benchmarks.uav_flow.review import (
    build_uav_flow_review_sections,
)


_OUTCOME_FIELDS = (
    "status",
    "completed",
    "stopped",
    "stopped_reason",
    "stop_source",
    "error",
    "cleanup_errors",
)
_RESULT_FIELDS = (
    "run_dir",
    "audit_summary",
    "visual_diagnostics",
    "plot_result",
    "ndtw",
    "turn_diagnostics",
    "backend_action_tracking",
    "scenario_diagnostics",
    "sr_judge",
    "stmr",
    "metric_error",
)
_RUN_STATUSES = {"running", "completed", "interrupted", "failed"}


def render_evaluation_report(
    eval_dir: str | Path,
    results: list[dict[str, Any]],
    case_reviews: Iterable[BenchmarkCaseReview],
    console_lines: list[str],
    *,
    report_metadata: dict[str, Any] | None = None,
    render_cases: bool = True,
) -> dict[str, str]:
    """Write one consistent JSON/Markdown/HTML evaluation snapshot.

    The caller decides whether case pages need rendering.  A live run writes a
    case page immediately in ``run_case`` and only refreshes the root index;
    offline recovery passes every reconstructed case here.
    """

    root = Path(eval_dir)
    reviews = tuple(case_reviews)
    summary_paths = write_eval_summary(
        root,
        results,
        console_lines,
        report_metadata=report_metadata,
    )
    summary_payload = _read_json(Path(summary_paths["summary_json"]))
    summary_review = BenchmarkSummaryReview(
        benchmark_name="UAV-Flow",
        protocol_variant=UAV_FLOW_PROTOCOL,
        metrics=dict(summary_payload.get("metrics") or {}),
        failure_counts=dict(summary_payload.get("failure_counts") or {}),
        cases=tuple(_summary_case(item) for item in results),
        metadata=_summary_metadata(summary_payload),
    )
    renderer = BenchmarkReviewRenderer()
    if render_cases:
        for review in reviews:
            renderer.write_case(root, review)
    review_paths = renderer.write_summary(root, summary_review)
    summary_paths.update(review_paths)
    _append_html_review_links(summary_paths, review_paths)
    return summary_paths


def recover_existing_evaluation(
    eval_dir: str | Path,
    *,
    run_status: str = "interrupted",
    expected_case_count: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Render an interrupted evaluation from finalized case directories.

    A directory without ``case.json`` is an expected incomplete case after an
    interruption and is left untouched.  A finalized case with a missing or
    malformed ``audit.json`` is excluded and reported as a validation error.
    ``strict`` turns those validation errors into a pre-write failure; it does
    not reject ordinary incomplete directories.
    """

    status = str(run_status).strip().lower()
    if status not in _RUN_STATUSES:
        raise ValueError(
            f"run_status must be one of {sorted(_RUN_STATUSES)}, got {run_status!r}"
        )
    if expected_case_count is not None and expected_case_count < 0:
        raise ValueError("expected_case_count must be non-negative")

    root = Path(eval_dir)
    cases_root = root / "cases"
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Evaluation cases directory not found: {cases_root}")

    recovered: list[tuple[str, dict[str, Any], BenchmarkCaseReview]] = []
    incomplete_case_ids: list[str] = []
    excluded_cases: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_path = case_dir / "case.json"
        audit_path = case_dir / "audit.json"
        if not case_path.is_file():
            incomplete_case_ids.append(case_dir.name)
            continue
        if not audit_path.is_file():
            excluded_cases.append(
                {"case_id": case_dir.name, "reason": "audit_json_missing"}
            )
            continue
        try:
            case_payload = _read_json(case_path)
            audit_payload = _read_json(audit_path)
            case_review = _case_review_from_payload(
                case_dir,
                case_payload,
                audit_payload,
                warnings=warnings,
            )
            result = _case_result_from_payload(case_dir, case_payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            excluded_cases.append(
                {
                    "case_id": case_dir.name,
                    "reason": "case_reconstruction_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        recovered.append(
            (str(audit_payload.get("started_at") or case_dir.name), result, case_review)
        )

    if strict and (excluded_cases or warnings):
        raise ValueError(
            "Strict report recovery rejected validation problems: "
            + json.dumps(
                {"excluded_cases": excluded_cases, "warnings": warnings},
                ensure_ascii=False,
            )
        )

    # Audit start time preserves actual execution order even when case IDs are
    # dataset timestamps unrelated to this evaluation run.
    recovered.sort(key=lambda item: (item[0], item[1]["case_id"]))
    results = [item[1] for item in recovered]
    case_reviews = [item[2] for item in recovered]
    expected = len(results) if expected_case_count is None else expected_case_count
    metadata = {
        "run_status": status,
        "expected_case_count": expected,
        "incomplete_case_ids": incomplete_case_ids,
        "excluded_case_ids": [item["case_id"] for item in excluded_cases],
        "report_generated_from": "existing_case_outputs",
    }
    console_lines = [
        f"uav_flow_report_recovery: output={root}",
        *(_console_line(item) for item in results),
    ]
    summary_paths = render_evaluation_report(
        root,
        results,
        case_reviews,
        console_lines,
        report_metadata=metadata,
        render_cases=True,
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "finalized_case_count": len(results),
        "scanned_case_directory_count": sum(
            1 for path in cases_root.iterdir() if path.is_dir()
        ),
        "generated_case_html_count": len(case_reviews),
        "excluded_cases": excluded_cases,
        "warnings": warnings,
        "summary_paths": summary_paths,
    }
    manifest_path = root / "report_manifest.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False),
    )
    return {**manifest, "report_manifest": str(manifest_path)}


def _case_review_from_payload(
    case_dir: Path,
    case_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    *,
    warnings: list[dict[str, Any]],
) -> BenchmarkCaseReview:
    case = _required_dict(case_payload, "case")
    case_id = str(case.get("case_id") or case_dir.name)
    if case_id != case_dir.name:
        raise ValueError(
            f"case_id {case_id!r} does not match directory {case_dir.name!r}"
        )
    audit = _audit_from_payload(audit_payload)
    frame_assets = tuple(
        _frame_asset_from_payload(item)
        for item in _dict_items(audit_payload.get("frame_assets"))
    )
    _validate_frame_assets(case_id, case_dir, frame_assets, warnings)

    ndtw_result = _optional_dict(case_payload.get("ndtw"))
    sr_result = _optional_dict(case_payload.get("sr_judge"))
    visual_diagnostics = _optional_dict(
        case_payload.get("visual_diagnostics")
    )
    stmr_summary = _optional_dict(case_payload.get("stmr"))
    plot_result = _optional_dict(case_payload.get("plot_result"))
    turn_diagnostics = _optional_dict_or_none(
        case_payload.get("turn_diagnostics")
    )
    backend_tracking = _optional_dict_or_none(
        case_payload.get("backend_action_tracking")
    )
    scenario_diagnostics = _optional_dict_or_none(
        case_payload.get("scenario_diagnostics")
    )
    outcome = {key: case_payload.get(key) for key in _OUTCOME_FIELDS}

    return BenchmarkCaseReview(
        case_id=case_id,
        benchmark_name="UAV-Flow",
        protocol_variant=UAV_FLOW_PROTOCOL,
        instruction=str(
            case.get("selected_instruction")
            or case.get("instruction_unified")
            or case.get("instruction")
            or ""
        ),
        outcome=outcome,
        metrics=_case_metrics(
            case,
            ndtw_result,
            sr_result,
            turn_diagnostics,
            backend_tracking,
        ),
        audit=audit,
        frame_assets=frame_assets,
        sections=build_uav_flow_review_sections(
            case_dir=case_dir,
            ndtw_result=ndtw_result,
            sr_result=sr_result,
            visual_diagnostics=visual_diagnostics,
            stmr_summary=stmr_summary,
            plot_result=plot_result,
            turn_diagnostics=turn_diagnostics,
            backend_action_tracking=backend_tracking,
            scenario_diagnostics=scenario_diagnostics,
        ),
        metadata=_optional_dict(case_payload.get("planner_safe_case")),
    )


def _audit_from_payload(payload: dict[str, Any]) -> BenchmarkAuditLog:
    return BenchmarkAuditLog(
        run_dir=str(payload.get("run_dir") or ""),
        started_at=_optional_string(payload.get("started_at")),
        ended_at=str(payload.get("ended_at") or ""),
        task_initial_observation_ref=_optional_string(
            payload.get("task_initial_observation_ref")
        ),
        observations=tuple(_dict_items(payload.get("observations"))),
        frames=tuple(_dict_items(payload.get("frames"))),
        poses=tuple(_dict_items(payload.get("poses"))),
        actions=tuple(_dict_items(payload.get("actions"))),
        agent_messages=tuple(_dict_items(payload.get("agent_messages"))),
        events=tuple(_dict_items(payload.get("events"))),
        debug_events=tuple(_dict_items(payload.get("debug_events"))),
        planner_debug=tuple(_dict_items(payload.get("planner_debug"))),
        agent_transcript=tuple(
            _dict_items(payload.get("agent_transcript"))
        ),
        observation_anchors={
            str(key): dict(value)
            for key, value in _optional_dict(
                payload.get("observation_anchors")
            ).items()
            if isinstance(value, dict)
        },
        diagnostics=tuple(_dict_items(payload.get("diagnostics"))),
    )


def _frame_asset_from_payload(payload: dict[str, Any]) -> ResolvedFrameAsset:
    return ResolvedFrameAsset(
        frame_id=str(payload.get("frame_id") or "frame"),
        observation_id=_optional_string(payload.get("observation_id")),
        modality=str(payload.get("modality") or "unknown"),
        timestamp=_optional_string(payload.get("timestamp")),
        display_path=_optional_string(payload.get("display_path")),
        raw_path=_optional_string(payload.get("raw_path")),
        source=_optional_string(payload.get("source")),
        unavailable_reason=_optional_string(payload.get("unavailable_reason")),
    )


def _case_result_from_payload(
    case_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    case = _required_dict(payload, "case")
    result = {
        "case_id": str(case.get("case_id") or case_dir.name),
        "class_name": case.get("class_name"),
        **{key: payload.get(key) for key in _OUTCOME_FIELDS},
        **{key: payload.get(key) for key in _RESULT_FIELDS},
        "case_dir": str(case_dir),
        "review_html": str(case_dir / "review.html"),
        "audit_path": str(case_dir / "audit.json"),
    }
    return result


def _case_metrics(
    case: dict[str, Any],
    ndtw: dict[str, Any],
    sr: dict[str, Any],
    turn: dict[str, Any] | None,
    tracking: dict[str, Any] | None,
) -> dict[str, Any]:
    turn = turn or {}
    tracking = tracking or {}
    turn_tracking = _optional_dict(turn.get("backend_turn_tracking"))
    return {
        "class_name": case.get("class_name"),
        "ndtw": ndtw.get("ndtw"),
        "ndtw_status": ndtw.get("status"),
        "sr_status": sr.get("status"),
        "reference_yaw_delta_deg": turn.get("reference_yaw_delta_deg"),
        "actual_yaw_delta_deg": turn.get("actual_yaw_delta_deg"),
        "final_yaw_error_deg": turn.get("final_yaw_error_deg"),
        "translation_drift_m": turn.get("translation_drift_m"),
        "turn_tracking_max_abs_error_deg": turn_tracking.get(
            "max_abs_error_deg"
        ),
        "translation_tracking_max_error_m": tracking.get(
            "max_position_error_m"
        ),
        "zero_motion_completion": tracking.get("zero_motion_completion"),
        "trace_valid": tracking.get("trace_valid"),
        "tracking_valid": tracking.get("tracking_valid"),
    }


def _summary_case(result: dict[str, Any]) -> dict[str, Any]:
    turn = _optional_dict(result.get("turn_diagnostics"))
    turn_tracking = _optional_dict(turn.get("backend_turn_tracking"))
    tracking = _optional_dict(result.get("backend_action_tracking"))
    ndtw = _optional_dict(result.get("ndtw"))
    return {
        "case_id": result.get("case_id"),
        "status": result.get("status"),
        "metrics": {
            "ndtw": ndtw.get("ndtw"),
            "reference_yaw_delta_deg": turn.get("reference_yaw_delta_deg"),
            "actual_yaw_delta_deg": turn.get("actual_yaw_delta_deg"),
            "final_yaw_error_deg": turn.get("final_yaw_error_deg"),
            "translation_drift_m": turn.get("translation_drift_m"),
            "turn_tracking_max_abs_error_deg": turn_tracking.get(
                "max_abs_error_deg"
            ),
            "translation_tracking_max_error_m": tracking.get(
                "max_position_error_m"
            ),
            "zero_motion_completion": tracking.get(
                "zero_motion_completion"
            ),
            "trace_valid": tracking.get("trace_valid"),
            "tracking_valid": tracking.get("tracking_valid"),
        },
    }


def _summary_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary.get(key)
        for key in (
            "run_status",
            "expected_case_count",
            "finalized_case_count",
            "incomplete_case_ids",
            "excluded_case_ids",
            "report_generated_from",
        )
        if key in summary
    }


def _console_line(result: dict[str, Any]) -> str:
    ndtw = _optional_dict(result.get("ndtw"))
    return (
        f"{result.get('case_id')}: status={result.get('status')} "
        f"ndtw={ndtw.get('ndtw')}"
    )


def _append_html_review_links(
    summary_paths: dict[str, str],
    review_paths: dict[str, str],
) -> None:
    html_review = review_paths.get("html_review")
    if not html_review:
        return
    summary_md = Path(summary_paths["summary_md"])
    console_txt = Path(summary_paths["console_txt"])
    atomic_write_text(
        summary_md,
        summary_md.read_text(encoding="utf-8")
        + f"\n## HTML Review\n\n- `{html_review}`\n",
    )
    atomic_write_text(
        console_txt,
        console_txt.read_text(encoding="utf-8")
        + f"html_review: {html_review}\n",
    )


def _validate_frame_assets(
    case_id: str,
    case_dir: Path,
    assets: tuple[ResolvedFrameAsset, ...],
    warnings: list[dict[str, Any]],
) -> None:
    for asset in assets:
        for field_name in ("display_path", "raw_path"):
            value = getattr(asset, field_name)
            if value and not _existing_report_path(value, case_dir):
                warnings.append(
                    {
                        "case_id": case_id,
                        "frame_id": asset.frame_id,
                        "field": field_name,
                        "path": value,
                        "reason": "declared_frame_asset_missing",
                    }
                )


def _existing_report_path(value: str, case_dir: Path) -> bool:
    path = Path(value)
    if path.is_file():
        return True
    return not path.is_absolute() and (case_dir / path).is_file()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Expected object field {key!r}")
    return value


def _optional_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_dict_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _optional_string(value: Any) -> str | None:
    return None if value in (None, "") else str(value)
