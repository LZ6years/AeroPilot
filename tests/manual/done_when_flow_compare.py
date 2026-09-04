"""Compare completed DoneWhen experiment outputs without starting a simulator.

This developer tool treats the persisted ``case.json`` and ``audit.json`` files
as an offline replay corpus.  It does not call UE, UnrealCV, EnvClient, the task
parser, or a model.  Its purpose is to make the three branch runs comparable
after each branch has produced outputs from the same fixed case IDs.

Example:

    python tests/manual/done_when_flow_compare.py ^
      --run A=H:/.../donewhen_static_dynamic_canary5_20260904 ^
      --run B=H:/.../donewhen_initial_grounded_canary5_20260904 ^
      --run C=H:/.../donewhen_adaptive_revision_canary5_20260904 ^
      --output H:/.../donewhen_flow_comparison.json

The report intentionally separates task behavior from infrastructure errors.
It never recomputes benchmark metrics or edits the source run directories.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


UNSTABLE_TERMINAL_PHRASES = (
    "visible",
    "in view",
    "depth confirms",
    "as judged by",
    "evidence confirms",
    "usable bearing",
)
IDEAL_EVIDENCE_UNKNOWN_PHRASES = (
    "exact distance",
    "world coordinate",
    "local coordinate axis",
    "contact sensor",
    "additional sensor",
    "cannot confirm with certainty",
)
INFRASTRUCTURE_TERMS = (
    "ssl",
    "eof",
    "connection refused",
    "timed out",
    "unrealcv",
    "camera follow",
)


def load_run(label: str, eval_dir: Path) -> dict[str, Any]:
    """Return one read-only aggregate from finalized case directories."""

    cases_root = eval_dir / "cases"
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    if not cases_root.is_dir():
        raise FileNotFoundError(f"cases directory not found: {cases_root}")
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_path = case_dir / "case.json"
        audit_path = case_dir / "audit.json"
        if not case_path.is_file() or not audit_path.is_file():
            skipped.append({"case_id": case_dir.name, "reason": "missing case.json or audit.json"})
            continue
        try:
            case_data = _read_json(case_path)
            audit_data = _read_json(audit_path)
        except (OSError, ValueError) as exc:
            skipped.append({"case_id": case_dir.name, "reason": f"invalid JSON: {exc}"})
            continue
        records.append(_case_record(case_dir.name, case_data, audit_data))

    ndtw_values = [record["ndtw"] for record in records if isinstance(record["ndtw"], (int, float))]
    return {
        "label": label,
        "eval_dir": str(eval_dir.resolve()),
        "finalized_case_count": len(records),
        "case_ids": [record["case_id"] for record in records],
        "mean_ndtw": statistics.fmean(ndtw_values) if ndtw_values else None,
        "total_planner_windows": sum(record["planner_windows"] for record in records),
        "loop_guard_count": sum(record["loop_guard"] for record in records),
        "infrastructure_error_count": sum(record["infrastructure_error"] for record in records),
        "unstable_terminal_condition_count": sum(record["unstable_terminal_conditions"] for record in records),
        "unknown_count": sum(record["unknown_count"] for record in records),
        "ideal_evidence_unknown_count": sum(record["ideal_evidence_unknown_count"] for record in records),
        "supported_count": sum(record["supported_count"] for record in records),
        "semantic_revision_count": sum(record["semantic_revision_count"] for record in records),
        "adaptive_kept_count": sum(record["adaptive_kept_count"] for record in records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "cases": records,
        "skipped": skipped,
    }


def compare_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a manifest that also verifies the fixed-case comparison contract."""

    case_sets = [set(run["case_ids"]) for run in runs]
    same_cases = all(case_set == case_sets[0] for case_set in case_sets[1:]) if case_sets else True
    return {
        "schema_version": "done_when_flow_comparison_v1",
        "same_case_ids": same_cases,
        "common_case_ids": sorted(set.intersection(*case_sets)) if case_sets else [],
        "runs": runs,
    }


def _case_record(case_id: str, case_data: dict[str, Any], audit_data: dict[str, Any]) -> dict[str, Any]:
    texts = list(_walk_strings(audit_data))
    lowered = [text.lower() for text in texts]
    assessments = list(_completion_items(audit_data))
    terminal_conditions = list(_done_when_conditions(audit_data))
    status = str(case_data.get("status", "unknown"))
    stopped_reason = str(case_data.get("stopped_reason") or "")
    error = str(case_data.get("error") or "")
    ndtw_data = case_data.get("ndtw", {})
    ndtw = ndtw_data.get("ndtw") if isinstance(ndtw_data, dict) else None
    events = audit_data.get("events", [])
    debug_events = audit_data.get("debug_events", [])
    event_names = [
        str(item.get("event", ""))
        for item in [*(events if isinstance(events, list) else []), *(debug_events if isinstance(debug_events, list) else [])]
        if isinstance(item, dict)
    ]
    return {
        "case_id": case_id,
        "class_name": _nested(case_data, "case", "class_name"),
        "status": status,
        "stopped_reason": stopped_reason or None,
        "ndtw": ndtw,
        "planner_windows": len(audit_data.get("planner_debug", [])) if isinstance(audit_data.get("planner_debug"), list) else 0,
        "loop_guard": int("loop_guard" in stopped_reason),
        "infrastructure_error": int(any(term in (error + " " + stopped_reason).lower() for term in INFRASTRUCTURE_TERMS)),
        "unstable_terminal_conditions": sum(
            any(phrase in condition.lower() for phrase in UNSTABLE_TERMINAL_PHRASES)
            for condition in terminal_conditions
        ),
        "unknown_count": sum(item.get("status") == "unknown" for item in assessments),
        "ideal_evidence_unknown_count": sum(
            item.get("status") == "unknown"
            and any(phrase in str(item.get("evidence", "")).lower() for phrase in IDEAL_EVIDENCE_UNKNOWN_PHRASES)
            for item in assessments
        ),
        "supported_count": sum(item.get("status") == "supported" for item in assessments),
        "semantic_revision_count": event_names.count("semantic_plan_update_applied"),
        "adaptive_kept_count": event_names.count("adaptive_done_when_kept"),
    }


def _completion_items(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if set(value) >= {"index", "status", "evidence"}:
            yield value
        for item in value.values():
            yield from _completion_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from _completion_items(item)


def _done_when_conditions(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        raw = value.get("done_when")
        if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
            yield from raw
        for item in value.values():
            yield from _done_when_conditions(item)
    elif isinstance(value, list):
        for item in value:
            yield from _done_when_conditions(item)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("root must be an object")
    return value


def _run_arg(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must be LABEL=EVAL_DIR")
    return label.strip(), Path(path.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_run_arg)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_runs([load_run(label, path) for label, path in args.run])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "same_case_ids": report["same_case_ids"]}, ensure_ascii=False))
    return 0 if report["same_case_ids"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
