"""Summarize runtime control action frequency from UAV-Flow run outputs.

This script is a post-run diagnostic aid. It reads existing benchmark summary
and run event files, then reports how often runtime control actions were
requested and applied. It does not affect benchmark execution or normal runtime
behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CONTROL_ACTION = "semantic_plan_update"


def main() -> int:
    args = _parser().parse_args()
    eval_dir = Path(args.eval_dir)
    summary_path = eval_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"summary.json not found under {eval_dir}")

    summary = _read_json(summary_path)
    cases = summary.get("cases", []) if isinstance(summary.get("cases"), list) else []
    case_summaries = [_summarize_case(case) for case in cases if isinstance(case, dict)]
    payload = {
        "eval_dir": str(eval_dir),
        "control_action": CONTROL_ACTION,
        "case_count": len(case_summaries),
        "cases": case_summaries,
        "totals": _totals(case_summaries),
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize semantic_plan_update frequency from UAV-Flow eval artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("eval_dir", help="UAV-Flow eval output directory containing summary.json.")
    parser.add_argument("--output-json", default=None, help="Optional JSON report path.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown report path.")
    return parser


def _summarize_case(case: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(case.get("run_dir", "")))
    events = _read_jsonl(run_dir / "events.jsonl")
    debug_events = _read_jsonl(run_dir / "debug" / "debug_events.jsonl")
    all_events = events + debug_events
    requests = _control_events(events, "runtime_control_action_requested")
    starts = _control_events(debug_events, "runtime_control_action_started")
    public_completions = _control_events(events, "runtime_control_action_completed")
    debug_completions = _control_debug_completions(debug_events)
    applied = _events(debug_events, "semantic_plan_update_applied")
    discarded = _events(debug_events, "semantic_plan_update_discarded")
    windows = _count_windows(all_events, run_dir)
    request_density = round(len(requests) / windows, 3) if windows else 0.0
    request_statuses = Counter(_payload_status(item) for item in requests)
    start_statuses = Counter(_payload_status(item) for item in starts)
    discard_reasons = Counter(_payload(item).get("discard_reason", "unknown") for item in discarded)
    phase_counts = Counter(_phase_key(item) for item in requests)
    repeated_pending_requests = _count_repeated_pending_requests(all_events)
    return {
        "case_id": case.get("case_id"),
        "class_name": case.get("class_name"),
        "status": case.get("status"),
        "completed": case.get("completed"),
        "run_dir": str(run_dir),
        "phase_window_count": windows,
        "request_count": len(requests),
        "request_density": request_density,
        "request_statuses": dict(sorted(request_statuses.items())),
        "started_count": len(starts),
        "start_statuses": dict(sorted(start_statuses.items())),
        "public_completed_count": len(public_completions),
        "debug_validation_completed_count": len(debug_completions),
        "applied_count": len(applied),
        "discarded_count": len(discarded),
        "discard_reasons": dict(sorted(discard_reasons.items())),
        "requests_by_phase": dict(sorted(phase_counts.items())),
        "repeated_pending_request_count": repeated_pending_requests,
    }


def _count_windows(events: list[dict[str, Any]], run_dir: Path) -> int:
    phase_window_events = sum(1 for item in events if item.get("event") == "phase_window_recorded")
    if phase_window_events:
        return phase_window_events
    planned_events = sum(1 for item in events if item.get("event") == "model_action_planned")
    if planned_events:
        return planned_events
    debug_dir = run_dir / "debug" / "action_planner"
    return len(list(debug_dir.glob("*.json"))) if debug_dir.exists() else 0


def _count_repeated_pending_requests(events: list[dict[str, Any]]) -> int:
    pending = False
    repeated = 0
    for item in sorted(events, key=lambda record: str(record.get("timestamp", ""))):
        event = item.get("event")
        if event == "runtime_control_action_started" and _payload(item).get("name") == CONTROL_ACTION:
            if _payload(item).get("status") == "pending":
                pending = True
        elif event == "runtime_control_action_requested" and _payload(item).get("name") == CONTROL_ACTION:
            if pending:
                repeated += 1
        elif event in {"runtime_control_action_completed", "semantic_plan_update_applied", "semantic_plan_update_discarded"}:
            if _payload(item).get("name") == CONTROL_ACTION or event.startswith("semantic_plan_update"):
                pending = False
    return repeated


def _totals(cases: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "phase_window_count": sum(int(case.get("phase_window_count") or 0) for case in cases),
        "request_count": sum(int(case.get("request_count") or 0) for case in cases),
        "started_count": sum(int(case.get("started_count") or 0) for case in cases),
        "applied_count": sum(int(case.get("applied_count") or 0) for case in cases),
        "discarded_count": sum(int(case.get("discarded_count") or 0) for case in cases),
        "repeated_pending_request_count": sum(int(case.get("repeated_pending_request_count") or 0) for case in cases),
    }
    windows = totals["phase_window_count"]
    totals["request_density"] = round(totals["request_count"] / windows, 3) if windows else 0.0
    return totals


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Runtime Control Frequency Summary",
        "",
        f"- Eval dir: `{payload.get('eval_dir')}`",
        f"- Control action: `{payload.get('control_action')}`",
        f"- Case count: {payload.get('case_count')}",
        "",
        "## Totals",
        "",
    ]
    totals = payload.get("totals", {})
    for key in [
        "phase_window_count",
        "request_count",
        "request_density",
        "started_count",
        "applied_count",
        "discarded_count",
        "repeated_pending_request_count",
    ]:
        lines.append(f"- {key}: {totals.get(key)}")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Class | Windows | Requests | Density | Started | Applied | Discarded | Repeated Pending |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in payload.get("cases", []):
        lines.append(
            "| {case_id} | {class_name} | {phase_window_count} | {request_count} | {request_density} | "
            "{started_count} | {applied_count} | {discarded_count} | {repeated_pending_request_count} |".format(
                **case
            )
        )
    return "\n".join(lines) + "\n"


def _control_events(records: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [
        item
        for item in records
        if item.get("event") == event_name and _payload(item).get("name") == CONTROL_ACTION
    ]


def _control_debug_completions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for item in records:
        if item.get("event") != "runtime_control_action_completed":
            continue
        payload = _payload(item)
        request = payload.get("request", {}) if isinstance(payload.get("request"), dict) else {}
        result = payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}
        if request.get("name") == CONTROL_ACTION or result.get("name") == CONTROL_ACTION:
            matches.append(item)
    return matches


def _events(records: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [item for item in records if item.get("event") == event_name]


def _phase_key(item: dict[str, Any]) -> str:
    payload = _payload(item)
    return str(payload.get("phase_key") or payload.get("phase") or "unknown")


def _payload_status(item: dict[str, Any]) -> str:
    return str(_payload(item).get("status") or "unknown")


def _payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload")
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
