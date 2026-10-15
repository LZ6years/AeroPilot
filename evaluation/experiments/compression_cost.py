"""Summarize captured planner exchanges and independent compaction-call costs.

Usage comes from provider responses, not projected message-size estimates.
Missing usage and in-flight calls remain explicit. A repeated background-job
snapshot is never treated as another billable compaction request.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int | None]:
    """Normalize Chat/Responses usage without inventing unreported counts."""
    value = usage or {}

    def count(*names: str) -> int | None:
        for name in names:
            number = value.get(name)
            if type(number) is int and number >= 0:
                return number
        return None

    details = value.get("prompt_tokens_details") or value.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return {
        "input_tokens": count("prompt_tokens", "input_tokens"),
        "output_tokens": count("completion_tokens", "output_tokens"),
        "cached_input_tokens": cached if type(cached) is int and cached >= 0 else None,
    }


def summarize_calls(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Deduplicate lifecycle records and sum only known provider counters."""
    calls: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["call_id"])
        previous = calls.get(key)
        if previous is None or record.get("status") != "started":
            calls[key] = record
    values = list(calls.values())
    usage = [normalize_usage(item.get("usage")) for item in values]
    result: dict[str, Any] = {
        "call_count": len(values),
        "pending_call_count": sum(item.get("status") == "started" for item in values),
        "error_call_count": sum(item.get("status") == "error" for item in values),
        "elapsed_seconds": sum(float(item.get("elapsed_seconds") or 0) for item in values),
    }
    for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
        known = [item[name] for item in usage if item[name] is not None]
        result[f"known_{name}"] = sum(known) if known else None
        result[f"missing_{name}_call_count"] = len(values) - len(known)
    return result


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Read one run without modifying saved evidence or calling a model."""
    root = Path(run_dir)
    compaction = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "debug/history_compaction_calls").glob("*.json"))
    ]
    planner = []
    accepted = set()
    for path in sorted([*(root / "debug/tool_loop").rglob("decision_*.json"),
                        *(root / "debug/model_call_metrics").rglob("decision_*.json")]):
        value = json.loads(path.read_text(encoding="utf-8"))
        rounds = value.get("model_rounds") or []
        for index, exchange in enumerate(rounds):
            response = exchange.get("response") or {}
            transport = exchange.get("transport") or {}
            planner.append({
                "call_id": f"{path.relative_to(root).as_posix()}:{index}",
                "status": "error" if exchange.get("error_type") else "parsed",
                "usage": response.get("usage"),
                "elapsed_seconds": transport.get("elapsed_seconds"),
            })
            projection = exchange.get("history_projection") or {}
            compaction_metadata = projection.get("compaction") or {}
            if compaction_metadata.get("source_digest"):
                accepted.add(str(compaction_metadata["source_digest"]))
    return {
        "run_dir": str(root),
        "planner": summarize_calls(planner),
        "compactor": summarize_calls(compaction),
        "accepted_capsule_count": len(accepted),
        "judge_included": False,
    }


def effective_planner_calls(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read success-attempt accounting while retaining excluded audit records.

    A logical response recovered through HTTP retries contributes one effective
    call and its final successful attempt's time. The enclosing trace duration
    remains an audit field because it includes failed attempts and backoff.
    Missing timings/usages are explicit; legacy elapsed time is never silently
    substituted for the required attempt measurement.
    """
    root = Path(run_dir)
    records = []
    for path in sorted([*(root / "debug/tool_loop").rglob("decision_*.json"),
                        *(root / "debug/model_call_metrics").rglob("decision_*.json")]):
        value = json.loads(path.read_text(encoding="utf-8"))
        for index, exchange in enumerate(value.get("model_rounds") or []):
            response = exchange.get("response") or {}
            transport = exchange.get("transport") or {}
            raw_attempts = transport.get("attempts")
            # Failed traces from an older client serialize error.attempts as
            # an integer over the original list. Counts remain usable, but
            # lost attempt timings cannot be reconstructed from total latency.
            attempts = [a for a in raw_attempts if isinstance(a, dict)] if isinstance(raw_attempts, list) else []
            count = transport.get("attempt_count")
            if type(count) is not int or count < 0:
                count = raw_attempts if type(raw_attempts) is int and raw_attempts >= 0 else len(attempts)
            detail_missing = max(0, count - len(attempts))
            if not isinstance(raw_attempts, list) and count:
                detail_missing = count
            successful = [a for a in attempts if a.get("status") == "success"]
            accepted = (not exchange.get("error_type") and isinstance(exchange.get("canonical_turn"), dict)
                        and bool(response) and transport.get("final_status") == "success")
            timing = successful[-1].get("elapsed_seconds") if successful else None
            timing = float(timing) if type(timing) in {int, float} and math.isfinite(timing) and timing >= 0 else None
            projection = exchange.get("history_projection") or {}
            media = projection.get("selected_visual_evidence") or []
            maps = sum(m.get("selection_tier") == "latest_group" for m in media)
            rgb = sum(m.get("source_kind") == "observation" and m.get("selection_tier") != "latest_group" for m in media)
            record = {
                "call_id": f"{path.relative_to(root).as_posix()}:{index}",
                "request_id": transport.get("request_id"), "effective": accepted,
                "excluded_reason": None if accepted else exchange.get("error_type") or "unaccepted_response",
                **normalize_usage(response.get("usage")),
                "decision_seconds": timing if accepted else None,
                "logical_seconds": transport.get("elapsed_seconds"),
                "transport_attempts": count,
                "missing_attempt_detail_count": detail_missing,
                "failed_attempts": (count if transport.get("final_status") == "failed" else max(0, count - 1))
                    if detail_missing else sum(a.get("status") != "success" for a in attempts),
                "failed_attempt_seconds": None if detail_missing else sum(float(a.get("elapsed_seconds") or 0) for a in attempts if a.get("status") != "success"),
                "backoff_seconds": None if detail_missing else sum(float(a.get("backoff_elapsed_seconds", a.get("backoff_seconds")) or 0) for a in attempts),
                "failure_categories": [a.get("category") for a in attempts if a.get("status") != "success"],
                "estimated_input_tokens": projection.get("estimated_input_tokens"),
                "actual_input_tokens": projection.get("actual_input_tokens"),
                "total_images": projection.get("total_image_count"), "rgb_images": rgb,
                "map_images": maps, "tool_images": len(media) - maps - rgb,
                "compression_enabled": projection.get("compression_enabled"),
                "compaction_status": projection.get("compaction_status"),
                "audit_path": str(path),
            }
            records.append(record)
    return records


def effective_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool requests, with missing-data counts instead of fabricated zero cost."""
    valid = [r for r in records if r["effective"]]
    result = {"calls": len(valid), "excluded_responses": len(records) - len(valid),
              "transport_attempts": sum(r["transport_attempts"] for r in records),
              "failed_attempts": sum(r["failed_attempts"] for r in records),
              "missing_attempt_detail_count": sum(r.get("missing_attempt_detail_count", 0) for r in records)}
    for name in ("failed_attempt_seconds", "backoff_seconds"):
        known = [r[name] for r in records if r.get(name) is not None]
        result[name] = sum(known) if len(known) == len(records) else None
        result["known_" + name] = sum(known)
    for name in ("input_tokens", "output_tokens", "cached_input_tokens", "decision_seconds"):
        known = [r[name] for r in valid if r.get(name) is not None]
        result[name] = sum(known) if len(known) == len(valid) else None
        result["known_" + name] = sum(known)
        result["missing_" + name] = len(valid) - len(known)
    result["decision_latency_seconds"] = result["decision_seconds"] / len(valid) if valid and result["decision_seconds"] is not None else None
    estimates = [r["estimated_input_tokens"] for r in records if type(r.get("estimated_input_tokens")) is int]
    result["peak_estimated_input_tokens"] = max(estimates) if estimates else None
    result["peak_estimate_coverage_complete"] = bool(records) and len(estimates) == len(records)
    for name in ("total_images", "rgb_images", "map_images", "tool_images"):
        known = [r[name] for r in valid if r.get(name) is not None]
        result["mean_" + name] = sum(known) / len(known) if known else None
        result["max_" + name] = max(known) if known else None
    return result
