"""Read-only scoring projection: admit explicit agent-blocked endpoints.

Usage: flyagent python blocked_success_audit.py --output CAMPAIGN.
Preserves original receipts and excludes technical failures. Reports both the
original 20 m SR and revised 20/25 m endpoint criteria, with per-case evidence.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue
from evaluation.experiments.image_budget_report import write_csv
from evaluation.benchmarks.core.report_io import write_json


def classify(dataset, result):
    metrics = result.get("metrics") or {}
    distance = metrics.get("distance_to_goal" if dataset == "aerial_vln" else "navigation_error")
    valid_distance = type(distance) in (int, float) and math.isfinite(distance) and distance >= 0
    blocked = result.get("stop_source") == "runtime.report_blocked" or str(result.get("stopped_reason", "")).startswith("agent_reported_blocked:")
    clean = not result.get("error") and not result.get("infrastructure_error") and not result.get("metric_error")
    normal = bool(result.get("completed") and not result.get("stopped"))
    original = bool(metrics.get("success"))
    return {"original_success": original, "blocked": blocked, "final_distance_m": distance if valid_distance else None,
            "within_20m": valid_distance and distance <= 20,
            "within_25m": valid_distance and distance <= 25,
            "revised_sr20_success": clean and (original or (blocked and valid_distance and distance <= 20)),
            "revised_sr25_success": clean and valid_distance and distance <= 25 and (normal or blocked or original)}


def audit(root):
    state = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    rows = []
    for key, item in state["tasks"].items():
        if item["status"] != "scored":
            continue
        result = item.get("result") or {}
        retained = any(a.get("status") == "scored" and a.get("valid_score") and a.get("cleanup_verified")
                       for a in item["attempts"])
        if not retained or ParallelRecoveryQueue.infrastructure_record(result):
            continue
        t = item["task"]
        rows.append({"task_key": key, "dataset": t["dataset"], "image_budget": t["image_budget"],
                     "episode_id": t["episode_id"], **classify(t["dataset"], result),
                     "stop_source": result.get("stop_source"), "stopped_reason": result.get("stopped_reason"),
                     "run_dir": result.get("run_dir")})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["image_budget"])].append(row)
    summary = []
    for (dataset, budget), items in sorted(groups.items()):
        counts = {k: sum(bool(r[k]) for r in items) for k in
                  ("original_success", "revised_sr20_success", "revised_sr25_success", "within_25m")}
        summary.append({"dataset": dataset, "image_budget": budget, "valid": len(items), **counts,
                        "blocked_failed_within20": sum(r["blocked"] and not r["original_success"] and r["within_20m"] for r in items),
                        "blocked_failed_within25": sum(r["blocked"] and not r["original_success"] and r["within_25m"] for r in items),
                        "other_failed_within25": sum(not r["blocked"] and not r["original_success"] and r["within_25m"] for r in items)})
    directory = root / "blocked_success_audit"
    directory.mkdir(exist_ok=True)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "rule": "SR20: original success OR explicit agent blocked at <=20m. SR25: normal completion or explicit agent blocked at <=25m. Technical failures excluded; original receipts unchanged.",
              "distance": "AerialVLN-S XY; TravelUAV 3D; final endpoint only, not oracle path minimum.",
              "phases": summary}
    write_json(directory / "metrics.json", report)
    write_csv(directory / "per_case.csv", rows)
    write_csv(directory / "failed_within25.csv", [r for r in rows if not r["original_success"] and r["within_25m"]])
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output.resolve())
