"""Common coverage and case review indexes; metric definitions remain adapters'."""
from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path

from evaluation.benchmarks.core.campaign.compat import adapter_for, dataset_name
from evaluation.benchmarks.core.report_io import atomic_write_text, write_json


def write_report(output, manifest, state):
    """Aggregate only valid retained results, keeping missing coverage explicit."""
    root = Path(output)
    rows, groups = [], {}
    for key, item in state["tasks"].items():
        task = item["task"]
        dataset = dataset_name(manifest, task)
        group = groups.setdefault((dataset, task["split"]), {"planned": 0, "valid": 0, "metrics": {}})
        group["planned"] += 1
        result = item.get("result") if item["status"] == "scored" else None
        original_metrics = (result or {}).get("metrics") or {}
        metrics = {field: original_metrics.get(field) for field in adapter_for(manifest, task).metric_fields} if result else {}
        row = {"task_key": key, "dataset": dataset, "split": task["split"], "episode_id": task["episode_id"],
               "scene_id": task.get("scene_id"), "status": item["status"],
               "attempts": len(item["attempts"]), "review_path": None, "metrics": metrics,
               "distance_to_goal": original_metrics.get("distance_to_goal"),
               "marginal_20_25m": original_metrics.get("marginal_20_25m", False),
               "stop_reason": (result or {}).get("stopped_reason") or (result or {}).get("status"),
               "cleanup_verified": bool(item["attempts"] and item["attempts"][-1].get("cleanup_verified"))}
        if result:
            group["valid"] += 1
            for field in adapter_for(manifest, task).metric_fields:
                value = metrics.get(field)
                if type(value) in (int, float, bool):
                    group["metrics"].setdefault(field, []).append(float(value))
            if result.get("run_dir"):
                row["review_path"] = str(Path(result["run_dir"]).parents[1] / "evaluation/cases" / task["episode_id"] / "review.html")
        rows.append(row)
    summaries = []
    for (dataset, split), group in groups.items():
        values = {key: sum(items) / len(items) * (100 if key in {"success", "success_25m", "sdtw", "ndtw"} else 1)
                  for key, items in group["metrics"].items() if items}
        metric_fields = adapter_for(manifest, {"dataset": dataset}).metric_fields
        summaries.append({"dataset": dataset, "split": split, "planned": group["planned"], "valid": group["valid"],
                          "complete": group["valid"] == group["planned"], "provisional": values,
                          "metric_coverage": {key: len(group["metrics"].get(key, [])) for key in metric_fields},
                          "paper": {key: values.get(key) if group["valid"] == group["planned"] and
                                    len(group["metrics"].get(key, [])) == group["planned"] else None for key in metric_fields}})
    record = {"status": state["status"], "reason": state.get("reason"), "fingerprint": manifest["fingerprint"],
              "planned_executions": len(rows), "valid_executions": sum(row["status"] == "scored" for row in rows),
              "groups": summaries, "excluded_attempts": sum(a["status"] not in {"scored", "running"} for item in state["tasks"].values() for a in item["attempts"])}
    record["marginal_cases"] = [row for row in rows if row["marginal_20_25m"]]
    record["first_attempt_scored"] = sum(row["status"] == "scored" and row["attempts"] == 1 for row in rows)
    record["retried_scored"] = sum(row["status"] == "scored" and row["attempts"] > 1 for row in rows)
    record["stability_passed"] = bool(rows) and all(row["status"] == "scored" and row["cleanup_verified"] for row in rows)
    write_json(root / "metrics.json", record)
    write_json(root / "marginal_cases.json", record["marginal_cases"])
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Campaign report</title>',
            '<style>body{font:15px/1.6 system-ui;margin:28px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd}</style>',
            f'<h1>Campaign report</h1><p>{html.escape(state["status"])} · {record["valid_executions"]}/{len(rows)}</p>',
            '<details><summary>Coverage and provisional metrics</summary><pre>' + html.escape(json.dumps(summaries, ensure_ascii=False, indent=2)) + '</pre></details>',
            '<details><summary>20m &lt; final distance &lt; 25m</summary><pre>' + html.escape(json.dumps(record["marginal_cases"], ensure_ascii=False, indent=2)) + '</pre></details>',
            '<table><tr><th>Dataset / scene</th><th>Split / case</th><th>Status</th><th>Attempts</th><th>Metrics</th></tr>']
    for row in rows:
        label = html.escape(row["episode_id"])
        if row["review_path"] and Path(row["review_path"]).is_file():
            href = Path(os.path.relpath(Path(row["review_path"]).resolve(), root.resolve())).as_posix()
            label = f'<a href="{html.escape(href, quote=True)}">{label}</a>'
        body.append(f'<tr><td>{html.escape(row["dataset"])} / {html.escape(str(row["scene_id"]))}</td><td>{html.escape(row["split"])} / {label}</td><td>{row["status"]}</td><td>{row["attempts"]}</td><td>{html.escape(json.dumps(row["metrics"], ensure_ascii=False))}</td></tr>')
    body.append('</table></html>')
    atomic_write_text(root / "report.html", ''.join(body))
    with (root / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["task_key"])
        writer.writeheader()
        writer.writerows({**row, "metrics": json.dumps(row["metrics"], ensure_ascii=False)} for row in rows)
    return record
