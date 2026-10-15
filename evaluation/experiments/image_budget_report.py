"""Read-only paired metrics/report projection of the authoritative task queue.

The controller owns progress. This module caches analysis of immutable finished
attempts, keeps infrastructure attempts separate, and never selects a better
navigation result or reduces the formal fixed-cohort metric denominator.
"""
from __future__ import annotations

from collections import Counter
import csv
import html
import hashlib
import io
import json
import math
import os
from pathlib import Path

from evaluation.benchmarks.core.report_io import atomic_write_text, write_json
from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue
from evaluation.experiments.compression_cost import effective_planner_calls, effective_totals


def reader_hashes():
    """Identity derived readers separately from immutable navigation source."""
    import sys
    paths = {"image_budget_report.py": Path(__file__),
             "compression_cost.py": Path(sys.modules[effective_planner_calls.__module__].__file__)}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}

ACCOUNTING_NOTE = (
    "AerialVLN-S 展示 SR 与 nDTW（所有有效任务的原始 nDTW 均值×100，包含导航失败）；TravelUAV 展示 SR 与 NE。"
    "SR 沿用 20m 成功判定；≤25m 仅标记有效任务的结束距离，AerialVLN-S 使用 XY 距离，TravelUAV 使用三维距离，不等同于成功。"
    "各配置固定使用 AerialVLN-S 与 TravelUAV 各 100 例。正常导航失败保留；基础设施错误、无效任务尝试和失败模型请求尝试单独审计。"
    "输入/输出 token（千）和有效 LLM 调用次数为 200 例每任务平均；传输重试后成功只计一次。"
    "决策时延（秒/请求）=有效决策请求最终成功尝试耗时之和/有效决策请求数，采用请求加权平均；"
    "包含网络和服务端等待、生成及返回，排除失败尝试、退避等待、工具、场景启动和摘要。"
    "第一阶段关闭压缩，摘要调用为零。缺失 usage 或成功尝试时长时，相应正式数值留空并报告覆盖率。"
    "实验中途更新过错误恢复条件；各次尝试记录 recovery_revision，已有有效结果保留，不能视为完全相同的执行条件。"
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def number(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def final_proximity(dataset, metrics):
    """Report endpoint proximity separately from the evaluator's success flag."""
    field = "distance_to_goal" if dataset == "aerial_vln" else "navigation_error"
    distance = metrics.get(field)
    if type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0:
        return {"final_distance_m": None, "within_25m": None}
    return {"final_distance_m": distance, "within_25m": distance <= 25}


def write_csv(path, rows):
    if not rows:
        atomic_write_text(path, "")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue(), encoding="utf-8-sig")


def analyze_attempt(root, task_key, attempt):
    """Cache only a finished attempt, never a mutable active request stream."""
    directory = Path(attempt.get("directory", ""))
    revision = hashlib.sha256(json.dumps(reader_hashes(), sort_keys=True).encode()).hexdigest()[:16]
    cache = root / "analysis" / revision / task_key / f'attempt_{attempt["number"]:02d}.json'
    if cache.exists():
        return read(cache)
    result = attempt.get("result") or {}
    run = result.get("run_dir")
    # A crash before result persistence can still leave auditable partial calls.
    if not run and directory.is_dir():
        runs = list((directory / "runs").glob("*"))
        run = str(runs[0]) if len(runs) == 1 else None
    requests = effective_planner_calls(run) if run else []
    compaction_files = list((Path(run) / "debug/history_compaction_calls").glob("*.json")) if run else []
    value = {"requests": requests, "totals": effective_totals(requests), "run_dir": run,
             "summary_audit_files": len(compaction_files)}
    write_json(cache, value)
    return value


def aggregate_group(rows, requests, budget):
    """Paper metrics require all 100+100 scores; provisional means are labeled."""
    group = [r for r in rows if r["image_budget"] == budget]
    valid = [r for r in group if r["status"] == "scored"]
    aerial = [r for r in valid if r["dataset"] == "aerial_vln"]
    travel = [r for r in valid if r["dataset"] == "travel_uav"]
    calls = [r for r in requests if r["image_budget"] == budget and r["retained_task_execution"]]
    totals = effective_totals(calls)
    metrics = {
        "aerial_sr": 100 * sum(r["success"] for r in aerial) / len(aerial) if aerial else None,
        "aerial_ndtw": 100 * sum(r["ndtw"] for r in aerial) / len(aerial) if aerial and all(r.get("ndtw") is not None for r in aerial) else None,
        "travel_sr": 100 * sum(r["success"] for r in travel) / len(travel) if travel else None,
        "travel_ne": sum(r["ne"] for r in travel) / len(travel) if travel and all(r["ne"] is not None for r in travel) else None,
        "input_k": totals["input_tokens"] / len(valid) / 1000 if valid and totals["input_tokens"] is not None else None,
        "output_k": totals["output_tokens"] / len(valid) / 1000 if valid and totals["output_tokens"] is not None else None,
        "calls_per_task": totals["calls"] / len(valid) if valid else None,
        "decision_latency_seconds": totals["decision_latency_seconds"],
    }
    complete = len(aerial) == 100 and len(travel) == 100 and len(group) == 200
    paper = metrics.copy() if complete else {k: None for k in metrics}
    # Navigation metrics can close per dataset; pooled costs still need 200.
    for prefix, dataset, valid_rows in (("aerial", "aerial_vln", aerial), ("travel", "travel_uav", travel)):
        if len(valid_rows) == 100 and sum(r["dataset"] == dataset for r in group) == 100:
            paper.update({key: value for key, value in metrics.items() if key.startswith(prefix + "_")})
    estimates = [r["peak_estimated_input_tokens"] for r in valid if r["peak_estimate_coverage_complete"]]
    distribution = {"below_24k": sum(v < 24000 for v in estimates),
                    "24k_to_40k": sum(24000 <= v < 40000 for v in estimates),
                    "at_least_40k": sum(v >= 40000 for v in estimates),
                    "unknown_cases": len(valid) - len(estimates)}
    accepted = [r for r in calls if r["effective"]]
    fulfillment = sum(r.get("total_images") == budget for r in accepted) / len(accepted) if accepted else None
    return {"image_budget": budget, "complete": complete, "valid_aerial": len(aerial), "valid_travel": len(travel),
            "status_counts": dict(Counter(r["status"] for r in group)), "provisional": metrics,
            "paper": paper, "cost_audit": totals,
            "image_budget_fulfillment_rate": fulfillment,
            "future_prefetch_distribution": distribution,
            "future_additional_executions_upper_bound": distribution["24k_to_40k"] + 2 * (distribution["at_least_40k"] + distribution["unknown_cases"])}


def paper_table(groups):
    header = ('<table><thead><tr><th rowspan="2">配置</th><th colspan="2">AerialVLN-S</th><th colspan="2">TravelUAV</th>'
              '<th rowspan="2">输入 token<br>（千/任务）</th><th rowspan="2">输出 token<br>（千/任务）</th>'
              '<th rowspan="2">LLM 调用<br>（次/任务）</th><th rowspan="2">决策时延<br>（秒/请求）</th></tr>'
              '<tr><th>SR ↑ (%)</th><th>nDTW ↑ (%)</th><th>SR ↑ (%)</th><th>NE ↓ (m)</th></tr></thead><tbody>')
    fields = ("aerial_sr", "aerial_ndtw", "travel_sr", "travel_ne", "input_k", "output_k", "calls_per_task", "decision_latency_seconds")
    rows = [f'<tr><td>图像预算：{g["image_budget"]} 张</td>' + ''.join(f'<td>{number(g["paper"][k])}</td>' for k in fields) + '</tr>' for g in groups]
    return header + ''.join(rows) + '</tbody></table>'


def charts(groups):
    """Small exportable SVG charts; incomplete cohorts are explicitly provisional."""
    panels = []
    for title, key, unit in (("AerialVLN-S SR", "aerial_sr", "%"), ("TravelUAV SR", "travel_sr", "%"),
                             ("输入 token / 任务", "input_k", "千"), ("决策时延 / 请求", "decision_latency_seconds", "秒")):
        data = [g["provisional"][key] for g in groups]
        ceiling = max([v for v in data if v is not None] or [1]) or 1
        panel = ['<svg xmlns="http://www.w3.org/2000/svg" width="290" height="210" role="img">',
                 f'<text x="12" y="22" font-size="14" fill="#17324f">{title}（{unit}）</text>']
        for index, (g, v) in enumerate(zip(groups, data)):
            height = (v / ceiling * 125) if v is not None else 0
            x = 25 + index * 90
            panel.extend([f'<rect x="{x}" y="{175-height}" width="50" height="{height}" rx="3" fill="#3586ac"/>',
                          f'<text x="{x}" y="{165-height}" font-size="12">{number(v)}</text>',
                          f'<text x="{x}" y="198" font-size="12">{g["image_budget"]} 张</text>'])
        panel.append('</svg>')
        panels.append(''.join(panel))
    return '<h2>临时统计图（未齐全时仅供监控）</h2>' + ''.join(panels)


def write_report(output, state=None):
    """Refresh human outputs without modifying progress, raw audits or scores."""
    root = Path(output)
    manifest = read(root / "manifest.json")
    state = state or read(root / "progress.json")
    cases, requests, attempts = [], [], []
    for key, item in state["tasks"].items():
        task = item["task"]
        retained = next((a for a in item["attempts"] if a.get("status") == "scored" and a.get("valid_score")
                         and a.get("cleanup_verified") and not ParallelRecoveryQueue.infrastructure_record(a.get("result") or item.get("result") or {})), None)
        result = item.get("result") if item["status"] == "scored" and retained else None
        status = "retry_pending" if item["status"] == "scored" and not result else item["status"]
        metrics = (result or {}).get("metrics") or {}
        row = {k: task[k] for k in ("dataset", "image_budget", "split", "episode_id", "scene_id", "difficulty", "historical_success_anchor")}
        row.update(task_key=key, status=status, success=int(bool(metrics.get("success"))) if result else None,
                   ndtw=metrics.get("ndtw"), ne=metrics.get("navigation_error"), attempts=len(item["attempts"]),
                   retained_attempt=retained["number"] if retained else None, run_dir=(result or {}).get("run_dir"),
                   review_path=None, peak_estimated_input_tokens=None, peak_estimate_coverage_complete=False)
        row.update(final_proximity(task["dataset"], metrics))
        if result and result.get("run_dir"):
            row["review_path"] = str(Path(result["run_dir"]).parents[1] / "evaluation/cases" / task["episode_id"] / "review.html")
        for attempt in item["attempts"]:
            if attempt["status"] == "running":
                continue
            analysis = analyze_attempt(root, key, attempt)
            keep = bool(retained and retained["number"] == attempt["number"])
            for call in analysis["requests"]:
                requests.append({**call, **{k: row[k] for k in ("dataset", "image_budget", "split", "episode_id")},
                                 "attempt_number": attempt["number"], "retained_task_execution": keep,
                                 "recovery_revision": attempt.get("recovery_revision", "original")})
            attempts.append({"task_key": key, "dataset": task["dataset"], "image_budget": task["image_budget"],
                             "attempt_number": attempt["number"], "status": attempt["status"], "included": keep,
                             "cleanup_verified": attempt.get("cleanup_verified"), "directory": attempt.get("directory"),
                             "recovery_revision": attempt.get("recovery_revision", "original"),
                             **analysis["totals"]})
            if keep:
                row.update(analysis["totals"])
                row["summary_calls"] = analysis["summary_audit_files"]
        cases.append(row)
    groups = [aggregate_group(cases, requests, b) for b in manifest["image_budgets"]]
    overlay = read(root / "parallel/execution_overlay.json") if (root / "parallel/execution_overlay.json").exists() else {}
    policy = overlay.get("scheduling_policy", {})
    phases = []
    for budget in manifest["image_budgets"]:
        for dataset in policy.get("dataset_order", ("aerial_vln", "travel_uav")):
            rows = [r for r in cases if r["image_budget"] == budget and r["dataset"] == dataset]
            good = [r for r in rows if r["status"] == "scored"]
            totals = effective_totals([r for r in requests if r["image_budget"] == budget and r["dataset"] == dataset and r["retained_task_execution"]])
            group = next(g for g in groups if g["image_budget"] == budget)
            prefix = "aerial_" if dataset == "aerial_vln" else "travel_"
            phases.append({"image_budget": budget, "dataset": dataset, "planned": len(rows), "valid": len(good),
                           "complete": len(good) == len(rows) == 100,
                           "counts": dict(Counter(r["status"] for r in rows)),
                           "within_25m_count": sum(r["within_25m"] is True for r in good),
                           "final_distance_coverage": sum(r["final_distance_m"] is not None for r in good),
                           "navigation": {k: v for k, v in group["provisional"].items() if k.startswith(prefix)},
                           "cost_audit": totals})
    current_phase = next((p for p in phases if not p["complete"]), None) if policy.get("dataset_order") else None
    live = read(root / "parallel/status.json") if (root / "parallel/status.json").exists() else {}
    timing = read(root / "parallel/progress_check.json") if (root / "parallel/progress_check.json").exists() else {}
    excluded = [r for r in requests if not r["retained_task_execution"] or not r["effective"]]
    report = {"status": state["status"], "reason": state.get("reason"), "fingerprint": manifest["fingerprint"],
              "report_reader_hashes": reader_hashes(),
              "logging_profile": read(root / "parallel/execution_overlay.json").get("logging_profile") if (root / "parallel/execution_overlay.json").exists() else None,
              "planned_executions": len(cases), "valid_executions": sum(r["status"] == "scored" for r in cases),
              "groups": groups, "accounting_note": ACCOUNTING_NOTE,
              "phases": phases, "current_phase": current_phase,
              "runtime": {"lanes": live.get("lanes", {}), "updated_at": live.get("updated_at"), "timing": timing},
              "excluded_attempts": sum(not a["included"] for a in attempts),
              "recovery_revisions": dict(Counter(a["recovery_revision"] for a in attempts)),
              "excluded_response_records": len(excluded),
              "excluded_known_input_tokens": sum(r["input_tokens"] or 0 for r in excluded),
              "excluded_known_output_tokens": sum(r["output_tokens"] or 0 for r in excluded),
              "selection_bias_note": "预先选定含历史成功锚点与困难任务的调参子集；结果只代表该固定子集，不宣称为全数据集随机估计。",
              "next_stage": "等待用户选择图片预算；不自动启动压缩实验。"}
    write_json(root / "metrics.json", report)
    write_csv(root / "per_case.csv", cases)
    write_csv(root / "per_request.csv", requests)
    write_csv(root / "attempt_audit.csv", attempts)
    table = paper_table(groups)
    phase_label = (f'{current_phase["image_budget"]} 张 / {current_phase["dataset"]}：{current_phase["valid"]}/{current_phase["planned"]}'
                   if current_phase else "按当前调度策略运行")
    progress_text = (f'当前阶段：{phase_label}；待重试：{sum(r["status"] == "retry_pending" for r in cases)}；'
                     f'预计剩余：{number(timing.get("remaining_hours_estimate"))} 小时。估算包含错误重试与停机；不足 4 例新增有效结果时留空。')
    atomic_write_text(root / "paper_table.html", table + '<p>' + ACCOUNTING_NOTE + '</p>')
    md = '# 第一阶段：图片预算实验\n\n' + f'状态：{state["status"]}；有效结果：{report["valid_executions"]}/600。\n\n' + progress_text + '\n\n' + table + '\n\n' + ACCOUNTING_NOTE + '\n\n' + report["selection_bias_note"] + '\n\n' + report["next_stage"]
    atomic_write_text(root / "report.md", md)
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>图片预算实验</title>',
            '<style>body{font:15px/1.6 system-ui;margin:28px;color:#17324f}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #dce1e7;padding:8px 12px}th{background:#f5f7f9}tbody tr:nth-child(even){background:#fafbfc}tr[data-near25="1"] td{background:#e6f4ed}a{color:#1765b1}input,select{padding:6px;margin:5px}details{margin:20px 0}.scroll{overflow:auto}svg{max-width:100%}</style>',
            f'<h1>图片预算实验</h1><p id="progress">状态：{html.escape(state["status"])} · 有效结果 {report["valid_executions"]}/600 · 两路并行 · 压缩关闭</p>',
            '<p>' + html.escape(progress_text) + '</p>',
            '<p>各数据集满 100 例后填写导航指标；每档满 200 例后填写汇总成本。</p>', table, '<p>' + ACCOUNTING_NOTE + '</p>',
            '<details open><summary>两路活动、阶段统计与耗时估计</summary><pre>' + html.escape(json.dumps({"lanes": live.get("lanes", {}), "phases": [{k: v for k, v in p.items() if k != "cost_audit"} for p in phases], "timing": timing}, ensure_ascii=False, indent=2)) + '</pre></details>',
            '<p>' + report["selection_bias_note"] + '</p>',
            '<p><a href="per_case.csv">逐任务 CSV</a> · <a href="per_request.csv">逐请求 CSV</a> · <a href="attempt_audit.csv">错误与重试审计</a> · <a href="metrics.json">机器可读指标</a> · <a href="paper_table.html">论文 HTML 表格</a></p>',
            '<details><summary>运行中临时统计、计时覆盖率与第二阶段分界</summary><pre>' + html.escape(json.dumps(groups, ensure_ascii=False, indent=2)) + '</pre></details>',
            charts(groups),
            f'<p>排除任务尝试：{report["excluded_attempts"]}；排除响应记录：{len(excluded)}；已知排除输入/输出 token：{report["excluded_known_input_tokens"]}/{report["excluded_known_output_tokens"]}。失败请求无 usage 时实际费用不可恢复，不能视为零。</p>',
            '<h2>任务比较与原始证据</h2><select id="dataset"><option value="">全部数据集</option><option>aerial_vln</option><option>travel_uav</option></select><select id="difficulty"><option value="">全部难度</option><option>high</option><option>medium</option><option>hard</option><option>easy</option></select><select id="budget"><option value="">全部预算</option><option>10</option><option>20</option><option>30</option></select><select id="split"><option value="">全部划分</option>' + ''.join(f'<option>{html.escape(s)}</option>' for s in manifest["counts"]) + '</select><label><input id="anchor" type="checkbox">仅历史成功锚点</label><label><input id="near25" type="checkbox">仅结束距离 ≤25m</label><input id="search" placeholder="任务 ID 或状态"><div class="scroll"><table id="cases"><thead><tr><th>数据集 / 任务</th><th>预算</th><th>划分 / 难度</th><th>状态</th><th>SR</th><th>nDTW (%) / NE (m)</th><th>结束距离 (m)</th><th>≤25m 标记</th><th>调用 / 输入千 token</th><th>时延秒/请求</th><th>上下文峰值</th><th>尝试</th></tr></thead><tbody>']
    for row in sorted(cases, key=lambda r: (r["dataset"], r["episode_id"], r["image_budget"])):
        label = html.escape(row["episode_id"])
        if row["review_path"] and Path(row["review_path"]).exists():
            # A migrated campaign may retain historical paths through a
            # directory junction; resolve both sides before computing its link.
            target = html.escape(Path(os.path.relpath(Path(row["review_path"]).resolve(), root.resolve())).as_posix(), quote=True)
            label = f'<a href="{target}">{label}</a>'
        body.append(f'<tr data-near25="{int(row["within_25m"] is True)}" data-dataset="{row["dataset"]}" data-difficulty="{row["difficulty"]}" data-split="{row["split"]}" data-budget="{row["image_budget"]}" data-anchor="{int(row["historical_success_anchor"])}"><td>{row["dataset"]}<br>{label}</td><td>{row["image_budget"]}</td><td>{row["split"]}<br>{row["difficulty"]}</td><td>{row["status"]}</td><td>{number(row["success"],0)}</td><td>{number(100 * row["ndtw"] if row["dataset"] == "aerial_vln" and row["ndtw"] is not None else row["ne"])}</td><td>{number(row["final_distance_m"])}</td><td>{"≤25m" if row["within_25m"] is True else (">25m" if row["within_25m"] is False else "--")}</td><td>{row.get("calls","--")} / {number(row.get("input_tokens")/1000 if row.get("input_tokens") is not None else None)}</td><td>{number(row.get("decision_latency_seconds"))}</td><td>{row["peak_estimated_input_tokens"] or "--"}</td><td>{row["attempts"]}</td></tr>')
    body.append('</tbody></table></div><script>function filter(){for(const r of document.querySelectorAll("#cases tbody tr")){let ok=true;for(const k of ["dataset","difficulty","budget","split"]){const v=document.getElementById(k).value;if(v&&r.dataset[k]!==v)ok=false;}if(document.getElementById("near25").checked&&r.dataset.near25!=="1")ok=false;if(document.getElementById("anchor").checked&&r.dataset.anchor!=="1")ok=false;if(!r.textContent.toLowerCase().includes(document.getElementById("search").value.toLowerCase()))ok=false;r.hidden=!ok;}}for(const e of document.querySelectorAll("select,input"))e.addEventListener("input",filter);</script><p>' + report["next_stage"] + '</p></html>')
    atomic_write_text(root / "report.html", ''.join(body))
    return report
