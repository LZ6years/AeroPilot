"""Export existing failure reviews with embedded images for offline inspection.

This human-only projection reuses retained process HTML and the existing SR20
audit rule. It never changes receipts, scores, source snapshots or trajectories.
Run with the flyagent interpreter and --campaign PATH --output NEW_DIRECTORY.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.manual.blocked_success_audit import classify
from evaluation.benchmarks.core.portable_review import embed_images


def case_facts(result: dict) -> tuple[dict, str]:
    """Describe observed events without asserting a causal failure diagnosis."""
    counts = Counter()
    final = ""
    transcript = Path(result["run_dir"]) / "memory/agent_transcript.jsonl"
    for line in transcript.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        payload = event.get("payload") or {}
        if event["kind"] == "assistant_turn":
            counts["decision_turns"] += 1
            for call in payload.get("tool_calls", []):
                if call["tool_name"] in {"runtime.report_completed", "runtime.report_blocked"}:
                    final = call.get("args", {}).get("reason", "")
        elif event["kind"] == "tool_result" and payload.get("status") in {"error", "rejected", "collision"}:
            counts[payload["status"]] += 1
    return dict(counts), final


def export(campaign: Path, output: Path, dataset: str, budget: int) -> dict:
    """Build a separate failure-only index and portable per-case reports."""
    state = json.loads((campaign / "progress.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for key, item in state["tasks"].items():
        task = item["task"]
        if item["status"] != "scored" or task["dataset"] != dataset or task["image_budget"] != budget:
            continue
        result = item["result"]
        flags = classify(dataset, result)
        if flags["revised_sr20_success"] or any(result.get(field) for field in ("error", "infrastructure_error", "metric_error")):
            continue
        source = Path(result["report"]["path"])
        document, images = embed_images(source)
        counts, final = case_facts(result)
        distance = flags["final_distance_m"]
        ending = "主动报告 blocked" if flags["blocked"] else "报告完成但未通过 SR20"
        if distance is not None and distance <= 25:
            note = "最终距离处于 20～25m，属于停止边界案例；需核对目标身份、测距语义及停止依据。"
        elif flags["blocked"]:
            note = "模型主动终止搜索；需结合图像判断目标是否遗漏、是否重复搜索及是否存在可行后续动作。"
        else:
            note = "完成判断与评测终点不一致；需核对候选目标、路线及距离语义，不能仅凭终点误差认定具体原因。"
        panel = ("<section style='background:#fff4df;padding:20px;border:1px solid #d4ac5b'>"
                 "<h2>失败检查摘要</h2>"
                 f"<p>{ending}；最终 NE：{distance:.2f}m。≤25m：{'是' if flags['within_25m'] else '否'}。</p>"
                 f"<p>{note}</p><p>调用统计：{html.escape(json.dumps(counts, ensure_ascii=False))}</p>"
                 "<p>以下异常统计来自保留的有效任务轨迹；事件可能已恢复，不代表失败原因，也不包含被排除的整例重试。</p>"
                 f"<p>模型最终理由：{html.escape(final)}</p>"
                 "<p>图片已嵌入，可单独复制本 HTML。下方原始 success 保留原口径，未修改评分。</p></section>")
        document = document.replace("<h2>Outcome</h2>", panel + "<h2>Outcome</h2>", 1)
        filename = hashlib.sha256(key.encode()).hexdigest()[:16] + ".html"
        (output / filename).write_text(document, encoding="utf-8")
        rows.append(dict(task=key, episode=task["episode_id"], distance=distance,
                         ending=ending, within25=flags["within_25m"], images=images,
                         counts=counts, note=note, final_reason=final, report=filename,
                         source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    rows.sort(key=lambda row: row["distance"])
    table = "".join(f"<tr><td><a href='{row['report']}'>{html.escape(row['episode'])}</a></td>"
                    f"<td>{row['distance']:.2f}</td><td>{row['ending']}</td><td>{row['images']}</td></tr>" for row in rows)
    index = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>TravelUAV 失败案例</title>"
             "<style>body{font:16px system-ui;margin:30px}td,th{padding:10px;border:1px solid #ddd}table{border-collapse:collapse}a{color:#165eb3}</style>"
             f"<h1>{html.escape(dataset)} · {budget} 张图片预算 · 失败案例</h1>"
             f"<p>共 {len(rows)} 例。按修正 SR20 筛选：原成功以及无技术错误、明确 blocked 且终点≤20m 的案例均排除。"
             "按最终距离排序；≤25m 单独保留标记。所有分析为观察事实和待核查项，不是已证实的因果归因。</p>"
             "<p>每例 HTML 内嵌图片，建议使用 Edge 或 Chrome 打开。移动整套报告时请保留索引和案例 HTML 在同一目录。</p>"
             "<table><tr><th>案例</th><th>最终 NE(m)</th><th>结束方式</th><th>图片数</th></tr>" + table + "</table></html>")
    (output / "index.html").write_text(index, encoding="utf-8")
    manifest = dict(dataset=dataset, budget=budget, cases=rows, image_count=sum(r["images"] for r in rows))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"cases": len(rows), "images": manifest["image_count"], "index": str(output / "index.html")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="travel_uav", choices=["travel_uav"])
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(export(args.campaign, args.output, args.dataset, args.budget), ensure_ascii=False))
