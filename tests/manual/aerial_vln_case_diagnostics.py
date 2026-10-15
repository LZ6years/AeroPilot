"""Render linked plan and capability receipts from immutable captured flights.

This is a human report reader. It does not regenerate model context or decide
whether a natural-language route/landmark interpretation was correct.
"""
import html
import json
from pathlib import Path

from evaluation.benchmarks.core.report_io import atomic_write_text


def captured_turns(run):
    """Read every captured round, including rejected/corrected responses.

    Provenance is explicit: argument-derived descriptions are not model prose.
    Receipts are joined by call ID, never by position within a multi-call turn.
    """
    run = Path(run)
    receipts = {}
    transcript = run / "memory/agent_transcript.jsonl"
    if transcript.exists():
        for line in transcript.read_text(encoding="utf-8").splitlines():
            event = json.loads(line) if line.strip() else {}
            if event.get("kind") == "tool_result":
                payload = event["payload"]
                receipts.setdefault(payload.get("tool_call_id"), []).append(payload)
    turns = []
    paths = sorted(run.glob("debug/tool_loop/*/decision_*.json"))
    decisions = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    for path, decision in sorted(decisions, key=lambda item: item[1].get("decision_index", 0)):
        for model_round in decision.get("model_rounds", []):
            canonical = model_round.get("canonical_turn") or {}
            response = model_round.get("response") or {}
            message = response.get("message") or {}
            intent = canonical.get("public_intent", "")
            origin = canonical.get("public_intent_source")
            if not canonical:
                intent = message.get("content") or ""
                origin = "provider_content" if intent else ("missing_content" if response else "missing_response")
            turns.append({"decision_index": decision.get("decision_index"),
                "round_index": model_round.get("round_index"),
                "observation_id": decision.get("checkpoint_observation_id"),
                "source": str(path), "intent": intent, "intent_source": origin or "unknown",
                "calls": [{**call, "results": receipts.get(call.get("tool_call_id"), [])}
                          for call in canonical.get("tool_calls", [])],
                "canonical_turn_captured": bool(canonical)})
    return turns


def render_turns(turns, root, link):
    """Escape captured public content; never render provider reasoning fields."""
    labels = {"provider_content": "模型原文", "tool_arguments_fallback": "参数生成的动作说明（非模型原文）",
              "missing_response": "未捕获响应", "missing_content": "响应没有公开意图", "unknown": "来源未记录"}
    body = '<h2>逐回合公开意图与工具结果</h2>'
    for turn in turns:
        body += '<details open><summary>' + html.escape(
            f'决策 {turn["decision_index"]} / 回合 {turn["round_index"]} · 观测 {turn["observation_id"]} · '
            + labels.get(turn["intent_source"], turn["intent_source"])) + '</summary>'
        body += '<a href="' + html.escape(link(root, Path(turn["source"])), quote=True) + '">原始请求与响应</a>'
        intent = turn["intent"]
        body += '<pre>' + html.escape(intent if isinstance(intent, str) else json.dumps(intent, ensure_ascii=False)) + '</pre>'
        if not turn["canonical_turn_captured"]:
            body += '<p>未生成规范工具回合；请查看原始响应及补正记录。</p>'
        for call in turn["calls"]:
            body += '<pre>' + html.escape(json.dumps(call, ensure_ascii=False, indent=2)) + '</pre>'
            if not call["results"]:
                body += '<p>未捕获此调用的执行结果。</p>'
        body += '</details>'
    return body


def write_case_diagnostics(root, pairs, link):
    """Link every plan submission to its actual request and observation boundary."""
    for pair in pairs:
        result = pair.get("current")
        if not result:
            continue
        run = Path(result["run_dir"])
        stats = pair.get("current_stats") or {}
        requests = {}
        for path in run.glob("debug/tool_loop/*/decision_*.json"):
            decision = json.loads(path.read_text(encoding="utf-8"))
            for model_round in decision.get("model_rounds", []):
                for call in (model_round.get("canonical_turn") or {}).get("tool_calls", []):
                    requests[call["tool_call_id"]] = path
        def source(call_id):
            path = requests.get(call_id)
            return f'<a href="{link(root,path)}">原始请求</a>' if path else "请求关联未知"
        body = '<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;margin:30px}pre{white-space:pre-wrap}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}details{margin:18px 0}</style>'
        body += f'<h1>{pair["index"]}. {html.escape(pair["task"]["episode_id"])}：计划与工具时间线</h1>'
        body += '<p><a href="comparison.html">分组对比</a> · <a href="context_modules.html">真实上下文</a></p>'
        body += '<p>计划记录模型判断；提交成功不证明地标或路线判断正确。物理窗口与返回成功的动作独立统计。地点照片是历史视野，表面射线距离不等于目标水平距离。</p>'
        body += render_turns(captured_turns(run), root, link)
        body += '<h2>计划提交</h2>'
        if not stats.get("plan_timeline"):
            body += '<p>没有成功计划提交；不能评价计划推进是否及时。原始拒绝仍列在下方。</p>'
        for entry in stats.get("plan_timeline", []):
            plan = (entry.get("data") or {}).get("plan", {})
            body += f'<details open><summary>revision {plan.get("revision", "未知")}；观测 {html.escape(str(entry.get("observation_id")))}；已记录物理窗口 {entry.get("physical_windows_before", "未知")}</summary>'
            body += source(entry.get("tool_call_id"))
            body += '<pre>' + html.escape(json.dumps(entry, ensure_ascii=False, indent=2)) + '</pre></details>'
        body += '<h2>结束时未完成步骤与间隔</h2><pre>' + html.escape(json.dumps({key: stats.get(key) for key in (
            "unfinished_final_steps", "physical_windows_between_updates", "physical_windows_after_last_update",
            "flight_actions_between_updates", "flight_actions_after_last_update", "plan_rejections")}, ensure_ascii=False, indent=2)) + '</pre>'
        body += '<h2>地点图实际版本</h2><ul>'
        for media in (stats.get("context_images") or {}).get("map_versions", []):
            if media.get("asset_ref"):
                body += f'<li><a href="{link(root,run/media["asset_ref"])}">{html.escape(media.get("frame_id") or media.get("selected_frame_id") or media["sha256"])}</a></li>'
        body += '</ul><h2>像素测距、地点保存与查阅</h2>'
        for receipt in stats.get("capability_timeline", []):
            body += '<details><summary>' + html.escape(receipt["tool"] + ' · ' + receipt["status"]) + '</summary>' + source(receipt.get("tool_call_id"))
            for item in receipt.get("model_inputs") or []:
                ref = item.get("media_ref") or {}
                if ref.get("ref"):
                    body += f' · <a href="{link(root,run/ref["ref"])}">标记图／历史照片／地图</a>'
            body += '<pre>' + html.escape(json.dumps(receipt, ensure_ascii=False, indent=2)) + '</pre></details>'
        body += '<h2>动作与观测</h2><pre>' + html.escape(json.dumps(stats.get("action_observation_timeline"), ensure_ascii=False, indent=2)) + '</pre>'
        atomic_write_text(root / f'case_diagnostics_{pair["index"]:02d}.html', body)
