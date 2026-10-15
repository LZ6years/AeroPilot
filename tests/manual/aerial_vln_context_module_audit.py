"""Audit captured modules and replay immutable historical transcript prefixes.

Candidate previews are offline projections, never actual provider requests.
Module character counts exclude image bytes; actual tokens belong to API usage.
"""
import copy
import hashlib
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from src.agent.context.media import MediaHydrator
from src.agent.context.projection import HistoryProjector
from src.agent.context.request_assembler import RequestAssembler
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult, TranscriptEvent
from src.tools.diagnostics import legacy_result_diagnostic
from src.tools.registry import default_registry
from src.plugins import PluginRegistry


def text_parts(message):
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    return [p["text"] for p in content or [] if p.get("type") == "text"]


def classify(text):
    for label in ("CURRENT TASK STATE", "STATE CHECKPOINT", "TOOL EVIDENCE", "CONTEXT IMAGE", "HISTORY CAPSULE", "INSTRUCTION CONTEXT"):
        if label in text:
            return label
    return "other"


def module_counts(messages):
    """Split actual serialized fields by module; these are character counts."""
    counts = Counter()
    for message in messages:
        for text in text_parts(message):
            label = classify(text)
            start = text.find("{")
            try:
                payload = json.loads(text[start:]) if start >= 0 else None
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("fragments"), list):
                for fragment in payload["fragments"]:
                    counts["dynamic." + str(fragment.get("name", "unknown"))] += len(json.dumps(fragment, ensure_ascii=False))
                metadata = {key: value for key, value in payload.items() if key != "fragments"}
                counts["checkpoint_metadata"] += len(json.dumps(metadata, ensure_ascii=False))
            elif isinstance(payload, dict) and isinstance(payload.get("static_module_guidance"), list):
                for guidance in payload["static_module_guidance"]:
                    counts["static." + str(guidance.get("name", "unknown"))] += len(json.dumps(guidance, ensure_ascii=False))
                metadata = {key: value for key, value in payload.items() if key != "static_module_guidance"}
                counts["instruction_metadata"] += len(json.dumps(metadata, ensure_ascii=False))
            else:
                counts["tool_result" if message["role"] == "tool" else label] += len(text)
    return dict(counts)


def preview(messages):
    """Keep readable request text and hashes, without duplicating image bytes."""
    result = copy.deepcopy(messages)
    for message in result:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    part["image_url"] = {"url": "[image bytes omitted from offline preview]",
                        "data_url_sha256": hashlib.sha256(url.encode()).hexdigest()}
    return result


def events_from_records(records):
    events = []
    for record in records:
        p = copy.deepcopy(record["payload"])
        kind = record["kind"]
        if kind == "assistant_turn":
            p = AssistantTurn(p["turn_id"], p.get("public_intent", ""),
                [AgentToolCall(**call) for call in p.get("tool_calls", [])],
                p.get("public_intent_source", "provider_content"))
        elif kind == "tool_result":
            # Only this offline view adapts legacy diagnostics.
            p["diagnostic"] = legacy_result_diagnostic(p)
            p = ToolResult(**{key: value for key, value in p.items() if key in ToolResult.__dataclass_fields__})
        events.append(TranscriptEvent(kind, p))
    return tuple(events)


def offline_audit_registry(current_config, historical_config):
    """Read old native calls without re-exposing removed tools online."""
    registry = default_registry(PluginRegistry.from_config(current_config))
    historical = default_registry(PluginRegistry.from_config(historical_config))
    for name in historical.names():
        if name not in registry.names():
            registry.register(historical.get(name))
    return registry


def historical_calls_match(original, projected, *, compressed):
    """Compression can omit calls; every retained call must stay exact/in order."""
    if not compressed:
        return original == projected
    remaining = iter(projected)
    return all(any(candidate == call for candidate in remaining) for call in original)


def replay_history(root, previous_root, write, digest, expected_requests=None):
    previous = json.loads((previous_root / "progress.json").read_text(encoding="utf-8"))
    registry = offline_audit_registry(root / "source/configs/plugins.json", previous_root / "source/configs/plugins.json")
    records = []; failures = []; unchanged = []
    for row in previous["cases"]:
        result = row["attempts"][-1]["result"]
        run = Path(result["run_dir"])
        transcript = run / "memory/agent_transcript.jsonl"
        before = digest(transcript)
        raw = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
        events = events_from_records(raw)
        turn_indices = {e.payload.turn_id: i for i, e in enumerate(events) if e.kind == "assistant_turn"}
        projector = HistoryProjector(assembler=RequestAssembler(MediaHydrator(base_dir=run)), media_policy={
            "max_request_images": 20, "max_tool_evidence_images": 4, "keep_recent_observations": 3,
            "pin_instruction_start": True, "pin_current": True})
        for path in sorted(run.glob("debug/tool_loop/*/decision_*.json")):
            request_before = digest(path)
            decision = json.loads(path.read_text(encoding="utf-8"))
            for model_round in decision.get("model_rounds", []):
                canonical = model_round.get("canonical_turn") or {}
                turn_id = canonical.get("turn_id")
                if turn_id not in turn_indices:
                    # A transport failure captured a real request but produced
                    # no assistant event. Do not invent its transcript boundary
                    # or silently drop it from the historical request count.
                    if not canonical and not model_round.get("response"):
                        from tests.manual.aerial_vln_plan_context_acceptance import request_checks
                        reasons, _ = request_checks(model_round)
                        original = model_round["request"]["messages"]
                        entry = {"episode_id": row["task"]["episode_id"], "decision_id": decision["decision_id"],
                            "source": str(path), "source_sha256": request_before, "transcript": str(transcript),
                            "transcript_sha256": before, "boundary": None, "passed": not reasons,
                            "reasons": reasons, "projection_status": "unavailable_no_response_boundary",
                            "original_module_chars": module_counts(original), "candidate_module_chars": {},
                            "candidate_audit": {}, "actual_usage": None}
                        preview_path = root / "context_replay" / f"{len(records)+1:04d}.json"
                        write(preview_path, {**entry, "notice": "Captured request checked unchanged; no response or offline projection available.",
                            "original_messages": preview(original), "candidate_messages": []})
                        records.append({**entry, "preview": str(preview_path)})
                        failures.extend({"source": str(path), "reason": reason} for reason in reasons)
                        continue
                    failures.append({"source": str(path), "reason": "Cannot locate canonical transcript boundary"})
                    continue
                boundary = turn_indices[turn_id]
                projection = projector.project_snapshot(events[:boundary], registry)
                original = model_round["request"]["messages"]
                static = []
                for message in original:
                    if message["role"] == "user":
                        break
                    static.append(copy.deepcopy(message))
                current = model_round.get("current_task_state") or decision.get("current_task_state") or {}
                candidate = [*static, *projection.messages, RequestAssembler.current_task_state_message(current)]
                serialized = json.dumps(candidate, ensure_ascii=False)
                headers = [part for m in candidate for part in text_parts(m) if part.startswith("STATE CHECKPOINT")]
                calls = {c["id"] for m in candidate for c in m.get("tool_calls", [])}
                reasons = []
                original_calls = [call for m in original for call in m.get("tool_calls", [])]
                projected_calls = [call for m in candidate for call in m.get("tool_calls", [])]
                compressed = any(part.startswith("HISTORY CAPSULE\n") for message in original for part in text_parts(message))
                if not historical_calls_match(original_calls, projected_calls, compressed=compressed):
                    reasons.append("native historical calls changed")
                if serialized.count("CURRENT TASK STATE\\n") != 1:
                    reasons.append("latest task state is not unique")
                if any(not re.match(r"STATE CHECKPOINT (CURRENT|HISTORY) ", h) for h in headers):
                    reasons.append("checkpoint has no time role")
                current_id = decision.get("checkpoint_observation_id")
                if any(("STATE CHECKPOINT CURRENT " in h) != (f"observation_id={current_id}." in h) for h in headers):
                    reasons.append("checkpoint time role differs from captured current observation")
                if projection.audit["total_image_count"] > 20:
                    reasons.append("image budget exceeded")
                for message in candidate:
                    if message["role"] == "tool" and message.get("tool_call_id") in calls:
                        if "requested_args" in json.loads(message["content"]):
                            reasons.append("duplicate requested args")
                entry = {"episode_id": row["task"]["episode_id"], "decision_id": decision["decision_id"],
                    "source": str(path), "source_sha256": request_before, "transcript": str(transcript),
                    "transcript_sha256": before, "boundary": boundary, "passed": not reasons,
                    "projection_status": "offline_uncompressed_prefix", "original_was_compressed": compressed,
                    "reasons": reasons, "original_module_chars": module_counts(original),
                    "candidate_module_chars": module_counts(candidate), "candidate_audit": projection.audit,
                    "actual_usage": (model_round.get("response") or {}).get("usage")}
                preview_path = root / "context_replay" / f"{len(records)+1:04d}.json"
                write(preview_path, {**entry, "notice": "Offline candidate projection; image bytes omitted. Not a captured or executed request.",
                    "original_messages": preview(original), "candidate_messages": preview(candidate)})
                entry["preview"] = str(preview_path)
                records.append(entry)
                failures.extend({"source": str(path), "reason": reason} for reason in reasons)
            if digest(path) != request_before:
                failures.append({"source": str(path), "reason": "Original request changed"})
        same = digest(transcript) == before
        unchanged.append({"path": str(transcript), "sha256": before, "unchanged": same})
        if not same:
            failures.append({"source": str(transcript), "reason": "Original transcript changed"})
    if expected_requests is not None and len(records) != expected_requests:
        failures.append({"reason": f"Expected {expected_requests} historical requests, replayed {len(records)}"})
    audit = {"passed": not failures, "requests": len(records), "failures": failures,
             "original_transcripts": unchanged, "records": records,
             "notice": "Offline projection audit. Per-module sizes are characters; token totals are captured API usage only."}
    write(root / "historical_request_replay.json", audit)
    return audit


def write_module_report(root, pairs, historical, write):
    historical = copy.deepcopy(historical)
    for entry in (historical or {}).get("records", []):
        saved = json.loads(Path(entry["preview"]).read_text(encoding="utf-8"))
        entry["original_module_chars"] = module_counts(saved["original_messages"])
        entry["candidate_module_chars"] = module_counts(saved["candidate_messages"])
    records = []
    for pair in pairs:
        result = pair.get("current")
        if not result:
            continue
        for path in sorted(Path(result["run_dir"]).glob("debug/tool_loop/*/decision_*.json")):
            decision = json.loads(path.read_text(encoding="utf-8"))
            for model_round in decision.get("model_rounds", []):
                from tests.manual.aerial_vln_plan_context_acceptance import delivered_media
                records.append({"episode_id": pair["task"]["episode_id"], "source": str(path),
                    "cohort": pair.get("cohort"), "request_media": delivered_media(model_round),
                    "module_chars": module_counts(model_round["request"]["messages"]),
                    "actual_usage": (model_round.get("response") or {}).get("usage"),
                    "projection": model_round.get("history_projection")})
    data = {"notice": "实际新请求与离线历史投影分开；模块字符数不等于 token。原始请求保持不变。模块字段按 JSON 序列化计字符，与完整消息总长度并非严格可加。",
            "report_reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "actual_current_requests": records, "historical_replay": historical,
            "image_cost_notice": "图像成本包含在整轮实际 API usage 中；模块字符数及图像数量仅作诊断，不据此分摊 token 或声称某个工具的因果成本。",
            "diagnostics": [{"episode_id": p["task"]["episode_id"], "stats": p.get("current_stats")} for p in pairs]}
    write(root / "context_modules.json", data)
    body = '<!doctype html><meta charset="utf-8"><title>上下文模块审计</title><style>body{font:15px system-ui;margin:30px}pre{white-space:pre-wrap}li{margin:8px}</style>'
    body += '<h1>上下文模块审计</h1><p>'+html.escape(data["notice"])+'</p><p><a href="comparison.html">导航对比</a> · <a href="context_modules.json">完整模块、诊断及 API usage</a></p>'
    historical_count = str(historical["requests"]) if historical and historical.get("requests") is not None else "未知数量的"
    body += '<h2>'+historical_count+' 份历史请求离线投影</h2><p>'+html.escape(str((historical or {}).get("passed", "未运行")))+'</p><ul>'
    for entry in (historical or {}).get("records", []):
        body += '<li><a href="context_replay/'+Path(entry["preview"]).name+'">'+html.escape(entry["episode_id"]+' '+entry["decision_id"])+'</a> '+str(entry["passed"])+'</li>'
    body += '</ul><h2>实际新请求</h2><pre>'+html.escape(json.dumps(records, ensure_ascii=False, indent=2))+'</pre>'
    body += '<h2>原始请求文件</h2><ul>'
    for entry in records:
        target = quote(os.path.relpath(entry["source"], root).replace("\\", "/"), safe="/:.")
        body += '<li><a href="'+target+'">'+html.escape(entry["episode_id"]+' '+Path(entry["source"]).name)+'</a></li>'
    body += '</ul>'
    (root / "context_modules.html").write_text(body, encoding="utf-8")
