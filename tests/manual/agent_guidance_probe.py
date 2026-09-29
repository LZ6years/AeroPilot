"""Paired, fixed-history guidance probe: two new requests, zero tool execution.

The original capture stays immutable. The revised variant changes tool prose,
the mission prompt, knowledge catalogs and benchmark protocol contexts. Saved
assistant/tool history (including previously loaded knowledge) is retained.
Use --prepare-only to inspect both exact inputs without making model calls.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.aerial_vln.protocol import AERIAL_VLN_PARSER_PROTOCOL_CONTEXT
from evaluation.benchmarks.travel_uav.protocol import TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
from src.llm import DefaultModelClient, ModelRequest
from src.llm.audit import snapshot_request
from src.llm.providers.openai_compatible import public_response_snapshot
from src.plugins import PluginRegistry
from src.agent.config import load_model_config
from src.tools.registry import default_registry
from tests.manual.tool_selection_ablation_probe import digest, restore_request, write_json, wire_check


def revised_request(original: dict) -> dict:
    """Change current guidance, preserving image pixels, task facts and past calls."""
    payload = copy.deepcopy(original)
    plugins = PluginRegistry.from_config()
    registry = default_registry(plugins)
    specs = {s.wire_name: s.to_provider_tool() for s in registry.model_specs()}
    payload["tools"] = [specs[t["function"]["name"]] for t in original["tools"]]
    catalog = [{"name": s.name, "description": s.description}
               for s in plugins.skill_catalog.skills if s.capability is None]

    def update_context(value):
        if isinstance(value, list):
            return [update_context(v) for v in value]
        if not isinstance(value, dict):
            return value
        updated = {}
        for key, item in value.items():
            if key == "knowledge_catalog":
                updated[key] = catalog
            elif key == "protocol_context" and isinstance(item, str) and item.startswith("AerialVLN"):
                updated[key] = AERIAL_VLN_PARSER_PROTOCOL_CONTEXT
            elif key == "protocol_context" and isinstance(item, str) and item.startswith("TravelUAV"):
                updated[key] = TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
            else:
                updated[key] = update_context(item)
        return updated

    replaced = 0
    for message in payload["messages"]:
        content = message.get("content")
        if message["role"] == "system" and isinstance(content, str) and content.startswith("You are FlyAgent's mission tool-loop agent."):
            message["content"] = (ROOT / "src/agent/planning/prompts/agent_tool_loop_system.txt").read_text(encoding="utf-8")
            replaced += 1
        if message["role"] != "user":
            continue
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for part in parts:
            if part.get("type") != "text" or not isinstance(part.get("text"), str):
                continue
            try:
                parsed = json.loads(part["text"])
            except json.JSONDecodeError:
                continue
            part["text"] = json.dumps(update_context(parsed), ensure_ascii=False)
        message["content"] = parts if isinstance(content, list) else parts[0]["text"]
    if replaced != 1:
        raise ValueError("Expected exactly one captured mission prompt")
    return payload


def run(source: Path, output: Path, *, round_index: int = 0, prepare_only: bool = False) -> dict:
    source = source.resolve(strict=True)
    run_dir = next(p for p in source.parents if (p / "memory/agent_transcript.jsonl").exists())
    output = output.resolve()
    if output.is_relative_to(run_dir) or run_dir.is_relative_to(output):
        raise ValueError("Output overlaps original run")
    record = json.loads(source.read_text(encoding="utf-8"))["model_rounds"][round_index]
    original = restore_request(record["request"], record["request_media"], run_dir)
    revised = revised_request(original)
    config = load_model_config()
    if config.model_name != original["model"]:
        raise ValueError("Paired historical comparison requires the captured model")
    output.mkdir(parents=True, exist_ok=False)
    result = {"source": str(source), "round_index": round_index, "original_request_sha256": digest(record["request"]),
              "kind": "new_offline_fixed_history_requests", "tools_executed": False,
              "config": config.redacted_summary(), "variants": [], "status": "prepared"}
    for name, payload in (("baseline", original), ("revised", revised)):
        snapshot, media = snapshot_request(payload)
        result["variants"].append({"name": name, "request": snapshot,
            "media_sha256": [m["sha256"] for m in media], "source_media": record["request_media"]})
    write_json(output / "probe.json", result)
    if prepare_only:
        return result
    client = DefaultModelClient(config.to_provider_settings())
    try:
        for row, payload in zip(result["variants"], (original, revised)):
            started = time.monotonic()
            trace = {}
            try:
                response = client.request(ModelRequest(messages=payload["messages"], tools=payload["tools"],
                    parameters={k: v for k, v in payload.items() if k not in ("messages", "tools", "model")}),
                    transport_trace=trace)
                row.update(response=public_response_snapshot(response.data), wire_check=wire_check(
                    public_response_snapshot(response.data), payload), returned_model=response.data.get("model"))
            finally:
                row.update(elapsed_seconds=time.monotonic() - started, transport=trace)
                write_json(output / "probe.json", result)
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
    write_json(output / "probe.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original decision JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = run(args.source, args.output, round_index=args.round_index, prepare_only=args.prepare_only)
    print(result["status"])
    raise SystemExit(1 if result["status"] == "failed" else 0)
