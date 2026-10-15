"""Check new teaching guidance against saved real navigation contexts.

Both variants are new offline requests, never reconstructions of old decisions.
Original history and RGB stay fixed. The guidance variant adds explicitly labelled
local knowledge receipts; no flight, perception, plan or terminal tool executes.
Contract validity is automatic; the quality of the navigation choice needs review.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from src.agent.config import load_model_config
from src.agent.transcript import AgentToolCall, AssistantTurn
from src.agent.planning.tool_contract_validator import validate_native_turn
from src.llm import ModelRequest, create_model_client
from src.llm.audit import snapshot_request
from src.plugins import PluginRegistry
from src.plugins.skills.knowledge_service import KnowledgeService
from src.tools.registry import default_registry
from tests.manual.agent_guidance_probe import revised_request
from tests.manual.aerial_vln_plan_context_probe import digest, guard, load_credentials, read, write
from tests.manual.tool_selection_ablation_probe import restore_request


# These positions select the audited situations, never their hidden scores.
SCENARIOS = (
    ("initial_response", 0, "decision_0002.json", ("basic_flight_control",)),
    ("junction_approach", 7, "decision_0025.json", ("route_traversal_and_passage", "basic_flight_control")),
    ("occluded_view", 7, "decision_0009.json", ("visual_target_task_reasoning", "basic_flight_control")),
    ("dark_view_recovery", 8, "decision_0040.json", ("abnormal_situation_recovery",)),
)


def observation_id(messages):
    """Read the newest labelled sensory checkpoint, ignoring task-state blocks."""
    for message in reversed(messages):
        parts = message.get("content")
        texts = [part.get("text", "") for part in parts] if isinstance(parts, list) else [parts]
        for text in texts:
            if isinstance(text, str) and text.startswith("STATE CHECKPOINT CURRENT"):
                match = re.search(r"\bobservation_id=([^\s.]+)", text)
                if match:
                    return match.group(1)
                if "{" in text:
                    return json.loads(text[text.index("{"):])["observation_id"]
    raise ValueError("Captured current observation is missing")


def with_guidance(payload, service, names):
    """Add local receipts without editing captured calls, images or task facts."""
    result = copy.deepcopy(payload)
    messages = result["messages"]
    states = [m for m in messages if isinstance(m.get("content"), str)
              and m["content"].startswith("CURRENT TASK STATE\n")]
    if len(states) != 1:
        raise ValueError("Expected one current task state")
    state = json.loads(states[0]["content"].split("\n", 1)[1])
    messages.remove(states[0])
    existing_ids = {c["id"] for m in messages for c in m.get("tool_calls", [])}
    for name in names:
        call_id = "offline_teaching_" + name
        if call_id in existing_ids:
            raise ValueError("Offline receipt call ID already exists")
        receipt = service.load(name)
        messages.append({"role": "assistant", "content": "Load flight-method guidance for this offline comparison.",
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": "knowledge_load", "arguments": json.dumps({"name": name})}}]})
        messages.append({"role": "tool", "tool_call_id": call_id,
            "content": json.dumps({"tool_name": "knowledge.load", "status": "ok",
                "message": "Local offline knowledge receipt; not environmental evidence.", "data": receipt})})
    index = {item["name"]: item for item in state.get("loaded_knowledge", [])}
    index.update({item["name"]: item for item in service.loaded_index()})
    state["loaded_knowledge"] = list(index.values())
    messages.append({"role": "system", "content": "CURRENT TASK STATE\n" + json.dumps(state)})
    return result


def model_request(payload):
    """Captured wire envelopes contain stream; the shared client owns that flag."""
    return ModelRequest(messages=payload["messages"], tools=payload["tools"],
        parameters={k: v for k, v in payload.items() if k not in {"messages", "tools", "model", "stream"}})


def run(root, history_root, *, prepare_only=False, client=None):
    """Use the frozen provider profile and preserve both original source hashes."""
    manifest = guard(root)
    load_credentials(manifest)
    progress = read(history_root / "progress.json")
    plugins = PluginRegistry.from_config(root / "source/configs/plugins.json")
    registry = default_registry(plugins)
    model = load_model_config(manifest["config"]["env_path"])
    output = root / "knowledge_behavior"
    output.mkdir(exist_ok=False)
    record = {"kind": "new_offline_fixed_history_requests", "tools_executed": False,
        "model": model.redacted_summary(), "scenarios": [], "passed": False,
        "notice": "Only knowledge receipts are supplied locally. Contract checks do not verify navigation semantics."}
    if not prepare_only:
        client = client or create_model_client(model.to_provider_settings())
    for name, index, filename, guidance in SCENARIOS:
        run_dir = Path(progress["cases"][index]["attempts"][-1]["result"]["run_dir"])
        candidates = list(run_dir.glob("debug/tool_loop/*/" + filename))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one saved real request for " + name)
        source = candidates[0]
        transcript = run_dir / "memory/agent_transcript.jsonl"
        before = (digest(source), digest(transcript))
        captured = read(source)["model_rounds"][0]
        original = restore_request(captured["request"], captured["request_media"], run_dir)
        current = revised_request(original)
        variants = (("catalog_only", current), ("with_guidance",
            with_guidance(current, KnowledgeService(plugins.skill_catalog), guidance)))
        row = {"scenario": name, "source": str(source), "source_sha256": before[0],
            "transcript_sha256": before[1], "guidance": guidance, "variants": []}
        record["scenarios"].append(row)
        for variant, payload in variants:
            snapshot, media = snapshot_request(payload)
            entry = {"variant": variant, "request": snapshot,
                "media_sha256": [item["sha256"] for item in media], "source_media": captured["request_media"]}
            row["variants"].append(entry)
            if prepare_only:
                continue
            write(output / (name + ".json"), row)
            write(output / "report.json", record)
            try:
                response = client.request(model_request(payload))
            except Exception as exc:
                entry.update(contract_valid=False, error_type=type(exc).__name__)
                record.update(status="failed", passed=False)
                write(output / (name + ".json"), row)
                write(output / "report.json", record)
                raise RuntimeError("Offline probe request failed; inspect saved report") from None
            entry.update(response=response.data, usage=response.usage, returned_model=response.data.get("model"))
            try:
                if response.data.get("model") != model.model_name:
                    raise ValueError("Provider returned a different model")
                calls = [AgentToolCall(call["id"], registry.from_wire_name(call["function"]["name"]).name,
                    json.loads(call["function"]["arguments"])) for call in response.tool_calls]
                turn = AssistantTurn("offline_" + name + "_" + variant, response.text or "", calls)
                prior = {c["id"] for m in payload["messages"] for c in m.get("tool_calls", [])}
                effect = validate_native_turn(turn, registry,
                    allowed_names={s.name for s in registry.model_specs()},
                    latest_observation={"observation_id": observation_id(payload["messages"])}, prior_call_ids=prior)
                entry.update(contract_valid=True, effect=effect, turn=turn.to_dict())
            except Exception as exc:
                entry.update(contract_valid=False, error_type=type(exc).__name__, error=str(exc))
            write(output / (name + ".json"), row)
            write(output / "report.json", record)
            print(json.dumps({"scenario": name, "variant": variant, "valid": entry["contract_valid"],
                "tools": [c["function"]["name"] for c in response.tool_calls]}), flush=True)
        row["original_unchanged"] = before == (digest(source), digest(transcript))
        row["same_images"] = row["variants"][0]["media_sha256"] == row["variants"][1]["media_sha256"]
        write(output / (name + ".json"), row)
    record.update(status="prepared" if prepare_only else "completed", passed=not prepare_only and all(
        row["original_unchanged"] and row["same_images"] and all(v.get("contract_valid") for v in row["variants"])
        for row in record["scenarios"]))
    write(output / "report.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    # Import and load guidance from the frozen candidate, never a later workspace edit.
    import os
    os.chdir(args.root.resolve() / "source")
    report = run(args.root.resolve(), args.history_root.resolve(), prepare_only=args.prepare_only)
    print(json.dumps({"status": report["status"], "passed": report["passed"]}))
    raise SystemExit(0 if args.prepare_only or report["passed"] else 2)
