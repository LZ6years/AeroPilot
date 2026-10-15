"""New protocol decisions on saved real requests; no tools are dispatched.

Only the caller-owned protocol text changes. Instructions, tools, task state,
history and original image bytes remain captured evidence. The focused checks
detect premature terminal calls or cancellation of the recorded remaining
steps, not the correctness of natural-language routes or landmark identities.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.benchmarks.aerial_vln.protocol import AERIAL_VLN_PARSER_PROTOCOL_CONTEXT
from src.agent.planning.tool_contract_validator import validate_native_turn
from src.agent.transcript import AgentToolCall, AssistantTurn
from src.llm import create_model_client
from src.llm.audit import snapshot_request
from src.agent.config import load_model_config
from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from tests.manual.aerial_vln_plan_context_probe import digest, guard, load_credentials, read, write
from tests.manual.flight_teaching_probe import model_request, observation_id
from tests.manual.plan_stage_behavior_probe import plan_from_request
from tests.manual.tool_selection_ablation_probe import restore_request


def candidate_request(original, protocol):
    """Replace exactly one instruction-envelope field; keep all other input."""
    payload = copy.deepcopy(original)
    replacements = 0
    old_protocol = None
    for message in payload["messages"]:
        text = message.get("content")
        if isinstance(text, str) and text.startswith("INSTRUCTION CONTEXT\n"):
            envelope = json.loads(text.split("\n", 1)[1])
            old_protocol = envelope["protocol_context"]
            envelope["protocol_context"] = protocol
            message["content"] = "INSTRUCTION CONTEXT\n" + json.dumps(envelope, ensure_ascii=False)
            replacements += 1
    if replacements != 1:
        raise ValueError("Expected exactly one captured instruction protocol")
    return payload, old_protocol


def request_snapshot(payload):
    """Use the shared audit while keeping binary image bytes outside JSON."""
    snapshot, media = snapshot_request(payload)
    return snapshot, [{key: value for key, value in item.items() if key != "data"} for item in media]


def scenario_sources(history_root, tasks):
    """Find recorded cancellations and terminal decisions by selected identity."""
    rows = read(history_root / "progress.json")["cases"]
    selected = {row["task"]["episode_id"]: row for row in rows}
    scenarios = []
    for task in tasks:
        cid = task["episode_id"]
        run_dir = Path(selected[cid]["attempts"][-1]["result"]["run_dir"])
        cancellation = None
        terminal = None
        for path in sorted(run_dir.glob("debug/tool_loop/*/decision_*.json")):
            for captured in read(path).get("model_rounds", []):
                calls = captured.get("canonical_turn", {}).get("tool_calls", [])
                cancelled = [step["id"] for call in calls if call["tool_name"] == "runtime.update_plan"
                             for step in call["args"]["steps"] if step.get("status") == "cancelled"]
                if cancelled and cancellation is None:
                    cancellation = ("before_cancellation", run_dir, path, captured, cancelled)
                if any(call["tool_name"] == "runtime.report_completed" for call in calls):
                    terminal = ("before_completion", run_dir, path, captured, [])
        if cancellation:
            scenarios.append((cid, *cancellation))
        if terminal is None:
            raise ValueError(f"Selected regression case has no recorded completion decision: {cid}")
        scenarios.append((cid, *terminal))
    return scenarios


def check_decision(calls, plan, protected_ids):
    """Check explicit premature completion, leaving route semantics for review."""
    candidate = plan
    for call in calls:
        if call.tool_name == "runtime.update_plan":
            candidate = candidate.updated(call.args)
    cancelled = [step.id for step in candidate.steps if step.status == "cancelled" and step.id in protected_ids]
    return {"premature_completion": any(call.tool_name == "runtime.report_completed" for call in calls),
            "cancelled_remaining_step_ids": cancelled, "candidate_plan": candidate.to_dict()}


def run(root, history_root):
    manifest = guard(root)
    if manifest.get("plan_mode") != "optional":
        raise ValueError("Protocol regression requires the authorized optional-plan surface")
    load_credentials(manifest)
    registry = default_registry(PluginRegistry.from_config(root / "source/configs/plugins.json"))
    model = load_model_config(manifest["config"]["env_path"])
    client = create_model_client(model.to_provider_settings())
    output = root / "protocol_behavior"
    output.mkdir(exist_ok=False)
    record = {"status": "running", "kind": "new_offline_saved_context_decisions", "tools_executed": False,
              "model": model.redacted_summary(), "scenarios": [],
              "notice": "Only protocol text changes. Route and landmark correctness require human review."}
    for cid, name, run_dir, source, captured, protected in scenario_sources(history_root, manifest["tasks"]):
        transcript = run_dir / "memory/agent_transcript.jsonl"
        original_refs = [run_dir / media["asset_ref"] for media in captured["request_media"]]
        evidence_paths = [source, transcript, *original_refs]
        before = {str(path): digest(path) for path in evidence_paths}
        original = restore_request(captured["request"], captured["request_media"], run_dir)
        payload, old_protocol = candidate_request(original, AERIAL_VLN_PARSER_PROTOCOL_CONTEXT)
        snapshot, media = request_snapshot(payload)
        _, original_media = request_snapshot(original)
        entry = {"episode_id": cid, "scenario": name, "attempt": 1, "source": str(source),
                 "original_evidence_sha256": before, "old_protocol": old_protocol,
                 "new_protocol": AERIAL_VLN_PARSER_PROTOCOL_CONTEXT, "protected_remaining_step_ids": protected,
                 "request": snapshot, "request_media": media,
                 "same_images": [m["sha256"] for m in media] == [m["sha256"] for m in original_media]}
        record["scenarios"].append(entry)
        write(output / "report.json", record)
        try:
            response = client.request(model_request(payload))
            entry.update(response=response.data, usage=response.usage, transport=response.transport_trace)
            if response.data.get("model") != model.model_name:
                raise ValueError("Provider returned a different model")
            calls = [AgentToolCall(call["id"], registry.from_wire_name(call["function"]["name"]).name,
                                  json.loads(call["function"]["arguments"])) for call in response.tool_calls]
            turn = AssistantTurn("offline_protocol_" + name, response.text or "", calls)
            allowed = {registry.from_wire_name(tool["function"]["name"]).name for tool in payload["tools"]}
            prior_ids = {c["id"] for message in payload["messages"] for c in message.get("tool_calls", [])}
            validate_native_turn(turn, registry, allowed_names=allowed,
                latest_observation={"observation_id": observation_id(payload["messages"])}, prior_call_ids=prior_ids)
            decision = check_decision(calls, plan_from_request(payload), protected)
            entry.update(contract_valid=True, turn=turn.to_dict(), **decision)
            entry["expected_behavior_observed"] = not decision["premature_completion"] and not decision["cancelled_remaining_step_ids"]
        except Exception as exc:
            entry.update(contract_valid=False, error_type=type(exc).__name__, error=str(exc))
        entry["original_unchanged"] = before == {str(path): digest(path) for path in evidence_paths}
        write(output / (cid + "_" + name + ".json"), entry)
        write(output / "report.json", record)
        print(json.dumps({"episode_id": cid, "scenario": name, "contract_valid": entry["contract_valid"],
                          "expected_observed": entry.get("expected_behavior_observed")}), flush=True)
    record["passed"] = all(e["contract_valid"] and e["same_images"] and e["original_unchanged"]
                           and e.get("expected_behavior_observed") for e in record["scenarios"])
    record["status"] = "passed" if record["passed"] else "failed"
    write(output / "report.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    frozen = root / "source/tests/manual/aerial_vln_protocol_behavior_probe.py"
    if frozen.is_file() and frozen.resolve() != Path(__file__).resolve():
        raise SystemExit(subprocess.run([sys.executable, "-u", str(frozen), *sys.argv[1:]], cwd=root / "source").returncode)
    result = run(root, args.history_root.resolve())
    print(json.dumps({"status": result["status"], "scenarios": len(result["scenarios"])}))
    raise SystemExit(0 if result["passed"] else 2)
