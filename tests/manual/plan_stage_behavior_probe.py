"""New offline decisions on saved real RGB/history; no flight tools execute.

Only the candidate plan description and planner/reminder prose replace captured
guidance. The simple-task variant explicitly derives a new instruction. All
other instruction, image, history and task-state bytes remain unchanged. These
checks diagnose model behavior, not route correctness or navigation success.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.config import load_model_config
from src.agent.planning.task_plan import TaskPlan, TaskPlanStep
from src.agent.planning.tool_contract_validator import validate_native_turn
from src.agent.transcript import AgentToolCall, AssistantTurn
from src.llm import create_model_client
from src.llm.audit import snapshot_request
from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from tests.manual.aerial_vln_plan_context_probe import digest, guard, load_credentials, read, write
from tests.manual.flight_teaching_probe import model_request, observation_id
from tests.manual.tool_selection_ablation_probe import restore_request


def candidate_request(original, source_root, registry, *, simple=False):
    """Copy captured input and replace only the owned candidate guidance."""
    payload = copy.deepcopy(original)
    replacement = registry.get("runtime.update_plan").to_provider_tool()
    for index, tool in enumerate(payload["tools"]):
        if tool["function"]["name"] == replacement["function"]["name"]:
            if tool["function"]["parameters"] != replacement["function"]["parameters"]:
                raise ValueError("Behavior-only probe cannot change the plan schema")
            payload["tools"][index] = replacement
    prompts = {
        "You are FlyAgent's mission tool-loop agent.": "agent_tool_loop_system.txt",
        "Progress checkpoint:": "task_progress_reminder.txt",
    }
    replaced = 0
    for message in payload["messages"]:
        content = message.get("content")
        if message["role"] == "system" and isinstance(content, str):
            for prefix, filename in prompts.items():
                if content.startswith(prefix):
                    message["content"] = (source_root / "src/agent/planning/prompts" / filename).read_text(encoding="utf-8").strip()
                    replaced += prefix.startswith("You are")
        if simple and message["role"] == "user" and isinstance(content, str) and content.startswith("INSTRUCTION CONTEXT\n"):
            envelope = json.loads(content.split("\n", 1)[1])
            envelope["instruction"] = "Rise a little to obtain a clearer view, then stop."
            message["content"] = "INSTRUCTION CONTEXT\n" + json.dumps(envelope, ensure_ascii=False)
            simple = False
    if replaced != 1 or simple:
        raise ValueError("Expected one mission prompt and a captured instruction envelope")
    return payload


def plan_from_request(payload):
    for message in payload["messages"]:
        content = message.get("content")
        if isinstance(content, str) and content.startswith("CURRENT TASK STATE\n"):
            value = json.loads(content.split("\n", 1)[1]).get("task_plan")
            return TaskPlan(tuple(TaskPlanStep.from_dict(step) for step in value["steps"]),
                            value["revision"], value["explanation"]) if value else TaskPlan()
    raise ValueError("Captured current task state is missing")


def scenario_sources(history_root):
    """Select original decisions by recorded calls, never hidden scores."""
    cases = read(history_root / "progress.json")["cases"]
    requests = {}
    for index in (11, 12, 14, 19):
        run = Path(cases[index]["attempts"][-1]["result"]["run_dir"])
        values = []
        for path in sorted(run.glob("debug/tool_loop/*/decision_*.json")):
            for captured in read(path)["model_rounds"]:
                values.append((run, path, captured))
        requests[index] = values
    def find(index, predicate):
        return next(value for value in requests[index]
                    if predicate(value[2].get("canonical_turn", {}).get("tool_calls", [])))
    physical = lambda calls: any(c["tool_name"].startswith("uav.") for c in calls)
    complex_source = find(11, physical)
    # A non-null description can repeat the old value. Select an actual recorded
    # correction relative to the authoritative state seen by that decision.
    def changes_description(value):
        run_dir, _, captured = value
        payload = restore_request(captured["request"], captured["request_media"], run_dir)
        previous = {step.id: step.description for step in plan_from_request(payload).steps}
        return any(call["tool_name"] == "runtime.update_plan"
                   and call["args"]["operation"] == "patch"
                   and any(step["description"] is not None
                           and step["description"] != previous.get(step["id"])
                           for step in call["args"]["steps"])
                   for call in captured.get("canonical_turn", {}).get("tool_calls", []))
    correction_source = next(value for value in requests[12] if changes_description(value))
    return [
        ("simple_destination", "no_plan_needed", complex_source, True),
        ("complex_creation", "plan_before_action", complex_source, False),
        ("stage_unchanged", "unchanged_stage", find(19, physical), False),
        ("stage_transition", "plan_before_action", find(19, lambda calls: any(
            c["tool_name"] == "runtime.update_plan" and c["args"]["operation"] == "patch" for c in calls)), False),
        ("judgment_correction", "plan_before_action", correction_source, False),
        ("completion_reconcile", "plan_before_action", find(19, lambda calls: any(
            c["tool_name"] == "runtime.report_completed" for c in calls)), False),
        ("blocked_preserves_unfinished", "blocked", find(14, lambda calls: any(
            c["tool_name"] == "runtime.report_blocked" for c in calls)), False),
    ]


def run(root, history_root):
    manifest = guard(root)
    load_credentials(manifest)
    registry = default_registry(PluginRegistry.from_config(root / "source/configs/plugins.json"))
    model = load_model_config(manifest["config"]["env_path"])
    client = create_model_client(model.to_provider_settings())
    output = root / "plan_behavior"
    output.mkdir(exist_ok=False)
    record = {"kind": "new_offline_saved_context_decisions", "tools_executed": False,
              "model": model.redacted_summary(), "scenarios": [],
              "notice": "Expected behavior is a diagnostic hypothesis, not proof of milestone or navigation correctness."}
    for name, expected, (run_dir, source, captured), simple in scenario_sources(history_root):
        transcript = run_dir / "memory/agent_transcript.jsonl"
        before = (digest(source), digest(transcript))
        original = restore_request(captured["request"], captured["request_media"], run_dir)
        payload = candidate_request(original, root / "source", registry, simple=simple)
        snapshot, media = snapshot_request(payload)
        _, original_media = snapshot_request(original)
        entry = {"scenario": name, "expected": expected, "source": str(source),
                 "source_sha256": before[0], "transcript_sha256": before[1],
                 "derived_instruction": simple, "request": snapshot,
                 "media_sha256": [item["sha256"] for item in media],
                 "same_images": [item["sha256"] for item in media] == [item["sha256"] for item in original_media]}
        record["scenarios"].append(entry)
        write(output / "report.json", record)
        try:
            response = client.request(model_request(payload))
            entry.update(response=response.data, usage=response.usage, transport=response.transport_trace)
            if response.data.get("model") != model.model_name:
                raise ValueError("Provider returned a different model")
            calls = [AgentToolCall(call["id"], registry.from_wire_name(call["function"]["name"]).name,
                                  json.loads(call["function"]["arguments"])) for call in response.tool_calls]
            turn = AssistantTurn("offline_" + name, response.text or "", calls)
            prior_ids = {c["id"] for message in payload["messages"] for c in message.get("tool_calls", [])}
            validate_native_turn(turn, registry, allowed_names=set(registry.names()),
                latest_observation={"observation_id": observation_id(payload["messages"])}, prior_call_ids=prior_ids)
            plan = plan_from_request(payload)
            updates = [call for call in calls if call.tool_name == "runtime.update_plan"]
            candidate = plan.updated(updates[0].args) if updates else plan
            tools = [call.tool_name for call in calls]
            observed = bool(updates) if expected == "plan_before_action" else not updates
            if expected == "blocked":
                observed = ("runtime.report_blocked" in tools or bool(updates)) and any(
                    step.status in {"pending", "in_progress"} for step in candidate.steps)
            entry.update(contract_valid=True, turn=turn.to_dict(), candidate_plan=candidate.to_dict(),
                         expected_behavior_observed=observed)
        except Exception as exc:
            entry.update(contract_valid=False, error_type=type(exc).__name__, error=str(exc))
        entry["original_unchanged"] = before == (digest(source), digest(transcript))
        write(output / (name + ".json"), entry)
        write(output / "report.json", record)
        print(json.dumps({"scenario": name, "valid": entry["contract_valid"],
                          "expected_observed": entry.get("expected_behavior_observed"),
                          "tools": [c["tool_name"] for c in entry.get("turn", {}).get("tool_calls", [])]}), flush=True)
    record["contract_passed"] = all(e["contract_valid"] and e["same_images"] and e["original_unchanged"]
                                    for e in record["scenarios"])
    record["expected_behavior_matches"] = sum(e.get("expected_behavior_observed") is True for e in record["scenarios"])
    record["status"] = "completed"
    write(output / "report.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.history_root.resolve())
    print(json.dumps({"contract_passed": result["contract_passed"],
                      "expected_behavior_matches": result["expected_behavior_matches"]}))
    raise SystemExit(0 if result["contract_passed"] else 2)
