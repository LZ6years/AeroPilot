"""Join immutable decision inputs and actual results by instruction and call IDs."""

from __future__ import annotations

from typing import Any

from src.operator_api.services.event_stream import public_value
from .replay import read_json


def build_decisions(repository, run_id: str, events: list[dict], transcript: list[dict]) -> list[dict[str, Any]]:
    run = repository.get(run_id)
    output: dict[tuple[str | None, str], dict] = {}
    goals = {}
    for row in transcript:
        if row.get("kind") == "user_instruction":
            payload = row.get("payload") or {}
            goals[row.get("instruction_id")] = payload.get("instruction", payload.get("text", ""))
    current_instruction = None
    for event in events:
        payload = event.get("payload") or {}
        current_instruction = event.get("instruction_id") or payload.get("instruction_id") or current_instruction
        if event.get("kind") == "instruction_started":
            goals[current_instruction] = payload.get("instruction", "")
        decision_id = event.get("decision_id") or payload.get("decision_id")
        if not decision_id:
            continue
        key = (current_instruction, decision_id)
        if event.get("kind") == "model_turn_recorded":
            debug = {}
            reference = payload.get("debug_ref")
            if reference:
                debug = read_json(repository.child(run, reference))
            rounds = debug.get("model_rounds") or []
            turn = payload.get("canonical_turn") or (rounds[0].get("canonical_turn") if rounds else {}) or {}
            output[key] = {
                "key": f"{current_instruction or ''}:{decision_id}",
                "instruction_id": current_instruction, "decision_id": decision_id,
                "timestamp": event.get("timestamp"), "seq": event["seq"],
                "instruction": goals.get(current_instruction, ""),
                "public_intent": turn.get("public_intent", ""),
                "public_intent_source": turn.get("public_intent_source"),
                "tool_calls": turn.get("tool_calls") or [], "tool_results": [],
                "observation_id": payload.get("observation_id") or debug.get("checkpoint_observation_id"),
                "task_state": debug.get("current_task_state") or {},
                "status": "pending", "debug_ref": reference,
            }
        elif key in output:
            selected = output[key]
            kind = event.get("kind")
            if kind == "physical_window_recorded":
                selected["status"] = payload.get("status", "unknown")
                selected["observation_after_id"] = payload.get("observation_after_id")
                selected["measurements"] = payload.get("measurements") or {}
                selected["tool_results"].extend(payload.get("ordered_steps") or [])
            elif kind in {"task_control_recorded", "read_only_turn_recorded", "plan_read_only_turn_recorded"}:
                results = payload.get("tool_results") or ([payload["tool_result"]] if payload.get("tool_result") else [])
                selected["tool_results"].extend(results)
                selected["status"] = "rejected" if any(row.get("status") == "rejected" for row in results) else "ok"
            elif kind in {"tool_turn_rejected", "operator_decision_discarded"}:
                selected["status"] = "rejected" if kind == "tool_turn_rejected" else "discarded"
                selected["error"] = payload.get("error") or payload.get("reason")
    return [public_value(row) for row in output.values()]
