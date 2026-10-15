"""Offline input variants keep history and image evidence unchanged."""
import copy
import json

import pytest
from pathlib import Path

from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from tests.support.planner import revised_request


def test_revised_request_changes_guidance_not_original_history_or_pixels():
    history = {"role": "tool", "tool_call_id": "prior", "content": "original knowledge result"}
    plugins = PluginRegistry.from_config()
    original = {"model": "unchanged", "tools": [s.to_provider_tool() for s in default_registry(plugins).model_specs()],
        "messages": [{"role": "system", "content": "You are FlyAgent's mission tool-loop agent. old"},
            history, {"role": "user", "content": [
                {"type": "text", "text": json.dumps({"protocol_context": "TravelUAV old context",
                    "instruction": "Turn right 79 degrees", "knowledge_catalog": []})},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,ORIGINAL"}}]}]}
    before = copy.deepcopy(original)
    changed = revised_request(original)
    assert original == before
    assert changed["messages"][1] == history
    assert changed["messages"][2]["content"][1] == original["messages"][2]["content"][1]
    context = json.loads(changed["messages"][2]["content"][0]["text"])
    assert context["instruction"] == "Turn right 79 degrees"
    assert "instruction start" in context["protocol_context"]
    assert len(context["knowledge_catalog"]) == 6
    assert "actual displacement" in changed["messages"][0]["content"]
    assert "inconsistent_depth_surfaces" in str(changed["tools"])


def test_probe_refuses_unrecognized_prompt_instead_of_silently_partial_rewrite():
    with pytest.raises(ValueError, match="mission prompt"):
        revised_request({"tools": [], "messages": []})


def test_saved_envelope_header_gets_current_catalog_and_single_action_prompt():
    payload = {"tools": [], "messages": [
        {"role": "system", "content": "You are FlyAgent's mission tool-loop agent. old"},
        {"role": "system", "content": "Use only tools declared in this request. old batching"},
        {"role": "user", "content": 'INSTRUCTION CONTEXT\n{"knowledge_catalog":[],"instruction":"Original task"}'}]}
    result = revised_request(payload)
    envelope = json.loads(result["messages"][2]["content"].split("\n", 1)[1])
    assert len(envelope["knowledge_catalog"]) == 6 and envelope["instruction"] == "Original task"
    assert "exactly one physical call" in result["messages"][1]["content"]
    assert payload["messages"][2]["content"].startswith('INSTRUCTION CONTEXT\n{"knowledge_catalog":[]')


def test_revised_guidance_keeps_feedback_resident_and_knowledge_on_demand():
    original = {"tools": [], "messages": [
        {"role": "system", "content": "You are FlyAgent's mission tool-loop agent. old"},
        {"role": "user", "content": 'INSTRUCTION CONTEXT\n{"knowledge_catalog":[],"instruction":"Observe the road"}'}]}
    result = revised_request(original)
    prompt = result["messages"][0]["content"]
    assert "principles below apply to every physical decision" in prompt
    assert "Load full basic-control guidance when parameter choice" in prompt
    assert "reuse loaded text without loading it again each turn" in prompt
    assert "When provided, CONTEXT IMAGE" in prompt
    assert not any(message["role"] == "tool" for message in result["messages"])
    assert all("Worked example:" not in str(message["content"]) for message in result["messages"])


def test_public_intent_and_generic_path_examples_live_in_owning_layers():
    root = Path(__file__).resolve().parents[1]
    prompt = (root / "src/agent/planning/prompts/agent_tool_loop_system.txt").read_text(encoding="utf-8")
    assert "one concise public intent with each" in prompt
    assert "whole movement path" in prompt
    base = root / "plugins/knowledge/uav_flight"
    basic = (base / "basic_flight_control/SKILL.md").read_text(encoding="utf-8")
    route = (base / "route_traversal_and_passage/SKILL.md").read_text(encoding="utf-8")
    assert "intersects the facade" in basic
    assert "Worked example:" in route
    reference = (base / "route_traversal_and_passage/references/route_progress_examples.md").read_text(encoding="utf-8")
    assert "turn once" not in reference
