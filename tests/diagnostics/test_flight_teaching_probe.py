"""Manual-probe diagnostics preserve real-history semantics and image identity."""
import copy
import json

import pytest

from src.plugins import PluginRegistry
from src.plugins.skills.knowledge_service import KnowledgeService
from tests.manual.flight_teaching_probe import model_request, observation_id, with_guidance


def request():
    return {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "past", "function": {
            "name": "uav_move_body", "arguments": '{"vx":2,"vy":0,"vz":0,"duration":2}'}}]},
        {"role": "tool", "tool_call_id": "past", "content": '{"status":"ok"}'},
        {"role": "user", "content": [
            {"type": "text", "text": 'STATE CHECKPOINT CURRENT checkpoint_2\n{"observation_id":"obs2"}'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ORIGINAL"}}]},
        {"role": "system", "content": 'CURRENT TASK STATE\n{"task_plan":null,"loaded_knowledge":[]}'}]}


def test_guidance_receipts_preserve_history_images_and_one_latest_state():
    original = request()
    before = copy.deepcopy(original)
    service = KnowledgeService(PluginRegistry.from_config().skill_catalog)
    result = with_guidance(original, service, ("basic_flight_control", "abnormal_situation_recovery"))
    assert original == before and result["messages"][:3] == original["messages"][:3]
    states = [m for m in result["messages"] if isinstance(m.get("content"), str)
              and m["content"].startswith("CURRENT TASK STATE\n")]
    assert len(states) == 1
    current = json.loads(states[0]["content"].split("\n", 1)[1])
    assert {item["name"] for item in current["loaded_knowledge"]} == {
        "basic_flight_control", "abnormal_situation_recovery"}
    assert current["task_plan"] is None and observation_id(result["messages"]) == "obs2"
    receipts = [json.loads(m["content"]) for m in result["messages"] if m.get("tool_call_id", "").startswith("offline_")]
    assert all(r["data"]["body_included"] and r["data"]["references"] == [] for r in receipts)


def test_guidance_refuses_missing_or_duplicate_state():
    service = KnowledgeService(PluginRegistry.from_config().skill_catalog)
    payload = request()
    payload["messages"].append(copy.deepcopy(payload["messages"][-1]))
    with pytest.raises(ValueError, match="one current task state"):
        with_guidance(payload, service, ("basic_flight_control",))
    assert service.loaded_index() == []


def test_real_label_without_json_identifies_current_rgb_observation():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "STATE CHECKPOINT CURRENT checkpoint_id=checkpoint_2 observation_id=obs2. Chronological observation state."},
        {"type": "image_url", "image_url": {"url": "original"}}]}]
    assert observation_id(messages) == "obs2"


def test_wire_stream_is_owned_by_client_not_generation_parameters():
    payload = {**request(), "tools": [], "model": "captured", "stream": False, "tool_choice": "auto"}
    adapted = model_request(payload)
    assert adapted.parameters == {"tool_choice": "auto"}
    assert "stream" in payload and adapted.messages == payload["messages"]
