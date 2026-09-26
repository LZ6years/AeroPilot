"""Offline input variants keep history and image evidence unchanged."""
import copy
import json

import pytest

from src.tools.registry import default_registry
from tests.manual.agent_guidance_probe import revised_request


def test_revised_request_changes_guidance_not_original_history_or_pixels():
    history = {"role": "tool", "tool_call_id": "prior", "content": "original knowledge result"}
    original = {"model": "unchanged", "tools": [s.to_provider_tool() for s in default_registry().model_specs()],
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
    assert len(context["knowledge_catalog"]) == 5
    assert "actual measured motion" in changed["messages"][0]["content"]
    assert "inconsistent_depth_surfaces" in str(changed["tools"])


def test_probe_refuses_unrecognized_prompt_instead_of_silently_partial_rewrite():
    with pytest.raises(ValueError, match="mission prompt"):
        revised_request({"tools": [], "messages": []})
