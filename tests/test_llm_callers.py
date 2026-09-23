"""Lock caller-owned message order and generation options across communication migration."""
from __future__ import annotations

import json

import pytest

from src.llm import ModelResponse
from src.llm.providers.openai_compatible import encode_request
from src.runtime.configs.model_config import ModelConfig


class CaptureClient:
    """Never performs I/O; freezes the exact outgoing wire dictionary."""
    def __init__(self, config):
        self.config = config
        self.payloads = []

    def request(self, request, *, error_prefix, transport_trace=None):
        self.payloads.append(encode_request(self.config.to_provider_settings(), request))
        return ModelResponse({"choices": [{"message": {"content": "{}"}}]})


def system(text):
    return {"role": "system", "content": text}


def test_planner_injected_client_matches_legacy_request_hook(tmp_path):
    from tests.test_model_action_planner import StubPlanner, make_transcript, response_with_calls
    from src.agent.planning.model_action_planner import ModelActionPlanner
    from src.tools.registry import default_registry

    response = response_with_calls(("call_1", "uav_hover", {}), content=None)
    legacy_hook = StubPlanner(response)
    config = legacy_hook.model_config
    client = CaptureClient(config)
    captured_request = client.request

    def request(model_request, **kwargs):
        captured_request(model_request, **kwargs)
        return ModelResponse(response)

    client.request = request
    transcript = make_transcript(tmp_path)
    before = transcript.snapshot()
    registry = default_registry()
    old_turn = legacy_hook.plan_turn(transcript, registry)
    new = ModelActionPlanner(config, client=client)
    new_turn = new.plan_turn(transcript, registry)
    assert client.payloads == legacy_hook.payloads
    assert new_turn.tool_calls == old_turn.tool_calls
    assert new_turn.public_intent_source == "tool_arguments_fallback"
    assert transcript.snapshot() == before
    assert new.last_exchange["request"] == legacy_hook.last_exchange["request"]
