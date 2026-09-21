"""Communication and message tooling contracts; no live provider or simulator."""
from __future__ import annotations

import ast
import base64
import copy
import json
from pathlib import Path

import pytest

from src.llm import DefaultModelClient, MessageBuilder, ModelRequest, ModelResponse, ProviderSettings
from src.llm.audit import snapshot_request
from src.llm.providers.openai_compatible import chat_completions_url, encode_request


def test_builder_order_edit_and_isolation():
    source = [{"role": "user", "content": [{"type": "text", "text": "original"}]}]
    builder = MessageBuilder().add_system("identity", key="identity")
    builder.extend(source, key="history").add_user("last", key="last")
    source[0]["content"][0]["text"] = "mutated source"
    queried = builder.get("history")
    queried[0]["content"][0]["text"] = "mutated result"
    assert builder.get("history")[0]["content"][0]["text"] == "original"
    builder.replace("identity", [{"role": "system", "content": "replacement"}])
    builder.remove("last")
    assert [m["role"] for m in builder.build()] == ["system", "user"]
    built = builder.build()
    built[0]["content"] = "mutated build"
    assert builder.build()[0]["content"] == "replacement"
    assert all("key" not in m for m in builder.build())


@pytest.mark.parametrize("operation", ["get", "replace", "remove"])
def test_builder_missing_key_is_explicit(operation):
    builder = MessageBuilder()
    with pytest.raises(KeyError):
        getattr(builder, operation)("missing", *([[]] if operation == "replace" else []))


def test_builder_has_no_implicit_deduplication():
    builder = MessageBuilder().add_system("same", key="first").add_system("same")
    builder.extend([], key="empty")
    assert builder.get("empty") == []
    assert len(builder.build()) == 2
    with pytest.raises(ValueError, match="duplicate"):
        builder.add_user("x", key="first")
    with pytest.raises(ValueError):
        builder.extend([], key="")


def test_request_and_wire_encoding_are_independent_copies():
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "example", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"}]
    tools = [{"type": "function", "function": {"name": "example"}}]
    parameters = {"temperature": 0, "tool_choice": "auto"}
    expected = copy.deepcopy(messages)
    request = ModelRequest(messages, tools, parameters)
    messages[0]["tool_calls"].clear()
    tools.clear()
    parameters.clear()
    payload = encode_request(ProviderSettings(model_name="model"), request)
    assert payload == {"model": "model", "messages": expected,
                       "tools": [{"type": "function", "function": {"name": "example"}}],
                       "temperature": 0, "tool_choice": "auto"}
    payload["messages"].clear()
    assert request.messages == expected
    assert "response_format" not in payload


@pytest.mark.parametrize("key", ["model", "messages", "tools"])
def test_reserved_request_parameters_fail(key):
    with pytest.raises(ValueError):
        ModelRequest([], parameters={key: "invalid"})


@pytest.mark.parametrize("base,expected", [
    ("", "https://api.openai.com/v1/chat/completions"),
    ("https://p.test/v1/", "https://p.test/v1/chat/completions"),
    ("https://p.test/v1/chat/completions", "https://p.test/v1/chat/completions"),
])
def test_provider_url(base, expected):
    assert chat_completions_url(base) == expected


@pytest.mark.parametrize("kwargs", [
    {"timeout_sec": 0}, {"provider_timeout_retries": -1},
    {"provider_connection_retries": -1}, {"provider_timeout_retries": True},
])
def test_invalid_transport_settings(kwargs):
    with pytest.raises(ValueError):
        ProviderSettings(**kwargs)


def test_unknown_provider_fails_before_io(monkeypatch):
    monkeypatch.setattr("src.llm.client.request_bytes", lambda *a, **k: pytest.fail("no I/O"))
    with pytest.raises(ValueError, match="Unsupported"):
        DefaultModelClient(ProviderSettings(provider="unknown")).request(
            ModelRequest([]), error_prefix="test")


def test_client_no_hidden_state_and_safe_audit(monkeypatch):
    image = "data:image/png;base64," + base64.b64encode(b"example image bytes").decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "CURRENT observation_id=obs_1"},
        {"type": "image_url", "image_url": {"url": image}}]}]
    original = copy.deepcopy(messages)
    calls = []

    def transport(url, **kwargs):
        calls.append((url, kwargs))
        kwargs["transport_trace"].update(attempt_count=1, retry_count=0, final_status="success")
        return json.dumps({"choices": [{"message": {
            "content": None, "tool_calls": [], "reasoning_content": "private thought"},
            "finish_reason": "tool_calls"}]}).encode()

    monkeypatch.setattr("src.llm.client.request_bytes", transport)
    settings = ProviderSettings(model_name="test", api_key="secret-key")
    client = DefaultModelClient(settings)
    result = client.request(ModelRequest(messages), error_prefix="test")
    client.request(ModelRequest([]), error_prefix="test")
    assert messages == original
    assert json.loads(calls[0][1]["body"])["messages"] == messages
    assert json.loads(calls[1][1]["body"])["messages"] == []
    assert calls[0][1]["headers"]["Authorization"] == "Bearer secret-key"
    assert result.media[0]["data"] == b"example image bytes"
    assert result.media[0]["request_path"] == ["messages", 0, "content", 1, "image_url", "url"]
    assert "timeline_role" not in result.media[0]
    audit = json.dumps([result.request_snapshot, result.response_snapshot, result.transport_trace])
    assert all(text not in audit for text in [image, "secret-key", "private thought"])
    assert result.response_snapshot["message"]["content"] is None
    assert "secret-key" not in repr(settings)


def test_snapshot_filters_private_fields_without_mutation():
    source = {"api_key": "secret", "reasoning_content": "hidden",
              "messages": [{"role": "user", "content": "public"}]}
    original = copy.deepcopy(source)
    snapshot, media = snapshot_request(source)
    assert snapshot == {"messages": source["messages"]}
    assert not media
    assert source == original


def test_snapshot_preserves_arbitrary_nested_tool_fields():
    payload = {"tools": [{"type": "function", "function": {
        "name": "example", "parameters": {"properties": {
            "analysis": {"type": "string"}}}}}]}
    snapshot, _ = snapshot_request(payload)
    assert snapshot == payload


def test_llm_dependency_boundary():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("src.agent", "src.runtime", "src.memory", "src.plugins")
    for path in (root / "src/llm").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        assert not any(name.startswith(forbidden) for name in imports), path
    assert not (root / "src/agent" / "provider_http.py").exists()


@pytest.mark.parametrize("content,expected", [
    ("hello", "hello"), ("", ""), (None, None),
    ([{"type": "text", "text": "structured"}], None),
])
def test_response_text_reads_only_public_first_choice(content, expected):
    response = ModelResponse({"choices": [
        {"message": {"content": content, "reasoning_content": "private"}},
        {"message": {"content": "second choice"}},
    ]})
    assert response.text == expected


@pytest.mark.parametrize("data", [
    {}, {"choices": None}, {"choices": []}, {"choices": [None]},
    {"choices": [{"message": None}]}, {"choices": [{"message": {"tool_calls": None}}]},
])
def test_response_missing_fields_have_no_invented_evidence(data):
    response = ModelResponse(data)
    assert response.text is None
    assert response.tool_calls == []
    assert response.usage is None


def test_response_conveniences_are_read_only_and_nested_copies():
    from dataclasses import FrozenInstanceError

    calls = [{"id": "call_1", "function": {"name": "example", "arguments": "{}"}}]
    response = ModelResponse({"choices": [{"message": {"content": None, "tool_calls": calls}}],
                             "usage": {"total_tokens": 0, "details": {"cached_tokens": 0}}})
    assert response.text is None  # Legal tool-only response, not an error.
    result_calls = response.tool_calls
    result_calls[0]["function"]["arguments"] = "changed"
    result_usage = response.usage
    result_usage["details"]["cached_tokens"] = 99
    assert response.tool_calls[0]["function"]["arguments"] == "{}"
    assert response.usage["details"]["cached_tokens"] == 0
    for attribute, value in (("text", "x"), ("tool_calls", []), ("usage", {})):
        with pytest.raises(FrozenInstanceError):
            setattr(response, attribute, value)


@pytest.mark.parametrize("prefix", [None, "Custom label"])
def test_error_prefix_default_and_override(monkeypatch, prefix):
    captured = {}

    def transport(url, **kwargs):
        captured.update(kwargs)
        return b'{"choices":[{"message":{"content":"hello"}}]}'

    monkeypatch.setattr("src.llm.client.request_bytes", transport)
    client = DefaultModelClient(ProviderSettings())
    kwargs = {} if prefix is None else {"error_prefix": prefix}
    response = client.request(ModelRequest([]), **kwargs)
    assert response.text == "hello"
    assert captured["error_prefix"] == (prefix or "Model request failed")
