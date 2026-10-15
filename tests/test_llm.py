"""Communication and message tooling contracts; no live provider or simulator."""
from __future__ import annotations

import ast
import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.llm import DefaultModelClient, MessageBuilder, ModelRequest, ModelResponse, ProviderSettings
from src.llm.audit import snapshot_request
from src.llm.client import normalize_openai_base_url
from src.llm.providers.openai_compatible import (
    decode_response,
    encode_request,
    is_complete_stream_response,
)
from src.llm.providers.openai_responses import (
    decode_responses_response,
    encode_responses_request,
)


class _SdkValue:
    def __init__(self, value, request_id="req_test"):
        self.value = value
        self._request_id = request_id

    def model_dump(self, *, mode):
        assert mode == "json"
        return copy.deepcopy(self.value)


class _Create:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _RawResponse:
    def __init__(self, body):
        self.body = body
        self.headers = {"x-request-id": "req_stream"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_bytes(self):
        yield self.body


class _FakeSdk:
    def __init__(self, *, chat=None, responses=None, stream=None):
        chat_create = _Create(chat)
        stream_create = _Create(stream)
        chat_create.with_streaming_response = SimpleNamespace(create=stream_create.create)
        self.chat = SimpleNamespace(completions=chat_create)
        self.responses_create = _Create(responses)
        self.responses = SimpleNamespace(create=self.responses_create.create)


def _chat_response(content="hello", **message):
    return _SdkValue({"choices": [{"message": {
        "role": "assistant", "content": content, **message,
    }, "finish_reason": "stop"}]})


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


@pytest.mark.parametrize("key", ["model", "messages", "tools", "stream"])
def test_reserved_request_parameters_fail(key):
    with pytest.raises(ValueError):
        ModelRequest([], parameters={key: "invalid"})


@pytest.mark.parametrize("base,expected", [
    ("", ""),
    ("https://p.test/v1/", "https://p.test/v1"),
    ("https://p.test/v1/chat/completions", "https://p.test/v1"),
    ("https://p.test/v1/responses", "https://p.test/v1"),
])
def test_sdk_base_url(base, expected):
    assert normalize_openai_base_url(base) == expected


def test_responses_encoding_preserves_order_tools_images_and_results():
    image = "data:image/png;base64," + base64.b64encode(b"pixels").decode()
    request = ModelRequest(
        messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": [
                {"type": "text", "text": "CURRENT"},
                {"type": "image_url", "image_url": {"url": image}},
            ]},
            {"role": "assistant", "content": "I will inspect.", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
        tools=[{"type": "function", "function": {
            "name": "inspect", "description": "Inspect.",
            "parameters": {"type": "object", "properties": {}},
        }}],
        parameters={
            "tool_choice": "auto",
            "temperature": 0,
            "reasoning_effort": "low",
            "max_tokens": 100,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "result", "strict": True,
                "schema": {"type": "object", "properties": {}},
            }},
        },
    )

    payload = encode_responses_request(
        ProviderSettings(model_name="model", api_protocol="responses"), request
    )

    assert payload["model"] == "model"
    assert payload["store"] is False
    assert payload["input"][1]["content"][1] == {
        "type": "input_image", "image_url": image,
    }
    assert payload["input"][2] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "I will inspect."}],
    }
    assert payload["input"][3] == {
        "type": "function_call", "call_id": "call_1",
        "name": "inspect", "arguments": "{}",
    }
    assert payload["input"][4] == {
        "type": "function_call_output", "call_id": "call_1", "output": "done",
    }
    assert payload["tools"][0]["name"] == "inspect"
    assert "function" not in payload["tools"][0]
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["max_output_tokens"] == 100
    assert payload["text"]["format"]["type"] == "json_schema"


def test_responses_decode_normalizes_public_text_and_function_calls():
    body = json.dumps({
        "id": "resp_1", "model": "model", "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"text": "private"}]},
            {"type": "message", "content": [
                {"type": "output_text", "text": "Checking."},
            ]},
            {"type": "function_call", "call_id": "call_1",
             "name": "inspect", "arguments": "{}"},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    }).encode()

    data = decode_responses_response(body)

    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["choices"][0]["message"] == {
        "role": "assistant", "content": "Checking.",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "inspect", "arguments": "{}",
        }}],
    }
    assert "private" not in json.dumps(data)
    assert data["usage"]["input_tokens"] == 10


def test_responses_client_uses_sdk_resource_and_logical_wire_audit():
    raw = _SdkValue({
        "id": "resp_1", "model": "model", "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "hello"},
        ]}],
    })
    sdk = _FakeSdk(responses=raw)
    client = DefaultModelClient(
        ProviderSettings(model_name="model", api_protocol="responses"),
        sdk_client=sdk,
    )

    result = client.request(ModelRequest(
        [{"role": "user", "content": "hi"}],
        parameters={"reasoning_effort": "low"},
    ))

    sent = sdk.responses_create.calls[0]
    assert sent["input"][0]["content"][0]["text"] == "hi"
    assert sent["reasoning"] == {"effort": "low"}
    assert "messages" not in sent
    assert result.text == "hello"
    assert result.request_snapshot["input"] == sent["input"]


@pytest.mark.parametrize("kwargs", [
    {"timeout_sec": 0}, {"provider_timeout_retries": -1},
    {"provider_connection_retries": -1}, {"provider_timeout_retries": True},
    {"stream": "true"},
])
def test_invalid_transport_settings(kwargs):
    with pytest.raises(ValueError):
        ProviderSettings(**kwargs)


def test_unknown_provider_fails_before_io():
    with pytest.raises(ValueError, match="unsupported model provider"):
        ProviderSettings(provider="unknown")


def test_client_no_hidden_state_and_safe_audit():
    image = "data:image/png;base64," + base64.b64encode(b"example image bytes").decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "CURRENT observation_id=obs_1"},
        {"type": "image_url", "image_url": {"url": image}}]}]
    original = copy.deepcopy(messages)
    sdk = _FakeSdk(chat=_SdkValue({"choices": [{"message": {
        "content": None, "tool_calls": [], "reasoning_content": "private thought"},
        "finish_reason": "tool_calls"}]}))
    settings = ProviderSettings(model_name="test", api_key="secret-key")
    client = DefaultModelClient(settings, sdk_client=sdk)
    result = client.request(ModelRequest(messages), error_prefix="test")
    client.request(ModelRequest([]), error_prefix="test")
    assert messages == original
    assert sdk.chat.completions.calls[0]["messages"] == messages
    assert sdk.chat.completions.calls[1]["messages"] == []
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
    assert not (root / "src/llm" / "transport.py").exists()
    assert not (root / "src/llm/providers/codex_sdk").exists()
    assert not (root / "requirements-codex.txt").exists()


def test_model_callers_do_not_import_urllib_transport():
    root = Path(__file__).resolve().parents[1]
    model_callers = [
        root / "src/llm",
        root / "evaluation/benchmarks/uav_flow/sr_judge.py",
        root / "evaluation/benchmarks/uav_flow/visual_diagnostics.py",
        root / "plugins/research/stmr_map/perception/model_landmark_extractor.py",
        root / "tests/manual/model_multimodal_smoke.py",
        root / "tests/manual/visual_instruction_eval.py",
    ]
    for entry in model_callers:
        paths = entry.rglob("*.py") if entry.is_dir() else [entry]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            assert "urllib.request" not in source, path
            assert "urllib.error" not in source, path


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


def test_sdk_success_uses_existing_model_response_contract():
    client = DefaultModelClient(
        ProviderSettings(), sdk_client=_FakeSdk(chat=_chat_response())
    )
    response = client.request(ModelRequest([]), error_prefix="Custom label")
    assert response.text == "hello"
    assert response.transport_trace["backend"] == "openai_python_sdk"


def test_streaming_client_requests_sse_and_normalizes_tool_calls():
    body = b"".join(
        [
            b'data: {"id":"resp_1","model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
            b'data: {"id":"resp_1","model":"gpt-test","choices":[{"index":0,"delta":{"content":"Checking. ","reasoning_content":"private","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"uav_move_","arguments":"{\\"vx\\":"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"resp_1","model":"gpt-test","choices":[{"index":0,"delta":{"content":"Moving.","tool_calls":[{"index":0,"function":{"name":"body","arguments":"1}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            b'data: {"id":"resp_1","model":"gpt-test","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n\n',
            b'data: [DONE]\n\n',
        ]
    )
    sdk = _FakeSdk(stream=_RawResponse(body))
    result = DefaultModelClient(
        ProviderSettings(model_name="gpt-test", stream=True), sdk_client=sdk
    ).request(ModelRequest([{"role": "user", "content": "move"}]))

    sent = sdk.chat.completions.with_streaming_response.create.__self__.calls[0]
    assert sent["stream"] is True
    assert result.text == "Checking. Moving."
    assert result.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "uav_move_body", "arguments": '{"vx":1}'},
        }
    ]
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    assert result.data["choices"][0]["finish_reason"] == "tool_calls"
    assert "reasoning_content" not in json.dumps(result.data)


def test_stream_decoder_accepts_json_provider_fallback():
    data = decode_response(
        b'{"model":"test","choices":[{"message":{"content":"hello"}}]}',
        stream=True,
    )

    assert data["choices"][0]["message"]["content"] == "hello"


def test_stream_completion_requires_terminal_choice_and_complete_tool_call():
    complete = b"".join(
        [
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"uav_move_body","arguments":"{\\"vx\\":1}"}}]},"finish_reason":"tool_calls"}]}\n\n',
        ]
    )
    missing_finish = complete.replace(b'"tool_calls"}]}', b'null}]}')
    truncated_arguments = complete.replace(b'{\\"vx\\":1}', b'{\\"vx\\":')

    assert is_complete_stream_response(complete) is True
    assert is_complete_stream_response(missing_finish) is False
    assert is_complete_stream_response(truncated_arguments) is False


@pytest.mark.parametrize(
    "body,match",
    [
        (b"data: [DONE]\n\n", "no JSON chunks"),
        (
            b'data: {"model":"test","choices":[],"usage":{"total_tokens":1}}\n\ndata: [DONE]\n\n',
            "no choices",
        ),
    ],
)
def test_stream_decoder_rejects_incomplete_or_error_streams(body, match):
    with pytest.raises(ValueError, match=match):
        decode_response(body, stream=True)


def test_stream_decoder_preserves_provider_errors():
    from src.llm.contracts import ProviderRequestError

    with pytest.raises(ProviderRequestError, match="upstream unavailable") as caught:
        decode_response(b'data: {"error":{"message":"upstream unavailable"}}\n\n', stream=True)
    assert caught.value.category == "provider_error"
    assert caught.value.retryable is False


def test_stream_protocol_error_marks_transport_trace_invalid():
    trace = {}
    client = DefaultModelClient(
        ProviderSettings(stream=True),
        sdk_client=_FakeSdk(stream=_RawResponse(b"data: [DONE]\n\n")),
    )

    with pytest.raises(ValueError, match="no JSON chunks"):
        client.request(ModelRequest([]), transport_trace=trace)

    assert trace["final_status"] == "failed"
    assert trace["failure_category"] == "invalid_response"
