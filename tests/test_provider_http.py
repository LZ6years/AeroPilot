"""Deterministic tests for the official OpenAI SDK transport boundary."""
from __future__ import annotations

import json
import ssl
from types import SimpleNamespace

import httpx2
import openai
import pytest

from src.llm import DefaultModelClient, ModelRequest, ProviderRequestError, ProviderSettings
from src.llm.client import normalize_openai_base_url


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("src.llm.client.time.monotonic", clock.time)
    monkeypatch.setattr("src.llm.client.time.sleep", clock.sleep)
    return clock


class _SdkValue:
    def __init__(self, value, request_id="req_test"):
        self.value = value
        self._request_id = request_id

    def model_dump(self, *, mode):
        assert mode == "json"
        return self.value


class _Create:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _RawResponse:
    def __init__(self, parts, *, error=None, request_id="req_stream"):
        self.parts = parts
        self.error = error
        self.headers = {"x-request-id": request_id}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True
        return False

    def iter_bytes(self):
        yield from self.parts
        if self.error is not None:
            raise self.error


class _FakeSdk:
    def __init__(self, *, chat=None, responses=None, stream=None):
        chat_create = _Create(chat)
        stream_create = _Create(stream)
        chat_create.with_streaming_response = SimpleNamespace(create=stream_create.create)
        self.chat = SimpleNamespace(completions=chat_create)
        self.responses = SimpleNamespace(create=_Create(responses).create)
        self.options = []
        self.closed = False

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def close(self):
        self.closed = True


def _chat_value(content="hello"):
    return _SdkValue({
        "id": "chat_1",
        "model": "model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    })


def _responses_value():
    return _SdkValue({
        "id": "resp_1",
        "model": "model",
        "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "hello"},
        ]}],
    })


@pytest.mark.parametrize("base,expected", [
    ("", ""),
    ("https://gateway.test/v1/", "https://gateway.test/v1"),
    ("https://gateway.test/v1/chat/completions", "https://gateway.test/v1"),
    ("https://gateway.test/v1/responses", "https://gateway.test/v1"),
])
def test_base_url_normalization(base, expected):
    assert normalize_openai_base_url(base) == expected


def test_sdk_client_is_lazy_reused_and_receives_configuration(monkeypatch):
    created = []
    sdk = _FakeSdk(chat=_chat_value())

    def constructor(**kwargs):
        created.append(kwargs)
        return sdk

    monkeypatch.setattr("src.llm.client.OpenAI", constructor)
    settings = ProviderSettings(
        model_name="model",
        api_key="secret",
        base_url="https://gateway.test/v1/chat/completions",
        timeout_sec=45,
        provider_timeout_retries=1,
        provider_connection_retries=3,
    )
    client = DefaultModelClient(settings)
    assert client.request(ModelRequest([{"role": "user", "content": "one"}])).text == "hello"
    assert client.request(ModelRequest([{"role": "user", "content": "two"}])).text == "hello"
    assert created == [{
        "api_key": "secret",
        "base_url": "https://gateway.test/v1",
        "timeout": 45,
        "max_retries": 0,
    }]
    assert len(sdk.chat.completions.calls) == 2


def test_provider_extensions_use_extra_body_and_audit_is_sdk_owned():
    sdk = _FakeSdk(chat=_chat_value())
    trace = {}
    result = DefaultModelClient(
        ProviderSettings(model_name="model"), sdk_client=sdk
    ).request(
        ModelRequest(
            [{"role": "user", "content": "hello"}],
            parameters={"temperature": 0, "enable_thinking": False},
        ),
        transport_trace=trace,
    )
    sent = sdk.chat.completions.calls[0]
    assert sent["temperature"] == 0
    assert sent["extra_body"] == {"enable_thinking": False}
    assert trace["backend"] == "openai_python_sdk"
    assert trace["sdk_version"] == openai.__version__
    assert trace["sdk_max_retries"] == 0
    assert trace["request_id"] == "req_test"
    assert trace["final_status"] == "success"
    assert trace["attempt_count"] == 1
    assert result.text == "hello"


def test_responses_uses_sdk_resource_and_preserves_public_instructions():
    raw = _responses_value()
    raw.value["instructions"] = "provider-visible instruction"
    create = _Create(raw)
    sdk = _FakeSdk(chat=_chat_value())
    sdk.responses = SimpleNamespace(create=create.create)
    result = DefaultModelClient(
        ProviderSettings(model_name="model", api_protocol="responses"),
        sdk_client=sdk,
    ).request(ModelRequest([{"role": "user", "content": "hello"}],
                           parameters={"reasoning_effort": "low"}))
    sent = create.calls[0]
    assert sent["store"] is False
    assert sent["reasoning"] == {"effort": "low"}
    assert "messages" not in sent and "instructions" not in sent
    assert result.data["instructions"] == "provider-visible instruction"
    assert result.text == "hello"


@pytest.mark.parametrize("kind,expected", [
    ("timeout", "timeout"),
    ("connection", "connection"),
    ("tls", "tls_certificate"),
])
def test_sdk_transport_errors_are_safely_classified(kind, expected):
    request = httpx2.Request("POST", "https://gateway.test/v1/responses")
    if kind == "timeout":
        error = openai.APITimeoutError(request=request)
    else:
        error = openai.APIConnectionError(request=request)
        if kind == "tls":
            error.__cause__ = ssl.SSLCertVerificationError("bad certificate")
    create = _Create(error=error)
    sdk = _FakeSdk(chat=_chat_value())
    sdk.responses = SimpleNamespace(create=create.create)
    trace = {}
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(
            ProviderSettings(api_protocol="responses"), sdk_client=sdk
        ).request(ModelRequest([]), error_prefix="probe", transport_trace=trace)
    assert raised.value.category == expected
    assert trace["failure_category"] == expected
    assert trace["final_status"] == "failed"
    assert "Authorization" not in json.dumps(trace)


def test_sdk_status_error_records_status_and_request_id_without_body():
    request = httpx2.Request("POST", "https://gateway.test/v1/responses")
    response = httpx2.Response(429, request=request, headers={"x-request-id": "req_429"})
    error = openai.APIStatusError(
        "rate limited",
        response=response,
        body={"secret": "must-not-be-audited"},
    )
    create = _Create(error=error)
    sdk = _FakeSdk(chat=_chat_value())
    sdk.responses = SimpleNamespace(create=create.create)
    trace = {}
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(
            ProviderSettings(api_protocol="responses"), sdk_client=sdk
        ).request(ModelRequest([]), transport_trace=trace)
    assert raised.value.category == "http_status"
    assert raised.value.status_code == 429
    assert trace["status_code"] == 429
    assert trace["request_id"] == "req_429"
    assert "must-not-be-audited" not in json.dumps(trace)


def test_streaming_uses_sdk_raw_response_and_merges_tool_call():
    parts = [
        b'data: {"id":"resp","model":"model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","function":{"name":"ping","arguments":"{"}}]},"finish_reason":null}]}\n\n',
        b'data: {"id":"resp","model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
        b'data: [DONE]\n\n',
    ]
    raw = _RawResponse(parts)
    sdk = _FakeSdk(stream=raw)
    result = DefaultModelClient(
        ProviderSettings(model_name="model", stream=True), sdk_client=sdk
    ).request(ModelRequest([{"role": "user", "content": "ping"}]))
    assert sdk.chat.completions.with_streaming_response is not None
    assert result.tool_calls[0]["function"] == {"name": "ping", "arguments": "{}"}
    assert result.transport_trace["request_id"] == "req_stream"


def test_complete_stream_survives_tail_disconnect_without_restart():
    complete = b'data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
    raw = _RawResponse([complete], error=ConnectionError("tail closed"))
    result = DefaultModelClient(
        ProviderSettings(stream=True), sdk_client=_FakeSdk(stream=raw)
    ).request(ModelRequest([]))
    assert result.text == "done"
    assert result.transport_trace["recovered_complete_stream"] is True


def test_truncated_stream_exhaustion_has_real_attempts_and_closes_responses():
    truncated = b'data: {"choices":[{"index":0,"delta":{"content":"part"},"finish_reason":null}]}\n\n'
    raw = _RawResponse([truncated], error=ConnectionError("tail closed"))
    sdk = _FakeSdk(stream=raw)
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(
            ProviderSettings(stream=True), sdk_client=sdk
        ).request(ModelRequest([]))
    assert len(sdk.chat.completions.with_streaming_response.create.__self__.calls) == 3
    assert raised.value.retryable and raised.value.attempts == 3
    assert raised.value.exhaustion_reason == "retry_count"
    assert raw.closed and not sdk.closed


@pytest.mark.parametrize("failures", [2, 7])
def test_stream_tls_recovers_without_publishing_partial_tool_calls(failures, clock):
    partial = b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"discard","function":{"name":"fly","arguments":"{"}}]},"finish_reason":null}]}\n\n'
    complete = b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"once","function":{"name":"fly","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n'
    broken = [_RawResponse([partial], error=ssl.SSLEOFError("TLS EOF")) for _ in range(failures)]
    pending = iter([*broken, _RawResponse([complete])])
    calls = []
    sdk = _FakeSdk()

    def create(**kwargs):
        calls.append(kwargs)
        return next(pending)

    sdk.chat.completions.with_streaming_response.create = create
    result = DefaultModelClient(ProviderSettings(stream=True, provider_connection_retries=8),
                                sdk_client=sdk).request(ModelRequest([]))
    assert result.tool_calls[0]["id"] == "once"
    assert len(result.tool_calls) == 1
    assert "discard" not in json.dumps(result.data)
    assert len(calls) == failures + 1 and all(call == calls[0] for call in calls)
    assert all(response.closed for response in broken)
    assert result.transport_trace["recovered"]
    assert result.transport_trace["attempt_count"] == failures + 1
    assert clock.sleeps == [1, 2, 4, 8, 16, 30, 30][:failures]
    assert all(option["max_retries"] == 0 for option in sdk.options)
    assert not sdk.closed


def test_deadline_caps_attempt_timeout_and_reports_exhaustion(clock):
    sdk = _FakeSdk()

    def create(**kwargs):
        clock.now += sdk.options[-1]["timeout"]
        raise ssl.SSLEOFError("TLS EOF")

    sdk.chat.completions.create = create
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(ProviderSettings(timeout_sec=100, retry_deadline_sec=600,
                                            provider_connection_retries=8), sdk_client=sdk).request(ModelRequest([]))
    assert raised.value.exhaustion_reason == "deadline"
    assert raised.value.elapsed_seconds == 600
    assert raised.value.attempts == 6
    assert sdk.options[-1]["timeout"] == 69


def test_timeout_and_connection_budgets_are_independent(clock):
    sdk = _FakeSdk()
    errors = iter([TimeoutError("slow"), ssl.SSLEOFError("EOF"), TimeoutError("slow again")])
    sdk.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(next(errors))
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(ProviderSettings(provider_timeout_retries=1,
                                            provider_connection_retries=8), sdk_client=sdk).request(ModelRequest([]))
    assert raised.value.category == "timeout" and raised.value.attempts == 3
    assert clock.sleeps == [1, 1]


def test_exhausted_failure_keeps_attempt_details_separate_from_error_counter(clock):
    sdk = _FakeSdk()
    def create(**kwargs):
        clock.now += 2
        raise TimeoutError("slow")
    sdk.chat.completions.create = create
    trace = {}
    with pytest.raises(ProviderRequestError):
        DefaultModelClient(ProviderSettings(provider_timeout_retries=1), sdk_client=sdk).request(ModelRequest([]), transport_trace=trace)
    assert trace["error_attempt_count"] == trace["attempt_count"] == 2
    assert len(trace["attempts"]) == 2
    assert [a["elapsed_seconds"] for a in trace["attempts"]] == [2, 2]
    assert trace["attempts"][0]["backoff_seconds"] == 1
    assert trace["elapsed_seconds"] == 5


def test_overshot_backoff_deadline_also_preserves_attempt_details(clock, monkeypatch):
    sdk = _FakeSdk()
    sdk.chat.completions.create = _Create(error=ConnectionError("offline")).create
    monkeypatch.setattr("src.llm.client.time.sleep", lambda _: setattr(clock, "now", 3))
    trace = {}
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(ProviderSettings(retry_deadline_sec=2), sdk_client=sdk).request(ModelRequest([]), transport_trace=trace)
    assert raised.value.exhaustion_reason == "deadline"
    assert trace["error_attempt_count"] == 1 and len(trace["attempts"]) == 1


def test_compactor_timeout_is_not_retried(clock):
    sdk = _FakeSdk()
    sdk.chat.completions.create = _Create(error=TimeoutError("slow")).create
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(ProviderSettings(timeout_sec=300, retry_deadline_sec=300,
                                            provider_timeout_retries=0,
                                            provider_connection_retries=1), sdk_client=sdk).request(ModelRequest([]))
    assert raised.value.attempts == 1 and clock.sleeps == []


def test_certificate_failure_is_not_retried(clock):
    sdk = _FakeSdk()
    error = httpx2.ReadError("read failed")
    error.__cause__ = ssl.SSLCertVerificationError("bad certificate")
    sdk.chat.completions.create = _Create(error=error).create
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(ProviderSettings(provider_connection_retries=8), sdk_client=sdk).request(ModelRequest([]))
    assert raised.value.category == "tls_certificate"
    assert not raised.value.retryable and raised.value.attempts == 1 and clock.sleeps == []


def test_eight_connection_retries_are_nine_wire_attempts(clock):
    sdk = _FakeSdk()
    create = _Create(error=ssl.SSLEOFError("EOF"))
    sdk.chat.completions.create = create.create
    with pytest.raises(ProviderRequestError) as raised:
        DefaultModelClient(ProviderSettings(provider_connection_retries=8), sdk_client=sdk).request(ModelRequest([]))
    assert raised.value.attempts == len(create.calls) == 9
    assert clock.sleeps == [1, 2, 4, 8, 16, 30, 30, 30]
    assert raised.value.exhaustion_reason == "retry_count"


def test_complete_response_survives_context_close_tls_failure(clock):
    class TailClose(_RawResponse):
        def __exit__(self, *_):
            super().__exit__(*_)
            raise ssl.SSLEOFError("TLS during response close")
    complete = b'data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
    sdk = _FakeSdk(stream=TailClose([complete]))
    result = DefaultModelClient(ProviderSettings(stream=True), sdk_client=sdk).request(ModelRequest([]))
    assert result.text == "done" and result.transport_trace["attempt_count"] == 1
    assert clock.sleeps == []


def test_owned_connections_are_rebuilt_but_injected_clients_are_kept(monkeypatch, clock):
    broken = _FakeSdk()
    broken.chat.completions.create = _Create(error=ssl.SSLEOFError("TLS EOF")).create
    working = _FakeSdk(chat=_chat_value())
    clients = iter([broken, working])
    monkeypatch.setattr("src.llm.client.OpenAI", lambda **kwargs: next(clients))
    result = DefaultModelClient(ProviderSettings(api_key="test-only")).request(ModelRequest([]))
    assert result.text == "hello" and broken.closed and not working.closed
