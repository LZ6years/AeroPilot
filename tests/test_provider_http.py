"""Deterministic tests for shared OpenAI-compatible HTTP retry behavior."""

from __future__ import annotations

import http.client
import io
import json
import ssl
import urllib.error
from email.message import Message

import pytest

from src.llm import DefaultModelClient, ModelRequest, ProviderRequestError
from src.runtime.configs.model_config import ModelConfig


def _request_json(config, payload, *, error_prefix, transport_trace=None):
    """Exercise the real shared client; only HTTP is stubbed by these tests."""
    request = ModelRequest(
        messages=payload.get("messages", []),
        tools=payload.get("tools"),
        parameters={key: value for key, value in payload.items()
                    if key not in {"model", "messages", "tools"}},
    )
    return DefaultModelClient(config.to_provider_settings()).request(
        request, error_prefix=error_prefix, transport_trace=transport_trace,
    ).data


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _config(**overrides) -> ModelConfig:
    values = {
        "provider": "openai_compatible",
        "model_name": "model",
        "api_key": "secret",
        "base_url": "https://provider.test/v1",
        "timeout_sec": 12,
        "provider_timeout_retries": 1,
        "provider_connection_retries": 2,
    }
    values.update(overrides)
    return ModelConfig(
        **values,
    )


def _http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://provider.test/v1/chat/completions",
        code,
        "provider error",
        headers,
        io.BytesIO(b'{"error":"capacity"}'),
    )


def test_transient_errors_use_exponential_delays_then_succeed(monkeypatch):
    outcomes = [_http_error(429), _http_error(503), _Response({"ok": True})]
    calls = []
    sleeps = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    result = _request_json(
        _config(), {"messages": []}, error_prefix="Planner failed"
    )

    assert result == {"ok": True}
    assert [item[1] for item in calls] == [12, 12, 12]
    assert sleeps == [2.0, 4.0]
    assert calls[0][0].get_header("Authorization") == "Bearer secret"
    assert [call[0].data for call in calls] == [calls[0][0].data] * 3


def test_retry_after_takes_precedence_and_is_bounded(monkeypatch):
    outcomes = [
        _http_error(429, retry_after="7"),
        _http_error(429, retry_after="999"),
        _Response({"ok": True}),
    ]
    sleeps = []

    def fake_urlopen(request, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    _request_json(
        _config(), {"messages": []}, error_prefix="Parser failed"
    )

    assert sleeps == [7.0, 60.0]


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_non_transient_http_error_fails_without_sleep(monkeypatch, status_code):
    sleeps = []

    def fake_urlopen(request, timeout):
        raise _http_error(status_code)

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    with pytest.raises(
        ProviderRequestError,
        match=rf"Parser failed: HTTP {status_code}",
    ) as raised:
        _request_json(
            _config(), {"messages": []}, error_prefix="Parser failed"
        )

    assert sleeps == []
    assert raised.value.retry_exhausted is False


def test_transient_error_exhaustion_keeps_final_body_and_prefix(monkeypatch):
    sleeps = []
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise _http_error(429)

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match=r"Native request failed: HTTP 429") as raised:
        _request_json(
            _config(), {"messages": []}, error_prefix="Native request failed"
        )

    assert calls == 5
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    assert '"capacity"' in str(raised.value)


def test_timeout_retries_once_then_succeeds_with_safe_trace(monkeypatch):
    outcomes = [TimeoutError("read operation timed out"), _Response({"ok": True})]
    sleeps = []
    calls = []
    trace = {}

    def fake_urlopen(request, timeout):
        calls.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    result = _request_json(
        _config(),
        {"messages": [{"content": "data:image/png;base64,private"}]},
        error_prefix="Updater failed",
        transport_trace=trace,
    )

    assert result == {"ok": True}
    assert sleeps == [2.0]
    assert len(calls) == 2
    assert trace == {
        "attempt_count": 2,
        "retry_count": 1,
        "final_status": "success",
        "retries": [
            {
                "attempt": 1,
                "category": "timeout",
                "error_type": "TimeoutError",
                "delay_sec": 2.0,
            }
        ],
    }
    assert "secret" not in json.dumps(trace)
    assert "data:image" not in json.dumps(trace)


def test_timeout_retry_exhaustion_raises_typed_error(monkeypatch):
    sleeps = []
    calls = 0
    trace = {}

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise TimeoutError("read operation timed out")

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    with pytest.raises(ProviderRequestError, match="read operation timed out") as raised:
        _request_json(
            _config(),
            {"messages": []},
            error_prefix="Planner failed",
            transport_trace=trace,
        )

    assert calls == 2
    assert sleeps == [2.0]
    assert raised.value.category == "timeout"
    assert raised.value.attempts == 2
    assert raised.value.retry_exhausted is True
    assert trace["final_status"] == "failed"
    assert trace["failure_category"] == "timeout"


def test_remote_disconnect_retries_twice_then_succeeds(monkeypatch):
    outcomes = [
        http.client.RemoteDisconnected("closed once"),
        ConnectionResetError("reset twice"),
        _Response({"ok": True}),
    ]
    sleeps = []

    def fake_urlopen(request, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    result = _request_json(
        _config(), {"messages": []}, error_prefix="Planner failed"
    )

    assert result == {"ok": True}
    assert sleeps == [2.0, 4.0]


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("temporary DNS failure"),
        BrokenPipeError("broken pipe"),
        http.client.IncompleteRead(b"partial", 10),
        ssl.SSLEOFError("unexpected EOF"),
    ],
    ids=["url-error", "broken-pipe", "incomplete-read", "tls-eof"],
)
def test_supported_connection_failures_retry_then_succeed(monkeypatch, failure):
    outcomes = [failure, _Response({"ok": True})]
    sleeps = []

    def fake_urlopen(request, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    result = _request_json(
        _config(), {"messages": []}, error_prefix="Planner failed"
    )

    assert result == {"ok": True}
    assert sleeps == [2.0]


def test_connection_retry_exhaustion_wraps_remote_disconnect(monkeypatch):
    sleeps = []
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise http.client.RemoteDisconnected("closed")

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    with pytest.raises(ProviderRequestError) as raised:
        _request_json(
            _config(), {"messages": []}, error_prefix="Planner failed"
        )

    assert calls == 3
    assert sleeps == [2.0, 4.0]
    assert raised.value.category == "connection"
    assert raised.value.attempts == 3
    assert raised.value.retry_exhausted is True


def test_mixed_transient_failures_respect_global_retry_ceiling(monkeypatch):
    outcomes = [
        _http_error(503),
        http.client.RemoteDisconnected("closed"),
        _http_error(429),
        ConnectionResetError("reset"),
        TimeoutError("timed out"),
    ]
    sleeps = []

    def fake_urlopen(request, timeout):
        raise outcomes.pop(0)

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    with pytest.raises(ProviderRequestError) as raised:
        _request_json(
            _config(), {"messages": []}, error_prefix="Planner failed"
        )

    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    assert raised.value.category == "timeout"
    assert raised.value.attempts == 5


def test_certificate_verification_error_is_not_retried(monkeypatch):
    sleeps = []
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.URLError(
            ssl.SSLCertVerificationError("certificate verify failed")
        )

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    with pytest.raises(ProviderRequestError) as raised:
        _request_json(
            _config(), {"messages": []}, error_prefix="Planner failed"
        )

    assert calls == 1
    assert sleeps == []
    assert raised.value.category == "tls"
    assert raised.value.retry_exhausted is False


def test_invalid_json_response_is_not_retried(monkeypatch):
    class InvalidResponse(_Response):
        def read(self) -> bytes:
            return b"not-json"

    calls = 0
    sleeps = []
    trace = {}

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return InvalidResponse({})

    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)

    with pytest.raises(json.JSONDecodeError):
        _request_json(
            _config(),
            {"messages": []},
            error_prefix="Planner failed",
            transport_trace=trace,
        )

    assert calls == 1
    assert sleeps == []
    assert trace["final_status"] == "failed"
    assert trace["failure_category"] == "invalid_response"
