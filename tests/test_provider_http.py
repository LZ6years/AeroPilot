"""Deterministic tests for shared OpenAI-compatible HTTP retry behavior."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message

import pytest

from src.agent.provider_http import request_openai_compatible_json
from src.runtime.configs.model_config import ModelConfig


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _config() -> ModelConfig:
    return ModelConfig(
        provider="openai_compatible",
        model_name="model",
        api_key="secret",
        base_url="https://provider.test/v1",
        timeout_sec=12,
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

    monkeypatch.setattr("src.agent.provider_http.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.agent.provider_http.time.sleep", sleeps.append)

    result = request_openai_compatible_json(
        _config(), {"messages": []}, error_prefix="Planner failed"
    )

    assert result == {"ok": True}
    assert [item[1] for item in calls] == [12, 12, 12]
    assert sleeps == [2.0, 4.0]
    assert calls[0][0].get_header("Authorization") == "Bearer secret"


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

    monkeypatch.setattr("src.agent.provider_http.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.agent.provider_http.time.sleep", sleeps.append)

    request_openai_compatible_json(
        _config(), {"messages": []}, error_prefix="Parser failed"
    )

    assert sleeps == [7.0, 60.0]


def test_non_transient_http_error_fails_without_sleep(monkeypatch):
    sleeps = []

    def fake_urlopen(request, timeout):
        raise _http_error(400)

    monkeypatch.setattr("src.agent.provider_http.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.agent.provider_http.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match=r"Parser failed: HTTP 400"):
        request_openai_compatible_json(
            _config(), {"messages": []}, error_prefix="Parser failed"
        )

    assert sleeps == []


def test_transient_error_exhaustion_keeps_final_body_and_prefix(monkeypatch):
    sleeps = []
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise _http_error(429)

    monkeypatch.setattr("src.agent.provider_http.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.agent.provider_http.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match=r"Native request failed: HTTP 429") as raised:
        request_openai_compatible_json(
            _config(), {"messages": []}, error_prefix="Native request failed"
        )

    assert calls == 5
    assert sleeps == [2.0, 4.0, 8.0, 16.0]
    assert '"capacity"' in str(raised.value)


def test_network_error_is_not_retried(monkeypatch):
    sleeps = []

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("src.agent.provider_http.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("src.agent.provider_http.time.sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="Updater failed:.*offline"):
        request_openai_compatible_json(
            _config(), {"messages": []}, error_prefix="Updater failed"
        )

    assert sleeps == []
