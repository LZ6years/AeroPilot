"""Stateless model contract backed by the official OpenAI Python SDK."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
import ssl
import random
import time
from typing import Any

import openai
import httpx2
from openai import OpenAI

from src.llm.audit import snapshot_request
from src.llm.recovery import CURRENT_RECOVERY, RequestGuard, retry_after_seconds
from src.llm.contracts import (
    DEFAULT_ERROR_PREFIX,
    ModelRequest,
    ModelResponse,
    ProviderRequestError,
    ProviderSettings,
)
from src.llm.providers.openai_compatible import (
    decode_response,
    encode_request,
    is_complete_stream_response,
    public_response_snapshot,
)
from src.llm.providers.openai_responses import (
    encode_responses_request,
    normalize_responses_response,
)


_CHAT_SDK_PARAMETERS = {
    "audio", "frequency_penalty", "function_call", "functions", "logit_bias",
    "logprobs", "max_completion_tokens", "max_tokens", "messages", "metadata",
    "modalities", "model", "moderation", "n", "parallel_tool_calls", "prediction",
    "presence_penalty", "prompt_cache_key", "prompt_cache_options",
    "prompt_cache_retention", "reasoning_effort", "response_format",
    "safety_identifier", "seed", "service_tier", "stop", "store", "stream",
    "stream_options", "temperature", "tool_choice", "tools", "top_logprobs",
    "top_p", "user", "verbosity", "web_search_options",
}

_RESPONSES_SDK_PARAMETERS = {
    "access_programs", "background", "context_management", "conversation", "include",
    "input", "instructions", "max_output_tokens", "max_tool_calls", "metadata", "model",
    "moderation", "parallel_tool_calls", "previous_response_id", "prompt",
    "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning",
    "safety_identifier", "service_tier", "store", "stream", "stream_options",
    "temperature", "text", "tool_choice", "tools", "top_logprobs", "top_p",
    "truncation", "user",
}


def normalize_openai_base_url(base_url: str) -> str:
    """Return an SDK base URL while accepting legacy endpoint URLs."""

    normalized = base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].rstrip("/")
            break
    return normalized


class DefaultModelClient:
    """One configured SDK client, with no conversation memory or business routing."""

    def __init__(self, settings: ProviderSettings, *, sdk_client: Any | None = None,
                 clock: Callable[[], float] | None = None,
                 sleep: Callable[[float], None] | None = None) -> None:
        self.settings = settings
        self._sdk_client = sdk_client
        self._owns_client = sdk_client is None
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep

    def request(
        self,
        request: ModelRequest,
        *,
        error_prefix: str = DEFAULT_ERROR_PREFIX,
        transport_trace: dict[str, Any] | None = None,
    ) -> ModelResponse:
        responses_api = self.settings.api_protocol == "responses"
        payload = (
            encode_responses_request(self.settings, request)
            if responses_api
            else encode_request(self.settings, request)
        )
        snapshot, media = snapshot_request(payload)
        trace = transport_trace if transport_trace is not None else {}
        trace.clear()
        trace.update(
            backend="openai_python_sdk",
            sdk_version=openai.__version__,
            sdk_max_retries=0,
            connection_retries=self.settings.provider_connection_retries,
            timeout_retries=self.settings.provider_timeout_retries,
            retry_deadline_sec=self.settings.retry_deadline_sec,
            attempts=[],
            attempt_count=0,
            final_status="pending",
        )
        started = self._clock()
        policy = CURRENT_RECOVERY.get()
        deadline = started + min(self.settings.retry_deadline_sec, policy.deadline_sec) if policy else started + self.settings.retry_deadline_sec
        if policy:
            trace.update(retry_deadline_sec=deadline-started, max_http_attempts=policy.max_attempts)
        used = {"timeout": 0, "connection": 0}
        try:
            while True:
                timeout = min(self.settings.timeout_sec, max(0.001, deadline - self._clock()))
                attempt_started = self._clock()
                attempt = {"number": trace["attempt_count"] + 1, "timeout_sec": timeout,
                           "response_headers_at": None, "first_chunk_at": None,
                           "last_chunk_at": None, "response_bytes": 0}
                trace["attempts"].append(attempt)
                trace["attempt_count"] += 1
                guard = RequestGuard(policy, deadline, self._clock) if policy else None
                try:
                    sdk = self._client()
                    options = getattr(sdk, "with_options", None)
                    if callable(options):
                        sdk = options(max_retries=0, timeout=timeout)
                    def exchange():
                        if responses_api:
                            raw_response, request_id = self._request_responses(payload, sdk)
                            return normalize_responses_response(raw_response), request_id
                        if self.settings.stream:
                            data, request_id, recovered = self._request_chat_stream(payload, sdk, deadline, attempt, guard)
                            if recovered:
                                trace["recovered_complete_stream"] = True
                            return data, request_id
                        return self._request_chat(payload, sdk)
                    data, request_id = guard.run(exchange, sdk.close) if guard else exchange()
                    if request_id:
                        trace["request_id"] = request_id
                    attempt.update(status="success", elapsed_seconds=self._clock() - attempt_started)
                    break
                except (ValueError, UnicodeDecodeError, TypeError):
                    attempt.update(status="failed", category="invalid_response",
                                   elapsed_seconds=self._clock() - attempt_started)
                    trace.update(final_status="failed", failure_category="invalid_response")
                    raise
                except Exception as exc:
                    error = self._provider_error(exc, error_prefix=error_prefix)
                    error.attempts = trace["attempt_count"] if error.attempts else 0
                    error.elapsed_seconds = self._clock() - started
                    attempt.update(status="failed", **error.to_dict())
                    attempt["exception_chain"] = [type(e).__name__ for e in _exception_chain(exc)]
                    attempt["elapsed_seconds"] = self._clock() - attempt_started
                    request_id = getattr(exc, "request_id", None)
                    if request_id:
                        trace["request_id"] = request_id
                    bucket = "timeout" if error.category == "timeout" else "connection"
                    limit = (self.settings.provider_timeout_retries if bucket == "timeout"
                             else self.settings.provider_connection_retries)
                    delay = (1, 2, 4, 8, 16, 30)[min(used[bucket], 5)]
                    if policy:
                        delay = retry_after_seconds(exc)
                        if delay is None:
                            delay = (5, 15, 30, 60, 60)[min(trace["attempt_count"]-1, 4)] * random.uniform(.8, 1.2)
                    remaining = deadline - self._clock()
                    reason = ("non_retryable" if not error.retryable else
                              "retry_count" if (trace["attempt_count"] >= policy.max_attempts if policy else used[bucket] >= limit) else
                              "deadline" if remaining <= (0 if policy else delay) else None)
                    # Waiting consumes the remaining deadline, never exceeds it.
                    delay = min(delay, max(0, remaining))
                    self._close_failed_response(exc)
                    if reason is not None:
                        error.retry_exhausted = error.retryable
                        error.exhaustion_reason = error.exhaustion_reason or reason
                        fields = error.to_dict()
                        fields["error_attempt_count"] = fields.pop("attempts", 0)
                        trace.update(final_status="failed", failure_category=error.category, **fields)
                        raise error from exc
                    used[bucket] += 1
                    attempt["backoff_seconds"] = delay
                    if not guard:
                        self._reset_connection()
                    elif self._owns_client:
                        # Guard closes the old SDK outside the foreground thread.
                        self._sdk_client = None
                    backoff_started = self._clock()
                    if guard:
                        try:
                            guard.sleep(delay, self._sleep)
                        except ProviderRequestError as cancelled:
                            cancelled.attempts = trace["attempt_count"]
                            trace.update(final_status="failed", failure_category=cancelled.category,
                                         exhaustion_reason=cancelled.exhaustion_reason)
                            raise
                        finally:
                            attempt["backoff_elapsed_seconds"] = self._clock()-backoff_started
                    else:
                        self._sleep(delay)
                    # A simulated or interrupted scheduler can overshoot its sleep.
                    if self._clock() >= deadline:
                        error.retry_exhausted = True
                        error.exhaustion_reason = "deadline"
                        error.elapsed_seconds = self._clock() - started
                        fields = error.to_dict()
                        fields["error_attempt_count"] = fields.pop("attempts", 0)
                        trace.update(final_status="failed", failure_category=error.category, **fields)
                        raise error from exc
        finally:
            trace["elapsed_seconds"] = self._clock() - started

        trace["final_status"] = "success"
        trace["recovered"] = trace["attempt_count"] > 1
        return ModelResponse(
            data,
            dict(trace),
            snapshot,
            media,
            public_response_snapshot(data),
        )

    def _client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        if not self.settings.api_key:
            raise ProviderRequestError(
                "Model request failed: API key is not configured",
                category="configuration",
                attempts=0,
                retry_exhausted=False,
            )
        kwargs: dict[str, Any] = {
            "api_key": self.settings.api_key,
            "timeout": self.settings.timeout_sec,
            "max_retries": 0,
        }
        base_url = normalize_openai_base_url(self.settings.base_url)
        if base_url:
            kwargs["base_url"] = base_url
        self._sdk_client = OpenAI(**kwargs)
        return self._sdk_client

    def _request_chat(self, payload: dict[str, Any], sdk: Any) -> tuple[dict[str, Any], str | None]:
        response = sdk.chat.completions.create(
            **_sdk_arguments(payload, _CHAT_SDK_PARAMETERS)
        )
        return _model_dump(response), _request_id(response)

    def _request_responses(
        self,
        payload: dict[str, Any],
        sdk: Any,
    ) -> tuple[dict[str, Any], str | None]:
        response = sdk.responses.create(
            **_sdk_arguments(payload, _RESPONSES_SDK_PARAMETERS)
        )
        return _model_dump(response), _request_id(response)

    def _request_chat_stream(
        self,
        payload: dict[str, Any],
        sdk: Any,
        deadline: float | None = None,
        attempt: dict | None = None,
        guard: RequestGuard | None = None,
    ) -> tuple[dict[str, Any], str | None, bool]:
        parts: list[bytes] = []
        recovered = False
        kwargs = _sdk_arguments(payload, _CHAT_SDK_PARAMETERS)
        request_id = None
        try:
            with sdk.chat.completions.with_streaming_response.create(**kwargs) as response:
                if guard:
                    guard.bind(response)
                request_id = _header_request_id(response)
                if attempt is not None:
                    attempt.update(response_headers_at=self._clock(), request_id=request_id,
                                   first_chunk_at=None, last_chunk_at=None, response_bytes=0)
                for part in response.iter_bytes():
                    if guard and guard.abandoned.is_set():
                        raise TimeoutError("Discarded cancelled stream")
                    if attempt is not None:
                        now = self._clock()
                        if attempt["first_chunk_at"] is None:
                            attempt["first_chunk_at"] = now
                        attempt.update(last_chunk_at=now, response_bytes=attempt["response_bytes"]+len(part))
                    parts.append(bytes(part))
                    if deadline is not None and self._clock() >= deadline:
                        raise TimeoutError("request recovery deadline reached while streaming")
        except Exception:
            body = b"".join(parts)
            if not is_complete_stream_response(body):
                raise
            recovered = True
        body = b"".join(parts)
        data = decode_response(body, stream=True)
        choices = data.get("choices") or []
        if not choices or any(choice.get("finish_reason") is None for choice in choices):
            raise ConnectionError("stream ended before a terminal response")
        return data, request_id, recovered

    @staticmethod
    def _close_failed_response(exc: Exception) -> None:
        response = getattr(exc, "response", None)
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass  # Preserve the classified request failure.

    def _reset_connection(self) -> None:
        if self._owns_client and self._sdk_client is not None:
            try:
                self._sdk_client.close()
            except Exception:
                pass
            self._sdk_client = None

    def _provider_error(
        self,
        exc: Exception,
        *,
        error_prefix: str,
    ) -> ProviderRequestError:
        if isinstance(exc, ProviderRequestError):
            return exc
        status_code: int | None = None
        retryable = False
        chain = list(_exception_chain(exc))
        if any(isinstance(item, ssl.SSLCertVerificationError) for item in chain):
            category = "tls_certificate"
        elif any(isinstance(item, (openai.APITimeoutError, httpx2.TimeoutException,
                                  TimeoutError)) for item in chain):
            category = "timeout"
            retryable = True
        elif any(isinstance(item, ssl.SSLError) for item in chain):
            category = "tls"
            retryable = True
        elif any(isinstance(item, (openai.APIConnectionError, httpx2.TransportError,
                                  ConnectionError, OSError)) for item in chain):
            category = "connection"
            retryable = True
        elif isinstance(exc, openai.APIStatusError):
            category = "http_status"
            status_code = exc.status_code
            retryable = status_code in {408, 409, 429} or status_code >= 500
            if status_code in {401, 403}:
                category = "authentication"
            elif status_code in {400, 404, 422}:
                category = "configuration"
        elif isinstance(exc, openai.APIResponseValidationError):
            category = "invalid_response"
        elif isinstance(exc, openai.APIError):
            category = "invalid_response"
        else:
            category = "unknown"
        detail = (
            "provider returned an HTTP error"
            if isinstance(exc, openai.APIStatusError)
            else _safe_error_detail(exc)
        )
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        return ProviderRequestError(
            f"{error_prefix}: {detail}{suffix}",
            category=category,
            attempts=1,
            retry_exhausted=False,
            retryable=retryable,
            status_code=status_code,
        )


def encode_provider_request(
    settings: ProviderSettings,
    request: ModelRequest,
) -> dict[str, Any]:
    """Build the logical SDK payload for caller-owned audit capture."""

    if settings.api_protocol == "responses":
        return encode_responses_request(settings, request)
    return encode_request(settings, request)


def _sdk_arguments(payload: dict[str, Any], known: set[str]) -> dict[str, Any]:
    direct: dict[str, Any] = {}
    extensions: dict[str, Any] = {}
    for key, value in deepcopy(payload).items():
        (direct if key in known else extensions)[key] = value
    if extensions:
        direct["extra_body"] = extensions
    return direct


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        result = deepcopy(value)
    else:
        dumper = getattr(value, "model_dump", None)
        if not callable(dumper):
            raise TypeError("OpenAI SDK response does not support model_dump")
        result = dumper(mode="json")
    if not isinstance(result, dict):
        raise TypeError("OpenAI SDK response must serialize to a JSON object")
    return result


def _request_id(value: Any) -> str | None:
    request_id = getattr(value, "_request_id", None)
    return request_id if isinstance(request_id, str) and request_id else None


def _header_request_id(value: Any) -> str | None:
    headers = getattr(value, "headers", None)
    if headers is None:
        return None
    request_id = headers.get("x-request-id") or headers.get("X-Request-ID")
    return request_id if isinstance(request_id, str) and request_id else None


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    pending = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(item for item in (current.__cause__, current.__context__)
                       if item is not None)


def _safe_error_detail(exc: BaseException) -> str:
    detail = " ".join(str(exc).split())
    return detail[:500] if detail else type(exc).__name__
