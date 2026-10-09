"""Shared request contracts; no agent, runtime or conversation state is owned here."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Protocol


DEFAULT_ERROR_PREFIX = "Model request failed"


@dataclass(frozen=True)
class ProviderSettings:
    """Transport-only configuration. Credentials must never appear in repr/audit."""
    provider: str = "openai_compatible"
    model_name: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    api_protocol: str = "chat_completions"
    stream: bool = False
    timeout_sec: float = 60.0
    provider_timeout_retries: int = 1
    provider_connection_retries: int = 2
    retry_deadline_sec: float = 600.0

    def __post_init__(self) -> None:
        if self.provider != "openai_compatible":
            raise ValueError(f"unsupported model provider: {self.provider}")
        if self.api_protocol not in ("chat_completions", "responses"):
            raise ValueError(
                "api_protocol must be chat_completions or responses"
            )
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")
        if self.api_protocol == "responses" and self.stream:
            raise ValueError("streaming Responses API is not supported")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if not math.isfinite(self.retry_deadline_sec) or self.retry_deadline_sec <= 0:
            raise ValueError("retry_deadline_sec must be finite and positive")
        for count in (self.provider_timeout_retries, self.provider_connection_retries):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("retry counts must be non-negative integers")


@dataclass(frozen=True)
class ModelRequest:
    """Caller-owned complete request; parameters never contain model/messages/tools."""
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if {"model", "messages", "tools", "stream"} & self.parameters.keys():
            raise ValueError(
                "model/messages/tools/stream are not generation parameters"
            )
        for name in ("messages", "tools", "parameters"):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))


@dataclass(frozen=True)
class ModelResponse:
    """Provider envelope plus human-only diagnostic data; no semantic interpretation."""
    data: dict[str, Any]
    transport_trace: dict[str, Any] = field(default_factory=dict)
    request_snapshot: dict[str, Any] = field(default_factory=dict)
    media: list[dict[str, Any]] = field(default_factory=list, repr=False)
    response_snapshot: dict[str, Any] = field(default_factory=dict)

    def _first_message(self) -> dict[str, Any]:
        """Read the current Chat Completions envelope without repairing it."""
        choices = self.data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return {}
        message = choices[0].get("message")
        return message if isinstance(message, dict) else {}

    @property
    def text(self) -> str | None:
        """First-choice string content, including empty text; no fallback intent.

        Null, missing or non-string content returns None. Structured content and
        other choices remain available in data; this accessor does not flatten
        them or mistake hidden reasoning for the public answer.
        """
        content = self._first_message().get("content")
        return content if isinstance(content, str) else None

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Independent copy of first-choice wire calls; arguments stay unparsed."""
        calls = self._first_message().get("tool_calls")
        return deepcopy(calls) if isinstance(calls, list) else []

    @property
    def usage(self) -> dict[str, Any] | None:
        """Independent provider usage dictionary, or None when unavailable."""
        value = self.data.get("usage")
        return deepcopy(value) if isinstance(value, dict) else None


class ModelClient(Protocol):
    """Injection seam used equally by production callers and deterministic fakes."""
    def request(self, request: ModelRequest, *, error_prefix: str = DEFAULT_ERROR_PREFIX,
                transport_trace: dict[str, Any] | None = None) -> ModelResponse: ...


class ProviderRequestError(RuntimeError):
    """A provider transport or HTTP failure with safe classification data."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        attempts: int,
        retry_exhausted: bool,
        status_code: int | None = None,
        retryable: bool = False,
        exhaustion_reason: str | None = None,
        elapsed_seconds: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.attempts = attempts
        self.retry_exhausted = retry_exhausted
        self.status_code = status_code
        self.retryable = retryable
        self.exhaustion_reason = exhaustion_reason
        self.elapsed_seconds = elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        """Keep transport facts across caller-owned error/report boundaries."""
        return {name: getattr(self, name) for name in (
            "category", "attempts", "retry_exhausted", "retryable",
            "status_code", "exhaustion_reason", "elapsed_seconds",
        )}
