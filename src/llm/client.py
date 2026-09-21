"""Stateless default model client. Callers own messages and persist returned audits."""
from __future__ import annotations

import json
from typing import Any

from src.llm.audit import snapshot_request
from src.llm.contracts import DEFAULT_ERROR_PREFIX, ModelRequest, ModelResponse, ProviderSettings
from src.llm.providers.openai_compatible import (
    chat_completions_url, decode_response, encode_headers, encode_request,
    public_response_snapshot,
)
from src.llm.transport import request_bytes


class DefaultModelClient:
    """One configured provider, no conversation memory or business routing."""

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings

    def request(self, request: ModelRequest, *, error_prefix: str = DEFAULT_ERROR_PREFIX,
                transport_trace: dict[str, Any] | None = None) -> ModelResponse:
        payload = encode_request(self.settings, request)
        snapshot, media = snapshot_request(payload)
        trace = transport_trace if transport_trace is not None else {}
        body = request_bytes(
            chat_completions_url(self.settings.base_url),
            headers=encode_headers(self.settings),
            body=json.dumps(payload).encode("utf-8"),
            timeout_sec=self.settings.timeout_sec,
            timeout_retries=self.settings.provider_timeout_retries,
            connection_retries=self.settings.provider_connection_retries,
            error_prefix=error_prefix, transport_trace=trace,
        )
        try:
            data = decode_response(body)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            # Decode failures are not network failures and never trigger retries.
            trace.update(final_status="failed", failure_category="invalid_response")
            raise
        return ModelResponse(data, trace, snapshot, media, public_response_snapshot(data))
