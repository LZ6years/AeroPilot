"""Shared HTTP transport for OpenAI-compatible model requests.

The semantic parser, native ToolCall planner, and semantic-plan updater all
share the same provider failure policy. Only transient capacity statuses are
retried; contract errors and ordinary client errors remain immediate failures.
This module does not interpret model responses or know about agent state.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from src.runtime.configs.model_config import ModelConfig


TRANSIENT_HTTP_STATUSES = frozenset({429, 502, 503, 504})
RETRY_DELAYS_SEC = (2.0, 4.0, 8.0, 16.0)
MAX_RETRY_AFTER_SEC = 60.0


def request_openai_compatible_json(
    model_config: ModelConfig,
    payload: dict[str, Any],
    *,
    error_prefix: str,
) -> dict[str, Any]:
    """POST JSON and retry only transient provider capacity responses.

    ``Retry-After`` takes precedence when supplied as seconds or an HTTP date.
    A defensive 60-second cap prevents a malformed provider header from
    blocking an agent indefinitely. The final exception retains the caller's
    established error prefix so existing diagnostics remain searchable.
    """

    url = model_config.chat_completions_url()
    encoded = json.dumps(payload).encode("utf-8")
    for attempt in range(len(RETRY_DELAYS_SEC) + 1):
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=model_config.timeout_sec
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
                if not isinstance(value, dict):
                    raise TypeError(
                        "model provider response must be a JSON object"
                    )
                return value
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if (
                exc.code in TRANSIENT_HTTP_STATUSES
                and attempt < len(RETRY_DELAYS_SEC)
            ):
                delay = _retry_delay_seconds(
                    exc.headers.get("Retry-After") if exc.headers else None,
                    fallback=RETRY_DELAYS_SEC[attempt],
                )
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"{error_prefix}: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"{error_prefix}: {exc}") from exc
    raise AssertionError("transient provider retry loop exhausted unexpectedly")


def _retry_delay_seconds(value: str | None, *, fallback: float) -> float:
    """Return a bounded Retry-After delay, falling back on parse failure."""

    if value is None:
        return fallback
    text = str(value).strip()
    try:
        return min(MAX_RETRY_AFTER_SEC, max(0.0, float(text)))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        remaining = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return min(MAX_RETRY_AFTER_SEC, max(0.0, remaining))
    except (TypeError, ValueError, OverflowError):
        return fallback
