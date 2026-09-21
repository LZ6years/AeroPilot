"""Byte-oriented HTTP transport with bounded retries and safe attempt metadata.

The caller supplies headers and an already serialized body. No message roles,
model response decoding, conversation state or business contracts belong here.
"""

from __future__ import annotations

import http.client
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from src.llm.contracts import ProviderRequestError


TRANSIENT_HTTP_STATUSES = frozenset({429, 502, 503, 504})
RETRY_DELAYS_SEC = (2.0, 4.0, 8.0, 16.0)
MAX_RETRY_AFTER_SEC = 60.0


def request_bytes(
    url: str, *, headers: dict[str, str], body: bytes, timeout_sec: float,
    timeout_retries: int, connection_retries: int, error_prefix: str,
    transport_trace: dict[str, Any] | None = None,
) -> bytes:
    """POST JSON with bounded retries for transient provider failures.

    ``Retry-After`` takes precedence when supplied as seconds or an HTTP date.
    A defensive 60-second cap prevents a malformed provider header from
    blocking an agent indefinitely. Connection and timeout budgets are
    configured independently because a read timeout already consumes the full
    request timeout. Across mixed failures, no request exceeds the existing
    four-retry transport ceiling.

    ``transport_trace`` is updated in place with safe attempt metadata. It
    intentionally contains no URL, headers, request body, response body, image
    bytes, or model reasoning.
    """

    encoded = body
    trace = transport_trace if transport_trace is not None else {}
    trace.clear()
    trace.update(
        {
            "attempt_count": 0,
            "retry_count": 0,
            "final_status": "pending",
            "retries": [],
        }
    )
    retry_counts = {"capacity": 0, "timeout": 0, "connection": 0}
    total_retries = 0
    attempt = 0
    while True:
        attempt += 1
        trace["attempt_count"] = attempt
        request = urllib.request.Request(
            url,
            data=encoded,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_sec
            ) as response:
                value = response.read()
                trace["final_status"] = "success"
                return value
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in TRANSIENT_HTTP_STATUSES
            if _can_retry(
                retryable=retryable,
                category_retry_count=retry_counts["capacity"],
                category_retry_limit=len(RETRY_DELAYS_SEC),
                total_retries=total_retries,
            ):
                delay = _retry_delay_seconds(
                    exc.headers.get("Retry-After") if exc.headers else None,
                    fallback=RETRY_DELAYS_SEC[total_retries],
                )
                _record_retry(
                    trace,
                    attempt=attempt,
                    category="capacity",
                    error_type=type(exc).__name__,
                    delay_sec=delay,
                    status_code=exc.code,
                )
                retry_counts["capacity"] += 1
                total_retries += 1
                time.sleep(delay)
                continue
            trace.update(
                {
                    "final_status": "failed",
                    "failure_category": "http_status",
                    "failure_error_type": type(exc).__name__,
                    "status_code": exc.code,
                }
            )
            raise ProviderRequestError(
                f"{error_prefix}: HTTP {exc.code} {exc.reason}; body={body}",
                category="http_status",
                attempts=attempt,
                retry_exhausted=retryable,
                status_code=exc.code,
            ) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            ssl.SSLError,
            EOFError,
        ) as exc:
            category, retryable = _classify_transport_error(exc)
            retry_limit = (
                timeout_retries
                if category == "timeout"
                else connection_retries
            )
            category_key = category if category in retry_counts else "connection"
            if _can_retry(
                retryable=retryable,
                category_retry_count=retry_counts[category_key],
                category_retry_limit=retry_limit,
                total_retries=total_retries,
            ):
                delay = RETRY_DELAYS_SEC[total_retries]
                _record_retry(
                    trace,
                    attempt=attempt,
                    category=category,
                    error_type=type(exc).__name__,
                    delay_sec=delay,
                )
                retry_counts[category_key] += 1
                total_retries += 1
                time.sleep(delay)
                continue
            trace.update(
                {
                    "final_status": "failed",
                    "failure_category": category,
                    "failure_error_type": type(exc).__name__,
                }
            )
            detail = _safe_transport_error_text(exc)
            raise ProviderRequestError(
                f"{error_prefix}: {detail}",
                category=category,
                attempts=attempt,
                retry_exhausted=retryable,
            ) from exc


def _can_retry(
    *,
    retryable: bool,
    category_retry_count: int,
    category_retry_limit: int,
    total_retries: int,
) -> bool:
    """Apply both the category budget and the global transport ceiling."""

    return (
        retryable
        and category_retry_count < category_retry_limit
        and total_retries < len(RETRY_DELAYS_SEC)
    )


def _record_retry(
    trace: dict[str, Any],
    *,
    attempt: int,
    category: str,
    error_type: str,
    delay_sec: float,
    status_code: int | None = None,
) -> None:
    """Append audit-safe retry metadata without copying provider payloads."""

    event: dict[str, Any] = {
        "attempt": attempt,
        "category": category,
        "error_type": error_type,
        "delay_sec": delay_sec,
    }
    if status_code is not None:
        event["status_code"] = status_code
    trace["retries"].append(event)
    trace["retry_count"] = len(trace["retries"])


def _classify_transport_error(exc: BaseException) -> tuple[str, bool]:
    """Return a stable retry category for supported transport exceptions."""

    chain = list(_exception_chain(exc))
    if any(isinstance(item, ssl.SSLCertVerificationError) for item in chain):
        return "tls", False
    if any(isinstance(item, TimeoutError) for item in chain):
        return "timeout", True
    if any(
        isinstance(
            item,
            (
                ConnectionError,
                http.client.IncompleteRead,
                http.client.BadStatusLine,
                ssl.SSLEOFError,
                EOFError,
            ),
        )
        for item in chain
    ):
        return "connection", True
    if any(isinstance(item, ssl.SSLError) for item in chain):
        return "tls", False
    # urllib wraps DNS and several socket failures in URLError. Certificate and
    # timeout causes were handled above; the remaining forms are bounded by the
    # connection retry budget.
    if isinstance(exc, urllib.error.URLError):
        return "connection", True
    return "connection", False


def _exception_chain(exc: BaseException):
    """Yield an exception and nested URLError reasons without looping."""

    seen: set[int] = set()
    current: BaseException | object | None = exc
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        yield current
        reason = getattr(current, "reason", None)
        current = reason if isinstance(reason, BaseException) else current.__cause__


def _safe_transport_error_text(exc: BaseException) -> str:
    """Keep the useful provider failure text bounded for logs and reports."""

    text = str(exc).strip() or type(exc).__name__
    return text[:1000]


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
