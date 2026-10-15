"""Caller-scoped transport recovery, cancellation and public progress callbacks.

No task/scene state is owned here. A bounded worker permits cancellation even
when a synchronous SDK read is blocked; late responses are never published.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import math
import queue
import threading
import time
from typing import Callable

from src.llm.contracts import ProviderRequestError


@dataclass(frozen=True)
class RecoveryPolicy:
    max_attempts: int = 6
    deadline_sec: float = 600.0
    progress: Callable[[dict], None] | None = None
    cancelled: Callable[[], bool] | None = None

    def __post_init__(self):
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not math.isfinite(self.deadline_sec) or self.deadline_sec <= 0:
            raise ValueError("deadline_sec must be positive")


CURRENT_RECOVERY = ContextVar("model_transport_recovery", default=None)


@contextmanager
def recovery_scope(policy):
    token = CURRENT_RECOVERY.set(policy)
    try:
        yield
    finally:
        CURRENT_RECOVERY.reset(token)


def service_error(message, *, status_code=None):
    """Classify explicit provider error envelopes, never model-generated text."""
    text = str(message).lower()
    category, retryable = "provider_error", False
    if status_code in (401, 403) or any(s in text for s in ("invalid api key", "unauthorized", "authentication")):
        category = "authentication"
    elif status_code in (400, 404, 422) or any(s in text for s in ("invalid parameter", "context length", "context_length", "model not found")):
        category = "configuration"
    elif status_code in (408, 409, 429) or (isinstance(status_code, int) and status_code >= 500) or any(s in text for s in (
        "无可用账号", "账号并发窗口已满", "account selection timed out", "upstream stream ended prematurely",
        "server_error", "you can retry your request",
        "rate limit", "overloaded", "temporarily unavailable", "no available account", "safe to retry")):
        category, retryable = "service_unavailable", True
    return ProviderRequestError("model provider SSE error: " + str(message), category=category,
        attempts=1, retry_exhausted=False, retryable=retryable, status_code=status_code)


def retry_after_seconds(exc):
    response = getattr(exc, "response", None)
    value = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    if not value:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class RequestGuard:
    """Keep cancellation/heartbeat outside the blocking network thread."""
    def __init__(self, policy, deadline, clock):
        self.policy, self.deadline, self.clock = policy, deadline, clock
        self.response = None
        self.abandoned = threading.Event()

    def check(self):
        reason = "cancelled" if self.policy.cancelled and self.policy.cancelled() else (
            "deadline" if self.clock() >= self.deadline else None)
        if reason:
            raise ProviderRequestError("Model request " + reason, category=reason, attempts=1,
                retry_exhausted=True, retryable=reason == "deadline", exhaustion_reason=reason)

    def emit(self, phase):
        if self.policy.progress:
            self.policy.progress({"phase": phase, "remaining_seconds": max(0, self.deadline-self.clock())})

    def bind(self, response):
        self.response = response
        if self.abandoned.is_set():
            response.close()
            raise TimeoutError("Discarded late response")

    def run(self, operation, close_client):
        result = queue.Queue(maxsize=1)
        def worker():
            try:
                result.put((True, operation()))
            except BaseException as exc:
                result.put((False, exc))
        self.check()
        threading.Thread(target=worker, name="model-http-attempt", daemon=True).start()
        next_progress = 0.0
        try:
            while True:
                self.check()
                if self.clock() >= next_progress:
                    self.emit("request_wait")
                    next_progress = self.clock() + 5
                try:
                    ok, value = result.get(timeout=min(.2, max(.001, self.deadline-self.clock())))
                except queue.Empty:
                    continue
                self.check()
                if ok:
                    return value
                raise value
        except BaseException:
            self.abandoned.set()
            def close():
                try:
                    if self.response is not None:
                        self.response.close()
                except Exception:
                    pass
                finally:
                    try:
                        close_client()
                    except Exception:
                        pass
            threading.Thread(target=close, name="model-http-close", daemon=True).start()
            raise

    def sleep(self, seconds, sleep):
        until = min(self.deadline, self.clock() + seconds)
        next_progress = 0.0
        while self.clock() < until:
            self.check()
            if self.clock() >= next_progress:
                self.emit("retry_backoff")
                next_progress = self.clock() + 5
            # The clock can advance between the loop condition, progress I/O
            # and this call. Never pass a negative remainder to time.sleep.
            remaining = until-self.clock()
            if remaining > 0:
                sleep(min(1, remaining))
        self.check()
