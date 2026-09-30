"""Task-local lifecycle for one non-blocking history-compaction request.

The daemon worker receives only an immutable candidate and stateless compactor.
It never reads Transcript/Runtime state and never commits a capsule. Polling,
validation, stale-result rejection, and state mutation remain on the planner
thread in ``ContextWindowManager``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from src.agent.context.compression.contracts import (
    CompactionCallResult,
    CompressionCandidate,
    HistoryCompactor,
)


@dataclass(frozen=True)
class CompactionJobSnapshot:
    """A lock-consistent view of the current background job."""

    status: str
    candidate: CompressionCandidate | None = None
    result: CompactionCallResult | None = None
    errors: tuple[str, ...] = ()
    started_at: float | None = None
    finished_at: float | None = None


class HistoryCompactionCoordinator:
    """Own at most one daemon compaction job for one TaskSession."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = "idle"
        self._candidate: CompressionCandidate | None = None
        self._result: CompactionCallResult | None = None
        self._errors: tuple[str, ...] = ()
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._abandoned = False
        self._failed_digests: set[str] = set()

    def submit(
        self, candidate: CompressionCandidate, compactor: HistoryCompactor
    ) -> bool:
        """Start one daemon worker, unless this task/digest is ineligible."""

        digest = candidate.request.source_digest
        with self._lock:
            if self._abandoned or self._status == "pending":
                return False
            if digest in self._failed_digests:
                return False
            if self._candidate is not None and self._candidate.request.source_digest == digest:
                return False
            self._status = "pending"
            self._candidate = candidate
            self._result = None
            self._errors = ()
            self._started_at = time.monotonic()
            self._finished_at = None
        threading.Thread(
            target=self._run,
            args=(candidate, compactor),
            name=f"history-compaction-{digest[:8]}",
            daemon=True,
        ).start()
        return True

    def _run(
        self, candidate: CompressionCandidate, compactor: HistoryCompactor
    ) -> None:
        try:
            result = compactor.compact(candidate.request)
        except Exception as exc:  # transport/model failures are audited, not raised in Planner
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                if self._abandoned or self._candidate is not candidate:
                    return
                self._status = "failed"
                self._errors = (error,)
                self._failed_digests.add(candidate.request.source_digest)
                self._finished_at = time.monotonic()
            return
        with self._lock:
            if self._abandoned or self._candidate is not candidate:
                return
            self._status = "ready"
            self._result = result
            self._finished_at = time.monotonic()

    def poll(self) -> CompactionJobSnapshot:
        with self._lock:
            return CompactionJobSnapshot(
                status=self._status,
                candidate=self._candidate,
                result=self._result,
                errors=self._errors,
                started_at=self._started_at,
                finished_at=self._finished_at,
            )

    def reject(self, source_digest: str, errors: tuple[str, ...]) -> None:
        """Mark a model result invalid so the same digest is never retried."""

        with self._lock:
            if (
                self._candidate is None
                or self._candidate.request.source_digest != source_digest
            ):
                return
            self._status = "failed"
            self._result = None
            self._errors = tuple(errors)
            self._failed_digests.add(source_digest)
            self._finished_at = time.monotonic()

    def consume(self, source_digest: str) -> None:
        """Release an accepted or stale completed job without joining its thread."""

        with self._lock:
            if (
                self._candidate is None
                or self._candidate.request.source_digest != source_digest
                or self._status == "pending"
            ):
                return
            self._status = "idle"
            self._candidate = None
            self._result = None
            self._errors = ()
            self._started_at = None
            self._finished_at = None

    def abandon(self) -> None:
        """Prevent all late worker results from becoming observable."""

        with self._lock:
            self._abandoned = True
            self._status = "abandoned"
            self._candidate = None
            self._result = None
            self._errors = ()
            self._finished_at = time.monotonic()
