"""Background execution for semantic_plan_update control actions.

This module owns the asynchronous job boundary. It never dispatches UAV tools;
it only asks a semantic updater for a constrained patch that TaskRuntime may
later apply after phase and version checks.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class SemanticPlanUpdateJob:
    """One queued semantic update request."""

    run_id: str
    phase_key: str
    semantic_plan_version: int
    request_payload: dict[str, Any]
    updater_context: dict[str, Any]


@dataclass(frozen=True)
class SemanticPlanUpdateResult:
    """Completed semantic update result."""

    job: SemanticPlanUpdateJob
    status: str
    patch: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.job.run_id,
            "phase_key": self.job.phase_key,
            "semantic_plan_version": self.job.semantic_plan_version,
            "status": self.status,
            "patch": dict(self.patch),
            "error": self.error,
        }


class SemanticPlanUpdateManager:
    """Single-worker manager for semantic plan update jobs."""

    def __init__(self, updater: Any | None = None) -> None:
        # Runtime composition must inject the updater.  Keeping None is useful
        # for disabled control planes and avoids a hidden cwd-based .env read.
        self.updater = updater
        self._queue: queue.Queue[SemanticPlanUpdateJob | None] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._pending_job: SemanticPlanUpdateJob | None = None
        self._completed_result: SemanticPlanUpdateResult | None = None

    def model_status(self) -> dict[str, str]:
        with self._lock:
            if self._pending_job is not None:
                return {"semantic_plan_update": "pending"}
        return {}

    def submit(self, job: SemanticPlanUpdateJob) -> tuple[bool, str]:
        if self.updater is None:
            return False, "semantic_plan_updater_unconfigured"
        with self._lock:
            if self._pending_job is not None:
                return False, "semantic_plan_update_already_pending"
            self._pending_job = job
            self._completed_result = None
            self._ensure_worker_locked()
            self._queue.put(job)
        return True, "submitted"

    def take_completed(self) -> SemanticPlanUpdateResult | None:
        with self._lock:
            result = self._completed_result
            if result is None:
                return None
            self._completed_result = None
            self._pending_job = None
            return result

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="semantic-plan-update-worker", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                patch = self.updater.update(dict(job.updater_context))
                result = SemanticPlanUpdateResult(job=job, status="completed", patch=dict(patch))
            except Exception as exc:
                result = SemanticPlanUpdateResult(job=job, status="failed", error=f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._completed_result = result
            self._queue.task_done()
