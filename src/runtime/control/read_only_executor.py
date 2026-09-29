"""Bounded scheduling of prevalidated read-only jobs, with no mission mutation.

Resource slots are reserved by the calling thread before submission. Blocked
jobs stay outside the pool, so a capability backlog cannot starve knowledge.
Results are returned in declaration order; callers own persistence and policy.
"""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
import threading
import time
from typing import Callable

from src.env_client import ActionResult
from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
from src.tools.schemas import ReadExecutionHints


@dataclass(frozen=True)
class ReadJob:
    call_id: str
    tool_name: str
    hints: ReadExecutionHints
    invoke: Callable[[], ActionResult]


class ReadOnlyBatchExecutor:
    """One bounded pool per concurrent batch; one/synchronous calls need none."""

    def __init__(self, config=None, event_sink=None):
        self.config = config or ReadOnlyExecutionConfig()
        self.event_sink = event_sink or (lambda event, payload: None)

    def _event(self, name, job=None, **values):
        self.event_sink(name, {"monotonic_sec": time.perf_counter(),
                              "thread_id": threading.get_ident(),
                              **({"tool_call_id": job.call_id, "tool": job.tool_name} if job else {}), **values})

    def _invoke(self, job):
        self._event("read_call_started", job)
        try:
            result = job.invoke()
        except Exception as exc:
            result = ActionResult(ok=False, action=job.tool_name, message=str(exc),
                                  data={"error_type": type(exc).__name__})
        except BaseException as exc:
            self._event("read_call_failed", job, error_type=type(exc).__name__, fatal=True)
            raise
        self._event("read_call_completed" if result.ok else "read_call_failed", job)
        return result

    def execute(self, jobs, *, on_completed=None):
        """Deliver each result on the caller thread, then return declaration order.

        A callback failure stops submission and disables further callbacks: the
        persistence consumer is no longer trustworthy. Tool failures, in contrast,
        still allow successful in-flight work to be delivered during shutdown.
        """
        jobs = list(jobs)
        self._event("read_batch_started", count=len(jobs), config=self.config.to_dict())
        pending = list(range(len(jobs)))
        results = [None] * len(jobs)
        running, held_keys, capability_counts = {}, set(), Counter()
        pool = None
        callback_failed = False
        primary_error = None

        def deliver(index, result):
            nonlocal callback_failed
            results[index] = result
            if on_completed is not None and not callback_failed:
                try:
                    on_completed(index, result)
                except BaseException:
                    callback_failed = True
                    raise

        def collect(completed):
            # Harvest the whole ready set even if one Future raises. Sorting is
            # deterministic for ties, not an assertion about completion time.
            nonlocal primary_error
            for future in sorted(completed, key=lambda value: running[value]):
                index = running.pop(future)
                hint = jobs[index].hints
                if hint.serial_key:
                    held_keys.remove(hint.serial_key)
                if hint.category == "capability":
                    capability_counts[hint.capability_key] -= 1
                if future.cancelled():
                    continue
                try:
                    deliver(index, future.result())
                except BaseException as exc:
                    if primary_error is None:
                        primary_error = exc
        try:
            for job in jobs:
                self._event("read_call_queued", job)
            if len(jobs) <= 1 or self.config.max_workers == 1:
                while pending:
                    index = pending.pop(0)
                    deliver(index, self._invoke(jobs[index]))
                return results
            pool = ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="flyagent-read")
            while pending or running:
                # Preserve FIFO within a serial key even when unrelated blocked
                # jobs are bypassed. Only the scheduler mutates these counters.
                blocked_keys = set()
                for index in pending[:]:
                    if len(running) >= self.config.max_workers:
                        break
                    job = jobs[index]
                    hint = job.hints
                    blocked = bool(hint.serial_key and (hint.serial_key in held_keys or hint.serial_key in blocked_keys))
                    if hint.category == "capability":
                        blocked |= (sum(capability_counts.values()) >= self.config.max_capability_calls
                                    or capability_counts[hint.capability_key] >= self.config.capability_limit(hint.capability_key))
                    if blocked:
                        if hint.serial_key:
                            blocked_keys.add(hint.serial_key)
                        continue
                    if hint.serial_key:
                        held_keys.add(hint.serial_key)
                    if hint.category == "capability":
                        capability_counts[hint.capability_key] += 1
                    try:
                        future = pool.submit(copy_context().run, self._invoke, job)
                    except BaseException:
                        if hint.serial_key:
                            held_keys.remove(hint.serial_key)
                        if hint.category == "capability":
                            capability_counts[hint.capability_key] -= 1
                        raise
                    pending.remove(index)
                    running[future] = index
                if not running:
                    raise RuntimeError("Read scheduler has pending jobs but no runnable work")
                completed, _ = wait(running, return_when=FIRST_COMPLETED)
                collect(completed)
                if primary_error is not None:
                    raise primary_error
            return results
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            raise
        finally:
            for index in pending:
                self._event("read_call_cancelled", jobs[index], reason="batch_exit_before_submission")
            for future, index in running.items():
                if future.cancel():
                    self._event("read_call_cancelled", jobs[index], reason="batch_exit_before_start")
            if pool:
                # Running Python/native work cannot be safely killed. No scene
                # cleanup or next model turn may race unfinished read work.
                while running:
                    completed, _ = wait(running, return_when=FIRST_COMPLETED)
                    collect(completed)
                pool.shutdown(wait=True, cancel_futures=True)
            self._event("read_batch_finished", count=len(jobs), returned=sum(r is not None for r in results))
