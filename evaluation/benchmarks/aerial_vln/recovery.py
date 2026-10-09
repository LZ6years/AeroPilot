"""Durable episode recovery, independent of navigation scores and transport."""
from __future__ import annotations

import json
from pathlib import Path
import time

from evaluation.benchmarks.core.report_io import write_json


class RecoveryQueue:
    """Persist before dispatch; retain one first valid score per split/episode."""

    def __init__(self, path, tasks, fingerprint, *, sleep=time.sleep):
        self.path = Path(path)
        self.sleep = sleep
        if self.path.is_file():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state["fingerprint"] != fingerprint:
                raise ValueError("Campaign source/config fingerprint changed")
        else:
            self.state = {"fingerprint": fingerprint, "status": "prepared", "reason": None,
                          "consecutive_provider_failures": 0,
                          "tasks": {self.key(task): {"task": task, "status": "pending", "attempts": [],
                                                    "result": None} for task in tasks}}
            self.save()

    @staticmethod
    def key(task):
        return f'{task["split"]}/{task["episode_id"]}'

    def save(self):
        write_json(self.path, self.state)

    def pause(self, reason):
        self.state.update(status="paused", reason=reason)
        self.save()
        return False

    def recover_interrupted(self, cleanup):
        """An abandoned attempt cannot be redispatched until cleanup is proven."""
        for item in self.state["tasks"].values():
            if item["status"] != "running":
                continue
            attempt = item["attempts"][-1]
            if not cleanup(item["task"], attempt):
                return self.pause("interrupted_cleanup_unverified")
            attempt.update(status="interrupted", cleanup_verified=True)
            item["status"] = "retry_pending" if len(item["attempts"]) < 3 else "retry_exhausted"
            self.save()
        return True

    def run(self, tasks, execute, probe, *, checkpoint=lambda: True, stop=lambda: False,
            after_episode=lambda: None):
        """Run untouched tasks before deferred retries; provider outages pause safely."""
        keys = [self.key(task) for task in tasks]
        self.state.update(status="running", reason=None)
        self.save()
        completed_since_check = 0
        while True:
            candidates = [key for key in keys if self.state["tasks"][key]["status"] == "pending"]
            if not candidates:
                candidates = [key for key in keys if self.state["tasks"][key]["status"] == "retry_pending"]
            if not candidates:
                if any(self.state["tasks"][key]["status"] != "scored" for key in keys):
                    return self.pause("episode_attempts_exhausted")
                self.state.update(status="stage_complete", reason=None)
                self.save()
                return True
            if stop():
                return self.pause("manual_interrupt")
            if completed_since_check >= 25:
                if not checkpoint():
                    return self.pause("storage_or_coverage_check")
                completed_since_check = 0
            item = self.state["tasks"][candidates[0]]
            if len(item["attempts"]) >= 3:
                item["status"] = "retry_exhausted"
                self.save()
                continue
            attempt = {"number": len(item["attempts"]) + 1, "status": "running"}
            item["attempts"].append(attempt)
            item["status"] = "running"
            self.save()
            try:
                receipt = execute(item["task"], attempt)
            except KeyboardInterrupt:
                # Keep running ownership for verified cleanup on resumption.
                return self.pause("manual_interrupt")
            except Exception as exc:
                attempt["exception"] = {"type": type(exc).__name__, "message": str(exc)}
                return self.pause("supervisor_exception")
            attempt.update(receipt)
            completed_since_check += 1
            if not receipt.get("cleanup_verified"):
                item["status"] = "retry_pending"
                return self.pause("cleanup_unverified")
            result = receipt.get("result") or {}
            if receipt.get("process", {}).get("reason") == "manual_interrupt":
                item["status"] = "retry_pending" if len(item["attempts"]) < 3 else "retry_exhausted"
                return self.pause("manual_interrupt")
            infrastructure = result.get("infrastructure_error") or {}
            error = infrastructure.get("error") or result.get("error") or {}
            if infrastructure.get("stage") == "provider":
                attempt["status"] = "provider_failed"
                item["status"] = "retry_pending" if len(item["attempts"]) < 3 else "retry_exhausted"
                self.state["consecutive_provider_failures"] += 1
                self.save()
                if not error.get("retryable", False):
                    return self.pause("non_retryable_provider_error")
                if self.state["consecutive_provider_failures"] >= 3:
                    restored = False
                    for delay in (30, 120, 300):
                        self.state["service_backoff_seconds"] = delay
                        self.save()
                        try:
                            self.sleep(delay)
                        except KeyboardInterrupt:
                            return self.pause("manual_interrupt")
                        if stop():
                            return self.pause("manual_interrupt")
                        if probe():
                            restored = True
                            self.state["consecutive_provider_failures"] = 0
                            break
                    if not restored:
                        return self.pause("provider_unavailable")
                continue
            self.state["consecutive_provider_failures"] = 0
            if infrastructure or not receipt.get("valid_score"):
                attempt["status"] = "infrastructure_failed"
                item["status"] = "retry_pending" if len(item["attempts"]) < 3 else "retry_exhausted"
                return self.pause("infrastructure_failure")
            attempt["status"] = "scored"
            # A navigation failure is a valid measurement, never a retry trigger.
            if item["result"] is None:
                item["result"] = result
            item["status"] = "scored"
            self.save()
            after_episode()
