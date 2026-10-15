"""Durable serial and parallel recovery shared by all benchmark campaigns.

The controller is the only state writer. Dataset runners return original
receipts; this module interprets them without changing navigation or metrics.
"""
from __future__ import annotations
import json
from pathlib import Path
import threading
import time
from evaluation.benchmarks.core.report_io import write_json
from evaluation.benchmarks.core.campaign.contracts import TaskIdentity

TRANSIENT_TRANSPORT_ERRORS = frozenset({
    "TimeoutError", "TimeoutExpired", "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "ConnectionRefusedError", "BrokenPipeError", "SSLEOFError", "SshOperationError",
})
TRANSPORT_FAILURE_STAGES = frozenset({"scene_setup", "case_setup", "scene_control", "cleanup", "case_execution"})


def scene_control_timeout(error):
    """Recognize the frozen adapter's RPC timeout after its cause was wrapped.

    SceneRecoveryRequired also covers unknown/unsafe control state, so the
    exception type alone never permits recovery. This exact adapter-generated
    diagnostic identifies msgpack RPC's timeout; cleanup remains mandatory at
    the receipt boundary. No action is replayed in the poisoned scene.
    """
    return (error.get("error_type") == "SceneRecoveryRequired"
            and error.get("error") == "AirSim freeze failed: Request timed out")



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
        return TaskIdentity.from_task(task).key

    @staticmethod
    def attempt_limit(item):
        """Default to three attempts; explicit operator grants are task-local."""
        limit = item.get("attempt_limit", 3)
        if type(limit) is not int or limit < 3:
            raise ValueError("Invalid task attempt limit")
        return limit

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
            item["status"] = "retry_pending" if len(item["attempts"]) < self.attempt_limit(item) else "retry_exhausted"
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
            if len(item["attempts"]) >= self.attempt_limit(item):
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
                item["status"] = "retry_pending" if len(item["attempts"]) < self.attempt_limit(item) else "retry_exhausted"
                return self.pause("manual_interrupt")
            infrastructure = result.get("infrastructure_error") or {}
            error = infrastructure.get("error") or result.get("error") or {}
            if infrastructure.get("stage") == "provider":
                attempt["status"] = "provider_failed"
                item["status"] = "retry_pending" if len(item["attempts"]) < self.attempt_limit(item) else "retry_exhausted"
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
                item["status"] = "retry_pending" if len(item["attempts"]) < self.attempt_limit(item) else "retry_exhausted"
                return self.pause("infrastructure_failure")
            attempt["status"] = "scored"
            # A navigation failure is a valid measurement, never a retry trigger.
            if item["result"] is None:
                item["result"] = result
            item["status"] = "scored"
            self.save()
            after_episode()

class ParallelRecoveryQueue(RecoveryQueue):
    """Extend the existing progress format with lane/attempt ownership.

    Tasks and fingerprint remain the frozen campaign's identities.
    Only the constructing thread may mutate or save progress. Optional lane
    metadata is additive, so the original serial/report readers still work.
    """

    def __init__(self, *args, save_retry=None, terminal_provider_errors=False, **kwargs):
        # The external controller supplies its bounded file-access policy while
        # the frozen queue retains ownership of the schema and atomic writer.
        self._save_retry = save_retry
        # Fixed-cohort evaluations may retain provider failures as unscored
        # terminal outcomes. Legacy campaigns keep their service recovery.
        self.terminal_provider_errors = terminal_provider_errors
        self._writer = threading.get_ident()
        super().__init__(*args, **kwargs)

    def terminal(self, item):
        return item["status"] == "scored" or (self.terminal_provider_errors and item["status"] == "model_error")

    def authorize_extra_attempt(self, task, *, reason, authorized_at):
        """Grant one exhausted task another attempt under an operator-held lock.

        The caller must verify that controller and host have stopped. Preserve
        every prior attempt and its number so evidence and accounting stay intact.
        """
        if threading.get_ident() != self._writer:
            raise RuntimeError("Only the controller may change retry grants")
        item = self.state["tasks"][self.key(task)]
        if self.state.get("status") != "paused" or any(x["status"] == "running" for x in self.state["tasks"].values()):
            raise ValueError("Retry grants require a stopped, paused queue")
        if item["status"] != "retry_exhausted" or item.get("result") is not None:
            raise ValueError("Only an exhausted unscored task can receive a retry")
        if not item["attempts"] or not all(a.get("cleanup_verified") and not a.get("valid_score") for a in item["attempts"]):
            raise ValueError("Every previous attempt must be invalid and cleaned")
        if not reason or not authorized_at:
            raise ValueError("Retry authorization requires an audit reason and time")
        old_limit = self.attempt_limit(item)
        item["attempt_limit"] = max(old_limit, len(item["attempts"])) + 1
        item.setdefault("retry_authorizations", []).append({"previous_limit": old_limit,
            "attempt_limit": item["attempt_limit"], "reason": reason, "authorized_at": authorized_at})
        item["status"] = "retry_pending"
        self.state.update(status="paused", reason="manual_interrupt")
        self.save()

    def reconcile_scored_errors(self):
        """Operator-only migration after draining, with the campaign lock held.

        Keep raw attempt results intact. Revoke only technical-error scores,
        never ordinary navigation failures; repeated reconciliation is a no-op.
        The caller backs up progress and verifies no controller/host is alive.
        """
        changes = []
        for key, item in self.state["tasks"].items():
            if item["status"] != "scored":
                continue
            scored = [a for a in item["attempts"] if a.get("status") == "scored"]
            invalid = [a for a in scored if self.infrastructure_record(a.get("result") or item.get("result") or {})]
            if not invalid:
                continue
            if any(not a.get("cleanup_verified") for a in invalid):
                raise ValueError("Cannot requeue a scored error without verified cleanup")
            for attempt in invalid:
                reason = self.infrastructure_record(attempt.get("result") or item.get("result") or {})
                changes.append({"task_key": key, "attempt": attempt["number"], "previous_status": "scored", "reason": reason})
                attempt.update(previous_status="scored", valid_score=False,
                               status="provider_failed" if reason.get("stage") == "provider" else "infrastructure_failed")
            retained = next((a for a in scored if a not in invalid), None)
            item["result"] = retained["result"] if retained else None
            item["status"] = "scored" if retained else ("retry_pending" if len(item["attempts"]) < self.attempt_limit(item) else "retry_exhausted")
        if changes:
            self.save()
        return changes

    def save(self):
        if threading.get_ident() != self._writer:
            raise RuntimeError("Only the controller may write campaign progress")
        if self._save_retry is None:
            super().save()
        else:
            self._save_retry(super().save)

    @staticmethod
    def infrastructure_record(result):
        """Read both staged and historical flat infrastructure receipts.

        TravelUAV's frozen runner can expose the exception itself rather than a
        stage/error wrapper. Its precise stage is unavailable, so use the coarse
        case_execution view. Classification still requires a known transport
        type and independent cleanup proof. Preserve raw result dictionaries.
        """
        if not isinstance(result, dict):
            return {}
        value = result.get("infrastructure_error")
        top_error = result.get("error")
        if not value and isinstance(top_error, dict) and top_error.get("error_type") == "ValueError":
            # Only the old wire decoder's explicit error-envelope prefix is
            # evidence. Arbitrary ValueError/navigation text is not a service error.
            message = str(top_error.get("error", ""))
            if message == "sleep length must be non-negative":
                # Explicit Python timing failure, never a navigation outcome.
                # Retain the raw ValueError and classify only this known error.
                return {"stage": "case_execution", "error": {**top_error,
                        "category": "runtime_backoff_error", "retryable": True}}
            if message.startswith("model provider SSE error:"):
                from src.llm.recovery import service_error
                classified = service_error(message.split(":", 1)[1].strip()).to_dict()
                return {"stage": "provider", "error": {**top_error, **classified,
                        "error_type": "ProviderRequestError", "legacy_error_type": "ValueError"}}
        if not value and isinstance(top_error, dict) and top_error.get("error_type") == "ProviderRequestError":
            # Some frozen runners retained this provider exhaustion only in
            # outcome.error after acquiring scene capabilities. It remains a
            # technical attempt, not a navigation failure with success=0.
            return {"stage": "provider", "error": top_error}
        if not isinstance(value, dict):
            return {}
        if "stage" in value or not isinstance(value.get("error_type"), str):
            return value
        kind = value["error_type"]
        if kind == "CleanupError":
            return {"stage": "cleanup", "errors": value.get("details", [])}
        stage = {"ProviderRequestError": "provider", "SceneRecoveryRequired": "scene_control"}.get(kind, "case_execution")
        return {"stage": stage, "error": value}

    @classmethod
    def provider_error(cls, result):
        """Return provider-only structured metadata; SSH text is never HTTP data."""
        infrastructure = cls.infrastructure_record(result)
        error = infrastructure.get("error")
        return error if infrastructure.get("stage") == "provider" and isinstance(error, dict) else {}

    @classmethod
    def infrastructure_errors(cls, result):
        """Normalize one error and cleanup's error list without hiding members.

        Mixed, empty or malformed error groups remain unclassified. Some older
        receipts store only the top-level error, so retain that read boundary.
        No stored result or error dictionary is modified by classification.
        """
        if not isinstance(result, dict):
            return ()
        infrastructure = cls.infrastructure_record(result)
        errors = []
        for key in ("error", "errors"):
            value = infrastructure.get(key)
            if value is None:
                continue
            values = value if key == "errors" else [value]
            if not isinstance(values, list) or any(not isinstance(error, dict) or not error for error in values):
                return ()
            errors.extend(values)
        if not errors:
            error = result.get("error")
            if isinstance(error, dict) and error:
                errors.append(error)
        return tuple(errors)

    @classmethod
    def retryable_infrastructure(cls, receipt):
        """Allow deferred transport recovery only after verified scene cleanup."""
        if receipt.get("cleanup_verified") is not True:
            return False
        result = receipt.get("result") or {}
        if not isinstance(result, dict):
            return False
        infrastructure = cls.infrastructure_record(result)
        if not isinstance(infrastructure, dict) or infrastructure.get("stage") not in TRANSPORT_FAILURE_STAGES:
            return False
        errors = cls.infrastructure_errors(result)
        return bool(errors) and all(
            (error.get("error_type") in TRANSIENT_TRANSPORT_ERRORS or
             scene_control_timeout(error) or
             (error.get("error_type") == "ValueError" and error.get("category") == "runtime_backoff_error"
              and error.get("error") == "sleep length must be non-negative"))
            and error.get("retryable") is not False
            for error in errors)

    @classmethod
    def recoverable_pause(cls, state):
        """Reclassify a generic pause from its matching current attempt receipt.

        A context pointer is not itself permission: validate task and attempt
        ownership, then use the same classifier as normal queue acceptance.
        Legacy progress without a cause remains readable and requires an
        explicit controller resume after inspecting its historical evidence.
        """
        if state.get("status") != "paused" or state.get("reason") != "infrastructure_failure":
            return False
        cause = state.get("pause_cause") or {}
        if not isinstance(cause, dict) or cause.get("reason") != state["reason"]:
            return False
        if not isinstance(cause.get("task_key"), str) or type(cause.get("attempt_number")) is not int:
            return False
        item = state.get("tasks", {}).get(cause.get("task_key"))
        if not item or item.get("status") == "scored" or not item.get("attempts"):
            return False
        attempt = item["attempts"][-1]
        if attempt.get("number") != cause.get("attempt_number") or attempt.get("status") != "infrastructure_failed":
            return False
        return cls.retryable_infrastructure(attempt)

    def pause(self, reason, *, task=None, number=None):
        """Persist the causal lease while preserving it through final cleanup."""
        if threading.get_ident() != self._writer:
            raise RuntimeError("Only the controller may pause campaign progress")
        if task is not None:
            self.state["pause_cause"] = {"task_key": self.key(task), "attempt_number": number, "reason": reason}
        elif reason != self.state.get("reason"):
            self.state.pop("pause_cause", None)
        return super().pause(reason)

    def claim(self, tasks, lane, *, metadata=None):
        """Persist an exclusive lease; untouched tasks precede deferred retries."""
        if threading.get_ident() != self._writer:
            raise RuntimeError("Only the controller may claim an episode")
        if any(item["status"] == "running" and item["attempts"][-1].get("lane") == lane
               for item in self.state["tasks"].values()):
            raise ValueError("Execution lane already has an active attempt")
        if {"number", "status", "lane"}.intersection(metadata or {}):
            raise ValueError("Metadata cannot replace lease ownership")
        keys = [self.key(task) for task in tasks]
        for status in ("pending", "retry_pending"):
            for key in keys:
                item = self.state["tasks"][key]
                if item["status"] != status:
                    continue
                if len(item["attempts"]) >= self.attempt_limit(item):
                    item["status"] = "retry_exhausted"
                    self.save()
                    continue
                attempt = {"number": len(item["attempts"]) + 1, "status": "running",
                           "lane": lane, **(metadata or {})}
                item["attempts"].append(attempt)
                item["status"] = "running"
                self.save()
                return item["task"], dict(attempt)
        return None

    def accept(self, task, number, receipt):
        """Commit one matching attempt; return the controller's next action.

        Cleanup is a prerequisite for dispatching another scene. Completed
        navigation failures are valid scores. Partial provider trajectories
        stay in attempt evidence and never become an episode result.
        """
        if threading.get_ident() != self._writer:
            raise RuntimeError("Only the controller may accept a receipt")
        item = self.state["tasks"][self.key(task)]
        if self.terminal(item):
            return "duplicate"
        if not item["attempts"] or item["attempts"][-1]["number"] != number:
            raise ValueError("Receipt does not belong to the current attempt")
        attempt = item["attempts"][-1]
        if attempt["status"] != "running":
            return "duplicate"
        result = receipt.get("result") or {}
        if result.get("episode_id") not in {None, task["episode_id"]}:
            raise ValueError("Receipt episode identity mismatch")
        attempt.update(receipt)
        deferred = "retry_pending" if len(item["attempts"]) < self.attempt_limit(item) else "retry_exhausted"
        if not receipt.get("cleanup_verified"):
            attempt["status"] = "cleanup_unverified"
            item["status"] = deferred
            self.pause("cleanup_unverified", task=task, number=number)
            return "pause"
        infrastructure = self.infrastructure_record(result)
        error = self.provider_error(result)
        if receipt.get("valid_score") and not infrastructure and result.get("metrics"):
            attempt["status"] = "scored"
            if item["result"] is None:
                item["result"] = result
            item["status"] = "scored"
            self.state["consecutive_provider_failures"] = 0
            self.save()
            return "scored"
        item["status"] = deferred
        if receipt.get("process", {}).get("reason"):
            attempt["status"] = "interrupted"
            self.save()
            return "interrupted"
        if infrastructure.get("stage") == "provider":
            if self.terminal_provider_errors:
                attempt["status"] = item["status"] = "model_error"
                item["result"] = result
                self.state["consecutive_provider_failures"] = 0
                self.save()
                return "model_error"
            attempt["status"] = "provider_failed"
            self.state["consecutive_provider_failures"] += 1
            self.save()
            if not error.get("retryable", False):
                self.pause("non_retryable_provider_error", task=task, number=number)
                return "pause"
            return "service_check" if self.state["consecutive_provider_failures"] >= 3 else "deferred"
        attempt["status"] = "infrastructure_failed"
        self.state["consecutive_provider_failures"] = 0
        # Final-observation timeout is carried in cleanup errors[], rather than
        # error. Verified rollback makes this attempt eligible for a later run;
        # it still cannot become a navigation score or interrupt the other lane.
        if self.retryable_infrastructure(receipt):
            self.save()
            return "deferred"
        self.pause("infrastructure_failure", task=task, number=number)
        return "pause"

    def recover_leases(self, recover):
        """Adopt durable completed results or defer proven-clean abandoned work."""
        verified = True
        for item in self.state["tasks"].values():
            unfinished_cleanup = item["status"] != "scored" and item["attempts"] \
                and item["attempts"][-1].get("cleanup_verified") is False
            if item["status"] != "running" and not unfinished_cleanup:
                continue
            attempt = item["attempts"][-1]
            receipt = recover(item["task"], attempt)
            if unfinished_cleanup and attempt["status"] != "running":
                attempt["previous_status"] = attempt["status"]
                attempt["status"] = "running"
            if self.accept(item["task"], attempt["number"], receipt) == "pause":
                verified = False
        if not verified:
            self.pause("interrupted_cleanup_unverified")
        return verified
