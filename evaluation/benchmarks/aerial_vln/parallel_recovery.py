"""Single-writer leases for supervised parallel AerialVLN attempts.

Execution workers never receive this queue. The controller persists a lease
before dispatch and commits a returned receipt only once. Navigation scores,
provider recovery, process supervision and scene cleanup retain their separate
ownership; this module does not launch processes or touch a simulator.
"""
from __future__ import annotations

import threading

from evaluation.benchmarks.aerial_vln.recovery import RecoveryQueue


# These are serialized transport exceptions, not navigation outcomes. Provider
# retries remain owned by src/llm and its separate service-recovery policy.
TRANSIENT_TRANSPORT_ERRORS = frozenset({
    "TimeoutError", "TimeoutExpired", "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "ConnectionRefusedError", "BrokenPipeError", "SSLEOFError",
})
TRANSPORT_FAILURE_STAGES = frozenset({"scene_setup", "case_setup", "scene_control", "cleanup"})


class ParallelRecoveryQueue(RecoveryQueue):
    """Extend the existing progress format with lane/attempt ownership.

    Tasks and fingerprint remain the frozen campaign's identities.
    Only the constructing thread may mutate or save progress. Optional lane
    metadata is additive, so the original serial/report readers still work.
    """

    def __init__(self, *args, save_retry=None, **kwargs):
        # The external controller supplies its bounded file-access policy while
        # the frozen queue retains ownership of the schema and atomic writer.
        self._save_retry = save_retry
        self._writer = threading.get_ident()
        super().__init__(*args, **kwargs)

    def save(self):
        if threading.get_ident() != self._writer:
            raise RuntimeError("Only the controller may write campaign progress")
        if self._save_retry is None:
            super().save()
        else:
            self._save_retry(super().save)

    @staticmethod
    def infrastructure_errors(result):
        """Normalize one error and cleanup's error list without hiding members.

        Mixed, empty or malformed error groups remain unclassified. Some older
        receipts store only the top-level error, so retain that read boundary.
        No stored result or error dictionary is modified by classification.
        """
        if not isinstance(result, dict):
            return ()
        infrastructure = result.get("infrastructure_error") or {}
        if not isinstance(infrastructure, dict):
            return ()
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
        infrastructure = result.get("infrastructure_error") or {}
        if not isinstance(infrastructure, dict) or infrastructure.get("stage") not in TRANSPORT_FAILURE_STAGES:
            return False
        errors = cls.infrastructure_errors(result)
        return bool(errors) and all(
            error.get("error_type") in TRANSIENT_TRANSPORT_ERRORS and error.get("retryable") is not False
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
                if len(item["attempts"]) >= 3:
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
        if item["status"] == "scored":
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
        deferred = "retry_pending" if len(item["attempts"]) < 3 else "retry_exhausted"
        if not receipt.get("cleanup_verified"):
            attempt["status"] = "cleanup_unverified"
            item["status"] = deferred
            self.pause("cleanup_unverified", task=task, number=number)
            return "pause"
        infrastructure = result.get("infrastructure_error") or {}
        error = infrastructure.get("error") or result.get("error") or {}
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
