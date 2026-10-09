"""Thread-safe operator requests acknowledged only at complete Runtime boundaries.

The web worker owns the instruction. HTTP/WebSocket threads may only signal this
object; they never execute a flight primitive or mutate MissionState themselves.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Literal


class RuntimeControl:
    """A task-local pause/stop latch; stopping cannot be undone by resume."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._desired = "running"
        self._status = "running"
        self._reason = "operator"
        self._finished = False

    def request(
        self, action: Literal["pause", "resume", "stop"], *, reason: str = "operator"
    ) -> dict[str, Any]:
        """Signal intent without waiting for the running SDK/provider request."""

        if action not in {"pause", "resume", "stop"}:
            raise ValueError(f"Unsupported operator control: {action}")
        with self._condition:
            if self._finished:
                return self._snapshot()
            if self._desired == "stopped" and action != "stop":
                raise ValueError("A stopped instruction cannot be resumed")
            desired = {"pause": "paused", "resume": "running", "stop": "stopped"}[action]
            if desired != self._desired:
                self._desired = desired
                self._status = {"pause": "pause_requested", "resume": "resume_requested", "stop": "stop_requested"}[action]
                self._reason = reason
            self._condition.notify_all()
            return self._snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        return {"status": self._status, "desired": self._desired, "reason": self._reason}

    def checkpoint(
        self, boundary: str, on_changed: Callable[[dict[str, Any]], None]
    ) -> bool:
        """Block while paused; return false on stop, before any next tool runs.

        Notify outside the condition lock so a UI sink cannot deadlock control.
        A model result already received before this boundary remains pending
        until resume; stop discards it in the owning loop.
        """

        while True:
            with self._condition:
                target = self._desired
                changed = self._status != target
                self._status = target
                payload = {**self._snapshot(), "boundary": boundary}
            if changed:
                on_changed(payload)
            if target == "stopped":
                return False
            if target == "running":
                return True
            with self._condition:
                if self._desired == "paused":
                    self._condition.wait(timeout=0.25)

    def finish(self) -> None:
        """Release waiters and prevent requests from affecting a later task."""

        with self._condition:
            self._finished = True
            self._desired = "stopped"
            self._status = "finished"
            self._condition.notify_all()
