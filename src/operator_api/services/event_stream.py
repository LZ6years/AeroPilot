"""Nonblocking Runtime sink with durable, ordered human-only delivery records."""

from __future__ import annotations

import copy
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def public_value(value: Any) -> Any:
    """Remove bytes and credentials from UI transport and detailed audit views."""

    if isinstance(value, dict):
        private = {"api_key", "authorization", "access_token", "reasoning_content", "base64"}
        return {key: public_value(item) for key, item in value.items() if key.lower() not in private}
    if isinstance(value, (list, tuple)):
        return [public_value(item) for item in value]
    if isinstance(value, bytes) or isinstance(value, str) and value.startswith("data:image/"):
        return "[media stored separately]"
    return value


class EventStream:
    """A queue decouples Runtime from disk and slow browser subscribers.

    The extra operator log records delivery and UI lifecycle, not model history.
    Raw Runtime artifacts remain authoritative and are never rewritten here.
    """

    def __init__(self, run_dir: Path, session_id: str):
        self.path = run_dir / "operator" / "events.jsonl"
        self.session_id = session_id
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._closed = False
        self.error: str | None = None
        self._thread = threading.Thread(target=self._write, name="operator-events", daemon=True)
        self._thread.start()

    def publish(self, kind: str, payload: dict[str, Any], *, instruction_id: str | None = None) -> None:
        # Freeze at emission, so later state cannot alter a past decision/event.
        frozen = public_value(copy.deepcopy(payload))
        with self._lock:
            if self._closed:
                return
            self._queue.put({
                "schema_version": "operator_event_v1", "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self.session_id,
                "instruction_id": frozen.get("instruction_id") or instruction_id,
                "decision_id": frozen.get("decision_id"), "kind": kind, "payload": frozen,
            })

    def since(self, seq: int) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._events[max(0, seq):])

    @property
    def seq(self) -> int:
        with self._lock:
            return len(self._events)

    def _write(self) -> None:
        handle = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a", encoding="utf-8")
        except OSError as exc:
            self.error = f"operator event persistence failed: {exc}"
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                with self._lock:
                    item["seq"] = len(self._events) + 1
                try:
                    if handle is not None:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        handle.flush()
                except OSError as exc:
                    self.error = f"operator event persistence failed: {exc}"
                with self._lock:
                    self._events.append(item)
        finally:
            if handle is not None:
                handle.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Operator events did not finish flushing")
