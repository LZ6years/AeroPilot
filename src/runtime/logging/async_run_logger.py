"""Asynchronously write run logs and local debug artifacts."""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class AsyncRunLogger:
    """Small async file logger for one FlyAgent run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        # Prepare output paths and start one writer thread for this run.
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.debug_events_path = self.run_dir / "debug" / "debug_events.jsonl"
        self.system_log_path = self.run_dir / "debug" / "system.log"
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._closed = False
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="flyagent-run-logger", daemon=True)
        self._thread.start()

    def info(self, message: str, **fields: Any) -> None:
        # Queue a human-readable info line for system.log.
        self._log("INFO", message, fields)

    def error(self, message: str, **fields: Any) -> None:
        # Queue a human-readable error line for system.log.
        self._log("ERROR", message, fields)

    def event(self, event: str, payload: dict[str, Any]) -> None:
        # Queue a user-facing structured event for events.jsonl.
        record = {
            "timestamp": self._timestamp(),
            "event": event,
            "payload": payload,
        }
        self._enqueue(lambda: self._append_jsonl(self.events_path, record))

    def debug_event(self, event: str, payload: dict[str, Any]) -> None:
        # Queue a detailed developer-facing event for debug/debug_events.jsonl.
        record = {
            "timestamp": self._timestamp(),
            "event": event,
            "payload": payload,
        }
        self._enqueue(lambda: self._append_jsonl(self.debug_events_path, record))

    def write_json(self, name: str, data: dict[str, Any]) -> Path:
        # Queue a JSON artifact write under the run directory.
        path = self._resolve(name)
        self._enqueue(lambda: self._write_text(path, json.dumps(data, indent=2, ensure_ascii=False)))
        return path

    def save_bytes(self, relative_path: str | Path, data: bytes) -> Path:
        # Queue a binary artifact write under the run directory.
        path = self._resolve(relative_path)
        self._enqueue(lambda: self._write_bytes(path, data))
        return path

    def save_text(self, relative_path: str | Path, text: str) -> Path:
        # Queue a text artifact write under the run directory.
        path = self._resolve(relative_path)
        self._enqueue(lambda: self._write_text(path, text))
        return path

    def close(self) -> None:
        # Flush all queued writes and stop the background writer.
        # Checking ``_closed`` and inserting the sentinel must be atomic with
        # enqueue. Otherwise a producer can pass the check, lose the race to
        # close(), and append a write *after* the sentinel; the worker then
        # exits and a completed run is missing audit artifacts.
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("AsyncRunLogger did not flush within 30 seconds")

    def _log(self, level: str, message: str, fields: dict[str, Any]) -> None:
        # Format one system log line with optional JSON fields.
        record = {
            "timestamp": self._timestamp(),
            "level": level,
            "message": message,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        self._enqueue(lambda: self._append_text(self.system_log_path, line + "\n"))

    def _enqueue(self, job: Callable[[], None]) -> None:
        # Add a write job unless the logger has already been closed.
        with self._state_lock:
            if self._closed:
                raise RuntimeError("AsyncRunLogger is already closed")
            self._queue.put(job)

    def _worker(self) -> None:
        # Drain queued write jobs until the sentinel is received.
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                job()
            except Exception as exc:
                self._write_worker_error(exc)
            finally:
                self._queue.task_done()

    def _write_worker_error(self, exc: Exception) -> None:
        # Record worker failures synchronously so artifact errors are not silent.
        payload = {
            "timestamp": self._timestamp(),
            "event": "artifact_failed",
            "payload": {"error_type": type(exc).__name__, "error": str(exc)},
        }
        self._append_jsonl(self.debug_events_path, payload)
        self._append_text(
            self.system_log_path,
            json.dumps(
                {
                    "timestamp": self._timestamp(),
                    "level": "ERROR",
                    "message": "async write failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
            + "\n",
        )

    def _resolve(self, relative_path: str | Path) -> Path:
        # Resolve a path below the run directory.
        path = self.run_dir / relative_path
        return path

    @staticmethod
    def _timestamp() -> str:
        # Return a UTC ISO timestamp for log correlation.
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        # Append one JSON object to a JSONL file.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _append_text(path: Path, text: str) -> None:
        # Append text to a UTF-8 file.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        # Write a complete UTF-8 text file.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        # Write a complete binary file.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
