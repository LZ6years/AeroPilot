"""Write run records and event logs for FlyAgent executions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RecordWriter:
    def __init__(self, run_dir: str | Path) -> None:
        # Prepare the run directory and event log path.
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.debug_events_path = self.run_dir / "debug" / "debug_events.jsonl"

    def write_json(self, name: str, data: dict[str, Any]) -> Path:
        # Persist a structured JSON artifact inside the run directory.
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_bytes(self, relative_path: str | Path, data: bytes) -> Path:
        """Persist one binary debug asset inside the run directory."""

        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def event(self, event: str, payload: dict[str, Any]) -> None:
        # Append a timestamped user-facing event to the JSONL event stream.
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def debug_event(self, event: str, payload: dict[str, Any]) -> None:
        # Append a detailed developer-facing event to debug/debug_events.jsonl.
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        self.debug_events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.debug_events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
