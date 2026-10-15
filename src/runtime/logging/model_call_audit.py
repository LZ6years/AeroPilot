"""Persist background model-call metadata independently of the async logger.

A task can finish while a compaction request is still in flight. Immutable
started/finished artifacts remain writable after AsyncRunLogger.close(), so
late or abandoned work is not omitted from experiment cost accounting.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelCallAuditWriter:
    """Caller-owned atomic sink for one run's background call ledger."""

    def __init__(self, run_dir: str | Path) -> None:
        self.directory = Path(run_dir) / "debug" / "history_compaction_calls"

    def __call__(self, record: dict[str, Any]) -> None:
        call_id = str(record["call_id"])
        if not call_id.isalnum():
            raise ValueError("model audit call_id must be alphanumeric")
        stage = "started" if record["status"] == "started" else "finished"
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{call_id}.{stage}.json"
        payload = {**record, "recorded_at": datetime.now(timezone.utc).isoformat()}
        descriptor, temporary_name = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
