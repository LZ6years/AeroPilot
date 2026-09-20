"""Human-only lifecycle evidence and aggregate errors for scene orchestration."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class SceneLifecycleError(RuntimeError):
    """Keep the initiating error separate from independent rollback failures."""

    def __init__(self, original_error: Exception, cleanup_errors: list[dict]):
        self.original_error = original_error
        self.cleanup_errors = cleanup_errors
        super().__init__(f"{original_error}; cleanup_errors={cleanup_errors}")


class SceneLifecycle:
    """Append-only scene diagnostics; elapsed time never depends on wall time."""

    def __init__(self, directory: Path):
        self.path = directory / "lifecycle.jsonl"
        self.started = time.monotonic()

    def record(self, stage: str, status: str = "ok", **details) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": time.monotonic() - self.started,
                "stage": stage, "status": status, **details,
            }, ensure_ascii=False) + "\n")
