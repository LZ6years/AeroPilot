"""Create per-run directories for FlyAgent records."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class RunStore:
    def __init__(self, root: str | Path = "runs") -> None:
        # Store the root directory that contains all run records.
        self.root = Path(root)

    def create_run_dir(self, prefix: str = "run") -> Path:
        # Create a timestamped run directory for one execution.
        run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
