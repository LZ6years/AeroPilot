"""Persist content-addressed model-only media under one Runtime run."""

from __future__ import annotations

import hashlib
from pathlib import Path


class ModelMediaStore:
    """Write deterministic derived PNGs without duplicating identical bytes."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "artifacts" / "model_media"

    def write_png(self, observation_id: str, data: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(data).hexdigest()
        safe_observation = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in observation_id
        ) or "unknown"
        path = self.root / safe_observation / f"{digest}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return path.relative_to(self.run_dir).as_posix(), digest
