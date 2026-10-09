"""Resolve operator service settings relative to an explicit repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperatorConfig:
    """Local browser service defaults; flight and provider configs stay upstream."""

    root: Path = Path(__file__).resolve().parents[2]
    maps_config: str = "configs/maps/airvln_maps.json"
    env_path: str = ".env"
    run_root: str = "runs"
    default_map_id: str = "cityenviron_win64"
    disconnect_grace_sec: float = 5.0

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path
