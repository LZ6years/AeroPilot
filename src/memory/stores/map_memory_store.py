"""Lightweight map memory store for MVP recall."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MapMemoryStore:
    entries: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "MapMemoryStore":
        manifest_path = Path(path)
        if not manifest_path.exists():
            return cls()
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        raw_entries = data.get("entries", data if isinstance(data, list) else [])
        store = cls()
        for entry in raw_entries:
            if isinstance(entry, dict):
                store.add(entry)
        return store

    def add(self, entry: dict[str, Any]) -> None:
        self.entries.append(_normalize_entry(entry))

    def recall(self, map_id: str, scene_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        matches = []
        for entry in self.entries:
            if entry.get("map_id") != map_id:
                continue
            if scene_id is not None and entry.get("scene_id") not in (None, scene_id):
                continue
            matches.append(dict(entry))
        return matches[-limit:]

    def save(self, path: str | Path) -> Path:
        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest_path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    data = dict(entry)
    data.setdefault("scene_id", None)
    data.setdefault("region_hint", "")
    data.setdefault("pose_hint", {})
    data.setdefault("landmark_description", "")
    data.setdefault("visual_evidence_ref", None)
    data.setdefault("depth_evidence_ref", None)
    data.setdefault("confidence", "low")
    data.setdefault("last_seen_at", None)
    data.setdefault("requires_reconfirmation", True)
    data.setdefault("risk_notes", [])
    data.setdefault("coordinate_frame", "airsim_ned")
    data.setdefault("origin_ref", "scene_start_pose")
    data.setdefault("yaw_ref", "airsim_world_yaw_deg")
    data.setdefault("pose_units", "meters")
    data.setdefault("altitude_ref", "airsim_ned_z")
    if "map_id" not in data:
        raise ValueError("MapMemory entry requires map_id")
    return data
