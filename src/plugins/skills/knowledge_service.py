"""Instruction-scoped immutable knowledge snapshots and incremental delivery.

No model calls or script execution occur here. Construct one service for each
instruction from the enabled catalog; do not share delivery state across runs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from src.plugins.skills.catalog import SkillCatalog
from src.plugins.skills.loader import read_declared_reference


class KnowledgeService:
    """Freeze enabled knowledge and track which bytes this instruction received."""

    def __init__(self, catalog: SkillCatalog) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._delivered: dict[str, set[str]] = {}
        for skill in catalog.skills:
            # Callable capabilities are executable tools, not knowledge files.
            if skill.capability is not None:
                continue
            if skill.name in self._entries:
                raise ValueError(f"ambiguous knowledge name: {skill.name}")
            manifest = skill.reference_manifest()
            references = {item["path"]: read_declared_reference(skill, item["path"])
                          for item in manifest}
            snapshot = {"body": skill.body, "references": references}
            version = hashlib.sha256(json.dumps(
                snapshot, sort_keys=True, ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            self._entries[skill.name] = {
                "name": skill.name, "description": skill.description,
                "version": version, "body": skill.body,
                "references": references, "reference_manifest": manifest,
            }

    def catalog(self) -> list[dict[str, str]]:
        """Lightweight model-visible index without paths, bodies or settings."""
        return [{"name": entry["name"], "description": entry["description"]}
                for entry in self._entries.values()]

    def loaded_index(self) -> list[dict[str, Any]]:
        return [{"name": name, "version": self._entries[name]["version"],
                 "reference_paths": sorted(path for path in delivered if path)}
                for name, delivered in self._delivered.items()]

    def load(self, name: str, reference_paths: list[str] | None = None) -> dict[str, Any]:
        """Return only newly requested text; invalid requests change no state."""
        if not isinstance(name, str) or name not in self._entries:
            raise ValueError("unknown or disabled knowledge name")
        requested = [] if reference_paths is None else reference_paths
        if not isinstance(requested, list) or any(not isinstance(path, str) for path in requested):
            raise ValueError("reference_paths must be a list of manifest paths")
        if len(set(requested)) != len(requested):
            raise ValueError("reference_paths must be unique")
        entry = self._entries[name]
        if any(path not in entry["references"] for path in requested):
            raise ValueError("reference is not declared by this knowledge entry")
        delivered = self._delivered.get(name, set())
        result: dict[str, Any] = {
            "name": name, "version": entry["version"],
            "reference_manifest": copy.deepcopy(entry["reference_manifest"]),
            "body_included": "" not in delivered,
            "references": [{"path": path, "text": entry["references"][path]}
                           for path in requested if path not in delivered],
        }
        if "" not in delivered:
            result["body"] = entry["body"]
        self._delivered[name] = delivered | {"", *requested}
        return result
