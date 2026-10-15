"""Shared task identities and adapter contracts for frozen benchmark campaigns."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class TaskIdentity:
    """Persisted identity; older Aerial campaigns omit dataset/configuration."""

    split: str
    episode_id: str
    dataset: str | None = None
    configuration: str | None = None

    @classmethod
    def from_task(cls, task: Mapping[str, Any]) -> "TaskIdentity":
        configuration = task.get("configuration_id")
        if "image_budget" in task:
            configuration = f"images_{task['image_budget']}"
        value = cls(task["split"], task["episode_id"], task.get("dataset"), configuration)
        for item in (value.dataset, value.configuration, value.split, value.episode_id):
            if item is not None and (not isinstance(item, str) or not item or item in {".", ".."} or "/" in item or "\\" in item):
                raise ValueError("Campaign identity fields must be non-empty path components")
        return value

    @property
    def key(self) -> str:
        return "/".join(item for item in (self.dataset, self.configuration, self.split, self.episode_id) if item is not None)

    def attempt_dir(self, output: Path, number: int) -> Path:
        if number < 1:
            raise ValueError("Attempt numbers start at one")
        parts = [item for item in (self.dataset, self.configuration, self.split) if item is not None]
        return Path(output).joinpath(*parts, "attempts", self.episode_id, f"attempt_{number:02d}")


@dataclass(frozen=True)
class PreparedCampaignSplit:
    """One dataset-owned selection and its resolved configuration, before flight."""

    config: dict[str, Any]
    tasks: tuple[dict[str, Any], ...]
    validation: dict[str, Any]


class CampaignAdapter(Protocol):
    """The common controller calls these dataset-owned evaluation boundaries."""

    dataset: str
    metric_fields: tuple[str, ...]

    def prepare(self, config: dict[str, Any], *, require_execution_ready: bool = False) -> PreparedCampaignSplit: ...

    def execute(self, config: dict[str, Any], task: dict[str, Any], directory: Path) -> dict[str, Any]: ...


def task_config(manifest: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Resolve explicit config keys; legacy manifests indexed configs by split."""
    return manifest["configs"][task.get("config_key", task["split"])]


def primary_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Resource/model ownership uses the campaign's declared primary config."""
    key = manifest.get("primary_config_key")
    if key is not None:
        return manifest["configs"][key]
    if manifest.get("tasks"):
        return task_config(manifest, manifest["tasks"][0])
    return next(iter(manifest["configs"].values()))
