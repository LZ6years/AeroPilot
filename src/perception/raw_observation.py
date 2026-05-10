"""Normalize raw environment observations without adding visual semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.env_client.schemas import Observation


@dataclass(frozen=True)
class RawObservation:
    scene_instance_id: str
    agent_id: str
    modalities: list[str]
    data: dict[str, Any]
    pose: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    source: str = "phase_step"

    @classmethod
    def from_env_observation(
        cls,
        observation: Observation | dict[str, Any],
        source: str = "phase_step",
        timestamp: str | None = None,
    ) -> "RawObservation":
        payload = observation.to_dict() if isinstance(observation, Observation) else dict(observation)
        data = dict(payload.get("data", {}))
        return cls(
            scene_instance_id=str(payload.get("scene_instance_id", "")),
            agent_id=str(payload.get("agent_id", "")),
            modalities=[str(item) for item in payload.get("modalities", [])],
            data=data,
            pose=_dict_or_empty(data.get("pose")),
            frames=_dict_or_empty(data.get("frames")),
            timestamp=timestamp or _utc_now(),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
