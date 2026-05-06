"""Define shared request and response schemas for environment C/S calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PortPolicy:
    server_port: int = 30000
    airsim_port_start: int = 30001
    max_scene_num: int = 8


@dataclass(frozen=True)
class MapPackage:
    map_id: str
    provider: str
    scene_id: str
    root: str | None
    launch: dict[str, Any]
    airsim: dict[str, Any]
    control: dict[str, Any] = field(default_factory=dict)
    unrealcv: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapPackage":
        # Validate and normalize a map configuration dictionary.
        missing = [key for key in ("map_id", "provider", "scene_id") if key not in data]
        if missing:
            raise ValueError(f"MapPackage missing required fields: {', '.join(missing)}")
        provider = str(data["provider"])
        airsim = data.get("airsim", {})
        unrealcv = data.get("unrealcv", {})
        control = dict(data.get("control", {}))
        control.setdefault("protocol", _legacy_protocol_for_provider(provider))
        protocol = str(control["protocol"])
        if protocol == "airsim":
            for key in ("vehicle_name", "camera_name"):
                if key not in airsim:
                    raise ValueError(f"MapPackage {data['map_id']} missing airsim.{key}")
        if protocol == "unrealcv":
            for key in ("host", "port", "camera_id"):
                if key not in unrealcv:
                    raise ValueError(f"MapPackage {data['map_id']} missing unrealcv.{key}")
        launch = data.get("launch", {})
        root = data.get("root")
        if provider in {"airvln", "local_ue", "unrealcv", "packaged_unreal"}:
            remote_or_manual = launch.get("manual", False) or launch.get("method") == "existing_process"
            if not root and not remote_or_manual:
                raise ValueError(f"MapPackage {data['map_id']} requires root unless launch.manual=true")
        if protocol not in {"airsim", "unrealcv"}:
            raise ValueError(f"MapPackage {data['map_id']} has unsupported control.protocol: {protocol}")
        return cls(
            map_id=str(data["map_id"]),
            provider=provider,
            scene_id=str(data["scene_id"]),
            root=str(root) if root is not None else None,
            launch=dict(launch),
            airsim=dict(airsim),
            control=control,
            unrealcv=dict(unrealcv),
            resources=dict(data.get("resources", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        # Convert the package into JSON-serializable data.
        return asdict(self)

    @property
    def control_protocol(self) -> str:
        # Return the runtime control protocol used by the simulation backend.
        return str(self.control.get("protocol", _legacy_protocol_for_provider(self.provider)))


def _legacy_protocol_for_provider(provider: str) -> str:
    # Keep old provider-based map configs working while the config model migrates.
    if provider == "unrealcv":
        return "unrealcv"
    return "airsim"


@dataclass
class SceneInstance:
    scene_instance_id: str
    map_id: str
    scene_id: str
    machine_id: str
    host: str
    server_port: int
    airsim_port: int
    settings_path: str
    process_id: int | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        # Convert the scene instance into JSON-serializable data.
        return asdict(self)


@dataclass(frozen=True)
class ActionRequest:
    scene_instance_id: str
    agent_id: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    action: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Convert the action result into JSON-serializable data.
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    scene_instance_id: str
    agent_id: str
    modalities: list[str]
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        # Convert the observation into JSON-serializable data.
        return asdict(self)
