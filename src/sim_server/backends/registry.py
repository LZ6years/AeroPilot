"""Create simulator backends by protocol without coupling EnvClient to SDKs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.runtime.configs.action_config import UavActionConfig
from .base import SimulationBackend


@dataclass(frozen=True)
class BackendContext:
    """Dependencies shared by backend factories for one scene instance."""

    action_config: UavActionConfig
    artifact_dir: Path
    save_artifacts: bool


BackendFactory = Callable[[BackendContext], SimulationBackend]


class BackendRegistry:
    """Map stable protocol names to per-scene backend factories."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, protocol: str, factory: BackendFactory) -> None:
        key = protocol.strip().lower()
        if not key:
            raise ValueError("Backend protocol cannot be empty")
        if key in self._factories:
            raise ValueError(f"Backend protocol already registered: {key}")
        self._factories[key] = factory

    def create(self, protocol: str, context: BackendContext) -> SimulationBackend:
        key = protocol.strip().lower()
        try:
            factory = self._factories[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(
                f"Unsupported control protocol: {protocol}; available: {available}"
            ) from exc
        backend = factory(context)
        return backend

    def protocols(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    @classmethod
    def defaults(cls) -> "BackendRegistry":
        # Import concrete backends lazily so registration remains the only
        # composition point and individual modules can be tested in isolation.
        from .airsim_backend import AirSimBackend
        from .unrealcv_backend import UnrealCvBackend

        registry = cls()
        registry.register(
            "airsim",
            lambda context: AirSimBackend(
                # Execution mode now controls process ownership only. A
                # production Registry always builds a real protocol adapter.
                action_config=context.action_config,
                artifact_dir=context.artifact_dir,
                save_artifacts=context.save_artifacts,
            ),
        )
        registry.register(
            "unrealcv",
            lambda context: UnrealCvBackend(
                action_config=context.action_config,
                artifact_dir=context.artifact_dir,
                save_artifacts=context.save_artifacts,
            ),
        )
        return registry
