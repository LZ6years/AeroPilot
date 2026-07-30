"""Define the backend protocol used inside SimulationServer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.env_client.schemas import (
    ActionResult,
    MapPackage,
    Observation,
    SceneCapabilities,
    SceneInstance,
)


@dataclass(frozen=True)
class BackendProbe:
    """Report whether an endpoint speaks the expected simulator protocol."""

    ready: bool
    detail: str = ""


class SimulationBackend(Protocol):
    def probe(self, package: MapPackage, host: str, port: int) -> BackendProbe:
        # Check protocol health without taking ownership of the simulator process.
        ...

    def describe_capabilities(self, package: MapPackage) -> SceneCapabilities:
        # Report protocol and action behavior before Runtime preflight.
        ...

    def start(self, package: MapPackage, scene: SceneInstance) -> SceneInstance:
        # Prepare backend state for a new scene instance.
        ...

    def stop(self, scene: SceneInstance) -> None:
        # Stop backend resources for a scene instance.
        ...

    def reset(self, scene: SceneInstance) -> SceneInstance:
        # Soft-reset backend state without changing process ownership.
        ...

    def get_observation(
        self,
        scene: SceneInstance,
        agent_id: str,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> Observation:
        # Read canonical observations using optional run-level capture choices.
        ...

    def execute_action(
        self, scene: SceneInstance, agent_id: str, action: str, args: dict
    ) -> ActionResult:
        # Execute one primitive backend action.
        ...
