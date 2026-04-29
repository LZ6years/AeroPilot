"""Define the backend protocol used inside SimulationServer."""

from __future__ import annotations

from typing import Protocol

from src.env_client.schemas import ActionResult, MapPackage, Observation, SceneInstance


class SimulationBackend(Protocol):
    def start(self, package: MapPackage, scene: SceneInstance) -> SceneInstance:
        # Prepare backend state for a new scene instance.
        ...

    def stop(self, scene: SceneInstance) -> None:
        # Stop backend resources for a scene instance.
        ...

    def get_observation(
        self, scene: SceneInstance, agent_id: str, modalities: list[str]
    ) -> Observation:
        # Read raw observations from the backend.
        ...

    def execute_action(
        self, scene: SceneInstance, agent_id: str, action: str, args: dict
    ) -> ActionResult:
        # Execute one primitive backend action.
        ...
