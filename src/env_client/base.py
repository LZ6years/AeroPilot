"""Define the EnvClient protocol used by Agent Runtime."""

from __future__ import annotations

from typing import Protocol

from .schemas import ActionResult, Observation, SceneInstance


class EnvClient(Protocol):
    def start_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        # Request a scene instance from the simulation service.
        ...

    def stop_scene(self, scene_instance_id: str) -> None:
        # Stop a running scene instance.
        ...

    def reset_scene(self, scene_instance_id: str) -> SceneInstance:
        # Reset a scene instance to a usable state.
        ...

    def health_check(self) -> bool:
        # Check whether the environment service is reachable.
        ...

    def get_observation(
        self, scene_instance_id: str, agent_id: str, modalities: list[str]
    ) -> Observation:
        # Fetch raw observation data for one UAV in one scene.
        ...

    def execute_action(
        self, scene_instance_id: str, agent_id: str, action: str, args: dict
    ) -> ActionResult:
        # Execute a primitive action through the environment service.
        ...
