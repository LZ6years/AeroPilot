"""Implement the in-process EnvClient for the first pseudo C/S slice."""

from __future__ import annotations

from src.env_client.schemas import ActionResult, Observation, SceneInstance
from src.sim_server.server import LocalSimulationServer


class LocalEnvClient:
    """Local pseudo C/S client.

    It keeps the EnvClient boundary while delegating calls to an in-process
    LocalSimulationServer.
    """

    def __init__(self, server: LocalSimulationServer) -> None:
        # Keep a reference to the local server implementation.
        self.server = server

    def start_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        # Forward scene startup to the local simulation server.
        return self.server.start_scene(map_id, scene_id)

    def get_or_start_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        # Reuse an existing running scene or start a new one through the server.
        return self.server.get_or_start_scene(map_id, scene_id)

    def stop_scene(self, scene_instance_id: str) -> None:
        # Forward scene shutdown to the local simulation server.
        self.server.stop_scene(scene_instance_id)

    def reset_scene(self, scene_instance_id: str) -> SceneInstance:
        # Forward scene reset to the local simulation server.
        return self.server.reset_scene(scene_instance_id)

    def health_check(self) -> bool:
        # Forward health checks to the local simulation server.
        return self.server.health_check()

    def get_observation(
        self, scene_instance_id: str, agent_id: str, modalities: list[str]
    ) -> Observation:
        # Forward observation requests without adding Agent logic.
        return self.server.get_observation(scene_instance_id, agent_id, modalities)

    def execute_action(
        self, scene_instance_id: str, agent_id: str, action: str, args: dict
    ) -> ActionResult:
        # Forward primitive actions without doing planning.
        return self.server.execute_action(scene_instance_id, agent_id, action, args)
