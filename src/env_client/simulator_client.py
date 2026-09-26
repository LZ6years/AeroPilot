"""Provide the single in-process EnvClient for every simulator deployment."""

from __future__ import annotations

from typing import Any

from src.env_client.base import ENV_CLIENT_API_VERSION
from src.env_client.observation_profile import ObservationProfile
from src.env_client.schemas import ActionResult, Observation, SceneInstance


class SimulatorEnvClient:
    """Delegate the stable EnvClient contract to a composed simulation service."""

    API_VERSION = ENV_CLIENT_API_VERSION
    """Public EnvClient contract implemented by this concrete client."""

    def __init__(
        self,
        service,
        observation_profile: ObservationProfile | None = None,
    ) -> None:
        """Bind one run-level observation profile to the stable client API.

        Benchmark and Demo code choose the profile.  Callers still request
        canonical names such as ``rgb`` and ``depth``; the concrete Backend
        receives the profile's capture-source mapping separately.
        """

        self.service = service
        self.observation_profile = observation_profile or ObservationProfile()

    def start_scene(
        self, map_id: str, scene_id: str | None = None
    ) -> SceneInstance:
        return self.service.start_scene(map_id, scene_id)

    def get_or_start_scene(
        self, map_id: str, scene_id: str | None = None
    ) -> SceneInstance:
        return self.service.get_or_start_scene(map_id, scene_id)

    def stop_scene(self, scene_instance_id: str) -> None:
        self.service.stop_scene(scene_instance_id)

    def reset_scene(self, scene_instance_id: str) -> SceneInstance:
        return self.service.reset_scene(scene_instance_id)

    def health_check(self) -> bool:
        return self.service.health_check()

    def get_observation(
        self,
        scene_instance_id: str,
        agent_id: str,
        modalities: list[str],
    ) -> Observation:
        return self.service.get_observation(
            scene_instance_id,
            agent_id,
            modalities,
            capture_sources=self.observation_profile.capture_sources,
        )

    def consume_post_action_state(self, scene_instance_id, agent_id, token):
        """Optional local control evidence; unsupported services fall back safely."""
        consume = getattr(self.service, "consume_post_action_state", None)
        return consume(scene_instance_id, agent_id, token) if callable(consume) else None

    def execute_action(
        self,
        scene_instance_id: str,
        agent_id: str,
        action: str,
        args: dict[str, Any],
    ) -> ActionResult:
        return self.service.execute_action(
            scene_instance_id, agent_id, action, args
        )
