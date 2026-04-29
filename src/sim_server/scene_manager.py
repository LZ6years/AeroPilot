"""Store active SceneInstance objects for the local server."""

from __future__ import annotations

from src.env_client.schemas import SceneInstance


class SceneManager:
    def __init__(self) -> None:
        # Keep active scenes keyed by instance id.
        self._scenes: dict[str, SceneInstance] = {}

    def add(self, scene: SceneInstance) -> None:
        # Register a newly created scene instance.
        self._scenes[scene.scene_instance_id] = scene

    def get(self, scene_instance_id: str) -> SceneInstance:
        # Fetch a scene or raise a clear key error.
        try:
            return self._scenes[scene_instance_id]
        except KeyError as exc:
            raise KeyError(f"Unknown scene instance: {scene_instance_id}") from exc

    def update_status(self, scene_instance_id: str, status: str) -> SceneInstance:
        # Update and return a scene's lifecycle status.
        scene = self.get(scene_instance_id)
        scene.status = status
        return scene

    def remove(self, scene_instance_id: str) -> SceneInstance:
        # Remove a scene from active tracking.
        try:
            return self._scenes.pop(scene_instance_id)
        except KeyError as exc:
            raise KeyError(f"Unknown scene instance: {scene_instance_id}") from exc
