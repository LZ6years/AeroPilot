"""Runtime ownership of acquired and reusable simulator scenes."""

from __future__ import annotations

from collections.abc import Callable

from src.env_client import EnvClient, SceneInstance
from src.runtime.task.session import InstructionSession


class SceneSessionService:
    """Track scene identity, reuse, preflight state, and acquisition rollback."""

    def __init__(
        self,
        env_client: EnvClient,
        *,
        record_contract: Callable[[SceneInstance], None],
        debug_event: Callable[[str, dict], None],
        on_forget: Callable[[SceneInstance], None] | None = None,
    ) -> None:
        self.env_client = env_client
        self.record_contract = record_contract
        self.debug_event = debug_event
        self.on_forget = on_forget
        self.preflight_scene_ids: set[str] = set()
        self.unusable_scene_ids: set[str] = set()
        self.active_reused_scenes: dict[tuple[str, str | None], SceneInstance] = {}

    def acquire(
        self, map_id: str, policy: str, scene_id: str | None
    ) -> SceneInstance:
        if policy == "reuse_running":
            key = self.scene_key(map_id, scene_id)
            known = self.active_reused_scenes.get(key)
            scene = self.env_client.get_or_start_scene(map_id, scene_id)
            self.active_reused_scenes[key] = scene
            self._record_acquired(
                scene,
                reused=known is not None
                and known.scene_instance_id == scene.scene_instance_id,
            )
            return scene
        scene = self.env_client.start_scene(map_id, scene_id)
        self._record_acquired(scene, reused=False)
        return scene

    def reused(self, map_id: str, scene_id: str | None) -> SceneInstance | None:
        return self.active_reused_scenes.get(self.scene_key(map_id, scene_id))

    def is_preflighted(self, scene: SceneInstance) -> bool:
        return scene.scene_instance_id in self.preflight_scene_ids

    def mark_preflighted(self, scene: SceneInstance) -> None:
        self.preflight_scene_ids.add(scene.scene_instance_id)

    def is_unusable(self, scene: SceneInstance) -> bool:
        return scene.scene_instance_id in self.unusable_scene_ids

    def mark_unusable(self, scene: SceneInstance) -> None:
        self.unusable_scene_ids.add(scene.scene_instance_id)

    def assert_session_usable(self, session: InstructionSession) -> None:
        if self.is_unusable(session.scene):
            from src.env_client.errors import SceneRecoveryRequired

            raise SceneRecoveryRequired(
                "Scene must be recreated before another instruction"
            )

    def mark_session_unusable(self, session: InstructionSession) -> None:
        self.mark_unusable(session.scene)

    def stop_session(self, session: InstructionSession) -> None:
        self.stop(session.scene)

    def stop(self, scene: SceneInstance) -> None:
        self.env_client.stop_scene(scene.scene_instance_id)
        self.preflight_scene_ids.discard(scene.scene_instance_id)
        self.forget(scene)
        self.debug_event("scene_stopped", scene.to_dict())

    def forget(self, scene: SceneInstance) -> None:
        self.unusable_scene_ids.discard(scene.scene_instance_id)
        for key, active in list(self.active_reused_scenes.items()):
            if active.scene_instance_id == scene.scene_instance_id:
                self.active_reused_scenes.pop(key, None)
        if self.on_forget is not None:
            self.on_forget(scene)

    @staticmethod
    def scene_key(map_id: str, scene_id: str | None) -> tuple[str, str | None]:
        return map_id, str(scene_id) if scene_id is not None else None

    def _record_acquired(self, scene: SceneInstance, *, reused: bool) -> None:
        try:
            self.record_contract(scene)
            self.debug_event("scene_reused" if reused else "scene_started", scene.to_dict())
        except Exception as original:
            if not reused:
                try:
                    self.env_client.stop_scene(scene.scene_instance_id)
                except Exception as cleanup_error:
                    cleanup_errors = list(
                        getattr(original, "cleanup_errors", [])
                    )
                    cleanup_errors.append(
                        {
                            "stage": "scene_acquisition_rollback",
                            "error_type": type(cleanup_error).__name__,
                            "error": str(cleanup_error),
                        }
                    )
                    original.cleanup_errors = cleanup_errors
                else:
                    self.preflight_scene_ids.discard(scene.scene_instance_id)
                    self.forget(scene)
            raise
