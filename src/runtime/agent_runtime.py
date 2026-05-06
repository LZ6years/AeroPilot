"""Provide the first minimal Agent Runtime shell."""

from __future__ import annotations

from src.env_client.base import EnvClient
from src.runtime.record_writer import RecordWriter
from src.tools.dispatcher import ToolDispatcher


class AgentRuntime:
    """Minimal runtime shell that uses EnvClient as its only environment path."""

    def __init__(self, env_client: EnvClient, recorder: RecordWriter) -> None:
        # Wire runtime dependencies while keeping environment access behind EnvClient.
        self.env_client = env_client
        self.recorder = recorder
        self.tools = ToolDispatcher(env_client)

    def run_smoke_test(self, map_id: str, agent_id: str = "uav_1") -> None:
        # Run a minimal environment and UAV primitive verification loop.
        scene = self.env_client.start_scene(map_id)
        try:
            self.recorder.write_json("scene_instance.json", scene.to_dict())
            self.recorder.event("scene_started", scene.to_dict())

            for action, args in [
                ("connect", {}),
                ("takeoff", {}),
                ("hover", {}),
                ("turn_relative", {"yaw_deg": 5}),
            ]:
                result = self.tools.call(scene.scene_instance_id, agent_id, f"uav.{action}", args)
                self.recorder.event("action_completed", result.to_dict())
                if not result.ok:
                    raise RuntimeError(f"Action failed: {action}: {result.message}")

            observation = self.tools.observe(
                scene.scene_instance_id, agent_id, ["rgb", "depth", "pose"]
            )
            self.recorder.write_json("observation.json", observation.to_dict())
            self.recorder.event("observation_updated", observation.to_dict())

            result = self.tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
            self.recorder.event("action_completed", result.to_dict())
        finally:
            self.env_client.stop_scene(scene.scene_instance_id)
            self.recorder.event("scene_stopped", scene.to_dict())
