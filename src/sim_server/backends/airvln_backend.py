"""Provide the first AirVLN-style backend skeleton."""

from __future__ import annotations

from src.adapters.uav_primitives import NullUavPrimitiveAdapter, UavPrimitiveAdapter
from src.env_client.schemas import ActionResult, MapPackage, Observation, SceneInstance


class AirVlnBackend:
    """AirVLN-style scene backend skeleton.

    The first implementation supports dry-run operation through a primitive UAV
    adapter. Real scene process startup is handled by SimulationServer before the
    backend receives the SceneInstance.
    """

    def __init__(self, uav: UavPrimitiveAdapter | None = None) -> None:
        # Use a real or dry-run UAV primitive adapter behind the backend.
        self.uav = uav or NullUavPrimitiveAdapter()

    def start(self, package: MapPackage, scene: SceneInstance) -> SceneInstance:
        # Mark the scene running once SimulationServer has prepared it.
        scene.status = "running"
        return scene

    def stop(self, scene: SceneInstance) -> None:
        # Mark backend scene state as stopped.
        scene.status = "stopped"

    def get_observation(
        self, scene: SceneInstance, agent_id: str, modalities: list[str]
    ) -> Observation:
        # Build raw observation data from primitive UAV calls.
        data = {}
        if "pose" in modalities:
            data["pose"] = self.uav.get_pose().data
        frame_modalities = [m for m in modalities if m != "pose"]
        if frame_modalities:
            data["frames"] = self.uav.get_frames(frame_modalities).data
        return Observation(scene.scene_instance_id, agent_id, modalities, data)

    def execute_action(
        self, scene: SceneInstance, agent_id: str, action: str, args: dict
    ) -> ActionResult:
        # Dispatch a primitive action and wrap the result for EnvClient.
        result = self._dispatch(action, args)
        return ActionResult(result.ok, action, result.message, result.data or {})

    def _dispatch(self, action: str, args: dict):
        # Map server action names onto the UAV primitive adapter.
        if action == "connect":
            return self.uav.connect()
        if action == "takeoff":
            return self.uav.takeoff()
        if action == "hover":
            return self.uav.hover()
        if action == "land":
            return self.uav.land()
        if action == "move_body":
            return self.uav.move_body(
                float(args.get("vx", 0.0)),
                float(args.get("vy", 0.0)),
                float(args.get("vz", 0.0)),
                float(args.get("duration", 0.0)),
            )
        if action == "turn_relative":
            return self.uav.turn_relative(float(args["yaw_deg"]))
        if action == "ascend":
            return self.uav.ascend(float(args["meters"]))
        if action == "descend":
            return self.uav.descend(float(args["meters"]))
        if action == "get_pose":
            return self.uav.get_pose()
        if action == "get_frames":
            return self.uav.get_frames(list(args.get("modalities", ["rgb"])))
        raise ValueError(f"Unsupported UAV primitive action: {action}")
