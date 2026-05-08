"""Provide the first AirVLN-style backend skeleton."""

from __future__ import annotations

from pathlib import Path

from src.adapters.airsim_uav import AirSimUavAdapter
from src.adapters.uav_primitives import NullUavPrimitiveAdapter, UavPrimitiveAdapter
from src.env_client.schemas import ActionResult, MapPackage, Observation, SceneInstance
from src.runtime.configs.action_config import UavActionConfig


class AirVlnBackend:
    """AirVLN-style scene backend skeleton.

    The first implementation supports dry-run operation through a primitive UAV
    adapter. Real scene process startup is handled by SimulationServer before the
    backend receives the SceneInstance.
    """

    def __init__(
        self,
        uav: UavPrimitiveAdapter | None = None,
        use_real_uav: bool = False,
        action_config: UavActionConfig | None = None,
        artifact_dir: str | Path | None = None,
        save_artifacts: bool = False,
    ) -> None:
        # Use a real or dry-run UAV primitive adapter behind the backend.
        self.uav = uav or NullUavPrimitiveAdapter()
        self.use_real_uav = use_real_uav
        self.action_config = action_config or UavActionConfig()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = save_artifacts

    def start(self, package: MapPackage, scene: SceneInstance) -> SceneInstance:
        # Mark the scene running once SimulationServer has prepared it.
        if self.use_real_uav:
            self.uav = AirSimUavAdapter(
                host=str(package.airsim.get("host", scene.host)),
                port=int(package.airsim.get("port", scene.airsim_port)),
                vehicle_name=package.airsim["vehicle_name"],
                camera_name=package.airsim["camera_name"],
                mode=str(package.airsim.get("mode", "Multirotor")),
                artifact_dir=self.artifact_dir,
                save_artifacts=self.save_artifacts,
            )
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
            vx, vy, vz, duration = self._move_body_args(args)
            return self.uav.move_body(vx, vy, vz, duration)
        if action == "turn_relative":
            return self.uav.turn_relative(float(args.get("yaw_deg", self.action_config.yaw_step_deg)))
        if action == "set_pose":
            return self.uav.set_pose(
                float(args["x"]),
                float(args["y"]),
                float(args["z"]),
                float(args.get("yaw_deg", 0.0)),
                float(args.get("pitch_deg", 0.0)),
                float(args.get("roll_deg", 0.0)),
                bool(args.get("ignore_collision", True)),
            )
        if action == "ascend":
            speed_mps, duration = self._vertical_args(args)
            return self.uav.ascend(speed_mps, duration)
        if action == "descend":
            speed_mps, duration = self._vertical_args(args)
            return self.uav.descend(speed_mps, duration)
        if action == "get_pose":
            return self.uav.get_pose()
        if action == "get_frames":
            return self.uav.get_frames(list(args.get("modalities", ["rgb"])))
        raise ValueError(f"Unsupported UAV primitive action: {action}")

    def _move_body_args(self, args: dict) -> tuple[float, float, float, float]:
        # Resolve semantic direction or explicit body-frame velocity into a fixed-scale move.
        duration = float(args.get("duration", self.action_config.move_duration_sec))
        direction = str(args.get("direction", "")).lower()
        if direction:
            speed = self.action_config.linear_velocity_mps
            vertical_speed = self.action_config.vertical_velocity_mps
            direction_map = {
                "forward": (speed, 0.0, 0.0),
                "backward": (-speed, 0.0, 0.0),
                "left": (0.0, -speed, 0.0),
                "right": (0.0, speed, 0.0),
                "up": (0.0, 0.0, -vertical_speed),
                "down": (0.0, 0.0, vertical_speed),
            }
            try:
                vx, vy, vz = direction_map[direction]
            except KeyError as exc:
                raise ValueError(f"Unsupported move_body direction: {direction}") from exc
            return vx, vy, vz, duration
        return (
            float(args.get("vx", self.action_config.linear_velocity_mps)),
            float(args.get("vy", 0.0)),
            float(args.get("vz", 0.0)),
            duration,
        )

    def _vertical_args(self, args: dict) -> tuple[float, float]:
        # Resolve meters, speed, and duration into a vertical speed-duration command.
        speed_mps = abs(float(args.get("speed_mps", self.action_config.vertical_velocity_mps)))
        if speed_mps <= 0.0:
            raise ValueError("vertical speed_mps must be greater than 0")
        if "duration_sec" in args:
            return speed_mps, float(args["duration_sec"])
        if "duration" in args:
            return speed_mps, float(args["duration"])
        if "meters" in args:
            return speed_mps, abs(float(args["meters"])) / speed_mps
        if self.action_config.vertical_duration_sec > 0.0:
            return speed_mps, self.action_config.vertical_duration_sec
        return speed_mps, abs(self.action_config.vertical_step_m) / speed_mps
