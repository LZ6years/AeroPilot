"""Implement the generic AirSim backend used by every AirSim map package."""

from __future__ import annotations

from pathlib import Path

from src.adapters.airsim_uav import AirSimUavAdapter
from src.adapters.uav_primitives import NullUavPrimitiveAdapter, UavPrimitiveAdapter
from src.env_client.schemas import (
    ActionResult,
    MapPackage,
    Observation,
    SceneCapabilities,
    SceneInstance,
)
from src.runtime.configs.action_config import UavActionConfig
from src.uav import CANONICAL_VERTICAL_SIGN, move_body_velocity_from_args


from .base import BackendProbe


class AirSimBackend:
    """Route primitive actions and observations through one scene-owned adapter."""

    def __init__(
        self,
        uav: UavPrimitiveAdapter | None = None,
        use_real_uav: bool = False,
        action_config: UavActionConfig | None = None,
        artifact_dir: str | Path | None = None,
        save_artifacts: bool = False,
    ) -> None:
        # Tests may inject an in-memory primitive adapter; the production
        # BackendRegistry always requests the real AirSim adapter.
        self.uav = uav or NullUavPrimitiveAdapter()
        self.use_real_uav = use_real_uav
        self.action_config = action_config or UavActionConfig()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = save_artifacts

    def probe(self, package: MapPackage, host: str, port: int) -> BackendProbe:
        """Verify AirSim RPC health instead of accepting a raw TCP listener."""

        if not self.use_real_uav:
            return BackendProbe(True, "in-memory AirSim test backend")
        try:
            import airsim

            client = airsim.VehicleClient(ip=host, port=port, timeout_value=2)
            ready = bool(client.ping())
            close = getattr(client, "close", None)
            if callable(close):
                close()
            return BackendProbe(ready, "AirSim RPC ping succeeded" if ready else "AirSim ping returned false")
        except Exception as exc:
            return BackendProbe(False, f"{type(exc).__name__}: {exc}")

    def describe_capabilities(self, package: MapPackage) -> SceneCapabilities:
        mode = str(package.airsim.get("mode", "Multirotor"))
        is_cv = mode.lower() in {"computervision", "computer_vision", "cv"}
        shared = (
            "connect",
            "hover",
            "move_body",
            "turn_relative",
            "set_pose",
            "ascend",
            "descend",
            "get_pose",
            "get_frames",
        )
        actions = shared if is_cv else (*shared[:1], "takeoff", *shared[1:2], "land", *shared[2:])
        return SceneCapabilities(
            protocol="airsim",
            control_mode="ComputerVision" if is_cv else "Multirotor",
            supported_actions=actions,
            observation_modalities=("rgb", "depth", "pose", "segmentation"),
            requires_takeoff=not is_cv,
            supports_soft_reset=True,
        )

    def start(self, package: MapPackage, scene: SceneInstance) -> SceneInstance:
        # Mark the scene running once SimulationServer has prepared it.
        if self.use_real_uav:
            self.uav = AirSimUavAdapter(
                host=scene.control_host,
                port=scene.control_port,
                vehicle_name=package.airsim["vehicle_name"],
                camera_name=package.airsim["camera_name"],
                mode=str(package.airsim.get("mode", "Multirotor")),
                artifact_dir=self.artifact_dir,
                save_artifacts=self.save_artifacts,
            )
        scene.status = "running"
        scene.capabilities = self.describe_capabilities(package)
        return scene

    def stop(self, scene: SceneInstance) -> None:
        # Close only this process's RPC client; process ownership lives above.
        close = getattr(self.uav, "close", None)
        if callable(close):
            close()
        scene.status = "stopped"

    def reset(self, scene: SceneInstance) -> SceneInstance:
        # Perform a soft simulator reset when supported by the concrete adapter.
        reset = getattr(self.uav, "reset", None)
        if callable(reset):
            result = reset()
            if not result.ok:
                raise RuntimeError(result.message)
        scene.status = "running"
        return scene

    def get_observation(
        self,
        scene: SceneInstance,
        agent_id: str,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> Observation:
        # Build raw observation data from primitive UAV calls.
        data = {}
        if "pose" in modalities:
            data["pose"] = self.uav.get_pose().data
        frame_modalities = [m for m in modalities if m != "pose"]
        if frame_modalities:
            data["frames"] = self.uav.get_frames(
                frame_modalities, capture_sources
            ).data
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
        # Resolve model-facing movement into canonical body-frame velocity. The
        # concrete primitive adapter owns any simulator-specific raw conversion.
        return move_body_velocity_from_args(args, self.action_config, vertical_sign=CANONICAL_VERTICAL_SIGN)

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
