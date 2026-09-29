"""Implement the generic AirSim backend used by every AirSim map package."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from uuid import uuid4
from contextlib import nullcontext

from src.adapters.airsim.uav import AirSimUavAdapter
from src.adapters.airsim.control import resolve_airsim_control_strategy
from src.adapters.airsim.client import probe_airsim_endpoint
from src.adapters.airsim.performance import AirSimPerformance
from src.adapters.uav_primitives import UavPrimitiveAdapter
from src.env_client.schemas import (
    ActionResult,
    MapPackage,
    Observation,
    SceneCapabilities,
    SceneInstance,
)
from src.uav.config import UavActionConfig
from src.uav import CANONICAL_VERTICAL_SIGN, move_body_velocity_from_args


from .base import BackendProbe


class AirSimBackend:
    """Route primitive actions and observations through one scene-owned adapter."""

    def __init__(
        self,
        uav: UavPrimitiveAdapter | None = None,
        action_config: UavActionConfig | None = None,
        artifact_dir: str | Path | None = None,
        save_artifacts: bool = False,
    ) -> None:
        # Explicit injection is the only replacement for the real SDK adapter.
        self.uav = uav
        self._injected_uav = uav is not None
        self.action_config = action_config or UavActionConfig()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = save_artifacts
        self._post_action_state = None

    def probe(self, package: MapPackage, host: str, port: int) -> BackendProbe:
        """Verify AirSim RPC health instead of accepting a raw TCP listener."""

        return BackendProbe(*probe_airsim_endpoint(host, port))

    def describe_capabilities(self, package: MapPackage) -> SceneCapabilities:
        mode = str(package.airsim.get("mode", "Multirotor"))
        strategy = resolve_airsim_control_strategy(
            mode,
            package.airsim.get("control_strategy"),
        )
        actions = (
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
        return SceneCapabilities(
            protocol="airsim",
            # Runtime needs the actual action semantics, not merely the server's
            # SimMode. The MapPackage retains ComputerVision/Multirotor.
            control_mode=strategy,
            supported_actions=actions,
            observation_modalities=("rgb", "depth", "pose", "segmentation"),
            requires_takeoff=False,
            supports_soft_reset=True,
        )

    def start(self, package: MapPackage, scene: SceneInstance) -> SceneInstance:
        # Mark the scene running once SimulationServer has prepared it.
        if not self._injected_uav:
            self.uav = AirSimUavAdapter(
                host=scene.control_host,
                port=scene.control_port,
                vehicle_name=package.airsim["vehicle_name"],
                camera_name=package.airsim["camera_name"],
                mode=str(package.airsim.get("mode", "Multirotor")),
                control_strategy=str(
                    package.airsim.get("control_strategy", "")
                ) or None,
                artifact_dir=self.artifact_dir,
                save_artifacts=self.save_artifacts,
            )
        scene.status = "running"
        scene.capabilities = self.describe_capabilities(package)
        return scene

    def stop(self, scene: SceneInstance) -> None:
        # Close only this process's RPC client; process ownership lives above.
        self._post_action_state = None
        close = getattr(self.uav, "close", None)
        if callable(close):
            close()
        scene.status = "stopped"

    def reset(self, scene: SceneInstance) -> SceneInstance:
        # Perform a soft simulator reset when supported by the concrete adapter.
        self._post_action_state = None
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
        performance = getattr(self.uav, "performance", None)
        if not isinstance(performance, AirSimPerformance):
            performance = None
        try:
            with performance.span("observation") if performance else nullcontext():
                return self._observe(scene, agent_id, modalities, capture_sources)
        except Exception:
            # A failed read cannot certify that the same connected paused state
            # is still available, even if no deliberate control write occurred.
            self._post_action_state = None
            raise
        finally:
            if performance:
                performance.flush("observation")

    def _observe(self, scene, agent_id, modalities, capture_sources):
        # Build raw observation data from primitive UAV calls.
        if self.uav is None:
            raise RuntimeError("AirSim Backend has no scene-bound adapter; start the scene first")
        data = {}
        if "pose" in modalities:
            result = self.uav.get_pose()
            if not result.ok:
                raise RuntimeError(f"AirSim pose observation failed: {result.message}")
            data["pose"] = result.data
        frame_modalities = [m for m in modalities if m != "pose"]
        if frame_modalities:
            result = self.uav.get_frames(
                frame_modalities, capture_sources
            )
            if not result.ok:
                raise RuntimeError(f"AirSim frame observation failed: {result.message}")
            data["frames"] = result.data
        return Observation(scene.scene_instance_id, agent_id, modalities, data)

    def execute_action(
        self, scene: SceneInstance, agent_id: str, action: str, args: dict
    ) -> ActionResult:
        # Dispatch a primitive action and wrap the result for EnvClient.
        # Every write invalidates the previous receipt, including failed writes.
        if action not in {"get_pose", "get_frames"}:
            self._post_action_state = None
        performance = getattr(self.uav, "performance", None)
        if not isinstance(performance, AirSimPerformance):
            performance = None
        try:
            with performance.span("action." + action) if performance else nullcontext():
                result = self._dispatch(action, args)
        except Exception:
            self._post_action_state = None
            raise
        finally:
            measurements = None
            if performance:
                measurements = performance.flush("action." + action)
        execution = dict(result.execution_state)
        data = dict(result.data or {})
        # The AirSim boundary owns this legacy primitive diagnostic envelope.
        # Runtime must neither interpret it nor forward it to the model.
        diagnostics = data.pop("action_diagnostics", None)
        if diagnostics is not None:
            execution["action_diagnostics"] = diagnostics
        if measurements is not None:
            execution["performance"] = measurements
        pose = execution.pop("verified_paused_pose", None)
        if result.ok and pose and not execution.get("requires_scene_reset"):
            token = uuid4().hex
            self._post_action_state = (scene.scene_instance_id, agent_id, token, deepcopy(pose))
            execution["post_action_state_token"] = token
        return ActionResult(result.ok, action, result.message, data, execution)

    def consume_post_action_state(self, scene_id, agent_id, token):
        """Consume an opaque scene/action receipt once, without another RPC."""
        cached = self._post_action_state
        if cached is None or cached[:3] != (scene_id, agent_id, token):
            return None
        self._post_action_state = None
        return deepcopy(cached[3])

    def _dispatch(self, action: str, args: dict):
        # Map server action names onto the UAV primitive adapter.
        if self.uav is None:
            raise RuntimeError("AirSim Backend has no scene-bound adapter; start the scene first")
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
            return self.uav.get_frames(list(args.get("modalities", ["rgb"])), args.get("capture_sources"))
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
