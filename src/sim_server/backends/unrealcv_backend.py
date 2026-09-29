"""Bind scene capabilities and canonical responses to a UnrealCV UAV adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.adapters.uav_primitives import PrimitiveResult, UavPrimitiveAdapter
from src.adapters.unrealcv.uav import UnrealCvUavAdapter
from src.adapters.unrealcv.client import probe_unrealcv_endpoint
from src.env_client.schemas import ActionResult, MapPackage, Observation, SceneCapabilities, SceneInstance
from src.uav.config import UavActionConfig
from src.uav import CANONICAL_VERTICAL_SIGN, move_body_velocity_from_args
from .base import BackendProbe
from .ground_clearance import enrich_pose_with_agl


class UnrealCvBackend:
    """Own per-scene adapter binding; protocol I/O belongs to src.adapters."""

    def __init__(self, action_config: UavActionConfig | None = None,
                 artifact_dir: str | Path | None = None, save_artifacts: bool = False,
                 uav: UavPrimitiveAdapter | None = None) -> None:
        self.action_config = action_config or UavActionConfig()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = save_artifacts
        self.package: MapPackage | None = None
        self.uav = uav
        self._injected_uav = uav is not None

    def probe(self, package: MapPackage, host: str, port: int) -> BackendProbe:
        return BackendProbe(*probe_unrealcv_endpoint(host, port))

    def describe_capabilities(self, package: MapPackage) -> SceneCapabilities:
        """Describe action semantics selected by the MapPackage strategy."""

        target = str(package.unrealcv.get("control_target", "camera")).lower()
        control_mode = "actor_pose" if target == "actor" else "camera_pose"
        return SceneCapabilities(
            protocol="unrealcv",
            control_mode=control_mode,
            supported_actions=(
                "connect",
                "hover",
                "move_body",
                "turn_relative",
                "set_pose",
                "ascend",
                "descend",
                "get_pose",
                "get_frames",
            ),
            observation_modalities=("rgb", "depth", "pose", "segmentation"),
            requires_takeoff=False,
            supports_soft_reset=False,
        )

    def start(self, package: MapPackage, scene: SceneInstance) -> SceneInstance:
        self.package = package
        if not self._injected_uav:
            self.uav = UnrealCvUavAdapter(
                scene.control_host, scene.control_port, package.unrealcv,
                artifact_dir=self.artifact_dir, save_artifacts=self.save_artifacts,
            )
        scene.capabilities = self.describe_capabilities(package)
        scene.status = "running"
        return scene

    def stop(self, scene: SceneInstance) -> None:
        self.release_connection(scene)
        scene.status = "stopped"

    def release_connection(self, scene: SceneInstance) -> None:
        """Allow scenario editors exclusive access without stopping the scene."""
        close = getattr(self.uav, "close", None)
        if callable(close):
            close()

    def reset(self, scene: SceneInstance) -> SceneInstance:
        # UnrealCV has no portable reset; capabilities continue to declare this.
        scene.status = "running"
        return scene

    def get_observation(self, scene: SceneInstance, agent_id: str,
                        modalities: list[str],
                        capture_sources: dict[str, str] | None = None) -> Observation:
        if self.uav is None:
            raise RuntimeError("UnrealCV Backend has no scene-bound adapter; start the scene first")
        data: dict[str, Any] = {}
        if "pose" in modalities:
            result = self.uav.get_pose()
            if not result.ok:
                raise RuntimeError(result.message)
            data["pose"] = self._enrich_pose(result.data or {})
        frame_modalities = [m for m in modalities if m != "pose"]
        if frame_modalities:
            result = self.uav.get_frames(frame_modalities, capture_sources)
            if not result.ok:
                raise RuntimeError(result.message)
            data["frames"] = (result.data or {}).get("frames", {})
        return Observation(scene.scene_instance_id, agent_id, modalities, data)

    def execute_action(self, scene: SceneInstance, agent_id: str,
                       action: str, args: dict) -> ActionResult:
        if self.uav is None:
            raise RuntimeError("UnrealCV Backend has no scene-bound adapter; start the scene first")
        result = self._dispatch(action, args)
        data = dict(result.data or {})
        if action == "get_pose":
            data = {"pose": self._enrich_pose(data)}
        elif isinstance(data.get("pose"), dict):
            data["pose"] = self._enrich_pose(data["pose"])
        return ActionResult(result.ok, action, result.message, data)

    def _enrich_pose(self, pose: dict[str, Any]) -> dict[str, Any]:
        # The shared Null adapter wraps poses; real adapters return the pose body.
        enriched = dict(pose.get("pose", pose))
        enriched.setdefault("coordinate_frame",
                            "unrealcv_world_m" if self._control_target() == "actor"
                            else "unrealcv_camera_native")
        return enrich_pose_with_agl(enriched, landed_origin_z_m=self._landed_origin_z_m())

    def _control_target(self) -> str:
        if self.package is None:
            return "camera"
        return str(self.package.unrealcv.get("control_target", "camera")).lower()

    def _landed_origin_z_m(self) -> float | None:
        """Return the optional local landing calibration for Actor control.

        Camera control has no vehicle-origin landing meaning, so even a stale
        actor block must not accidentally produce AGL in that mode.
        """

        if self.package is None or self._control_target() != "actor":
            return None
        actor = self.package.unrealcv.get("actor", {})
        if not isinstance(actor, dict) or "landed_origin_z_m" not in actor:
            return None
        return float(actor["landed_origin_z_m"])

    def _dispatch(self, action: str, args: dict):
        # Map server action names onto the UAV primitive adapter.
        if action == "connect":
            return self.uav.connect()
        if action == "takeoff":
            return PrimitiveResult(False, "UnrealCV does not support takeoff", {"pose": self.uav.get_pose().data or {}})
        if action == "hover":
            return self.uav.hover()
        if action == "land":
            return PrimitiveResult(False, "UnrealCV does not support land", {"pose": self.uav.get_pose().data or {}})
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
