"""Orchestrate UnrealCV scene lifecycle, pose control, and observations.

The Backend remains the protocol boundary used by ``SimulationServer``.
Camera-versus-actor motion is delegated to ``unrealcv_control`` and image
capture is delegated to ``unrealcv_capture`` so Benchmark-specific scenario
composition never enters the simulator layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.env_client.schemas import (
    ActionResult,
    MapPackage,
    Observation,
    SceneCapabilities,
    SceneInstance,
)
from src.runtime.configs.action_config import UavActionConfig
from src.uav import (
    UNREALCV_VERTICAL_SIGN,
    body_velocity_to_world_delta,
    move_body_velocity_from_args,
)

from .base import BackendProbe
from .unrealcv_capture import UnrealCvCapture
from .unrealcv_control import UnrealCvPoseControl, create_pose_control


@dataclass
class UnrealCvState:
    """Store canonical pose for explicitly injected in-memory test operation."""

    pose: list[float]


class UnrealCvBackend:
    """Implement the stable SimulationBackend contract for UnrealCV maps."""

    def __init__(
        self,
        use_real_client: bool = False,
        action_config: UavActionConfig | None = None,
        artifact_dir: str | Path | None = None,
        save_artifacts: bool = False,
    ) -> None:
        self.use_real_client = use_real_client
        self.action_config = action_config or UavActionConfig()
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = bool(save_artifacts)
        self.client: Any | None = None
        self.control: UnrealCvPoseControl | None = None
        self.capture: UnrealCvCapture | None = None
        self.package: MapPackage | None = None
        self.host = "127.0.0.1"
        self.port = 9010
        self.state = UnrealCvState([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])

    def probe(self, package: MapPackage, host: str, port: int) -> BackendProbe:
        """Check only that the endpoint speaks UnrealCV; do not mutate a scene."""

        if not self.use_real_client:
            return BackendProbe(True, "in-memory UnrealCV test backend")
        try:
            from unrealcv import Client

            client = Client((host, int(port)))
            client.connect()
            ready = bool(client.isconnected())
            if ready:
                client.disconnect()
            return BackendProbe(
                ready,
                "UnrealCV connected" if ready else "UnrealCV connection was rejected",
            )
        except Exception as exc:
            return BackendProbe(False, f"{type(exc).__name__}: {exc}")

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
        """Bind one per-scene Backend instance to its resolved endpoint."""

        self.package = package
        self.host = scene.control_host
        self.port = scene.control_port
        scene.capabilities = self.describe_capabilities(package)
        scene.status = "running"
        return scene

    def stop(self, scene: SceneInstance) -> None:
        """Disconnect only this Backend client; process ownership lives above."""

        if self.client is not None:
            self.client.disconnect()
        self.client = None
        self.control = None
        self.capture = None
        scene.status = "stopped"

    def reset(self, scene: SceneInstance) -> SceneInstance:
        """Keep the connection alive without claiming a generic UE reset."""

        # UnrealCV has no portable scene-reset command. Capabilities advertise
        # this honestly so Benchmark code can restore its episode start pose and
        # scenario objects through their respective owning layers.
        scene.status = "running"
        return scene

    def get_observation(
        self,
        scene: SceneInstance,
        agent_id: str,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> Observation:
        """Return canonical pose plus synchronized requested image modalities."""

        data: dict[str, Any] = {}
        if "pose" in modalities:
            data["pose"] = self._model_pose()
        frame_modalities = [modality for modality in modalities if modality != "pose"]
        if frame_modalities:
            data["frames"] = self._get_frames(
                frame_modalities, capture_sources
            )
        return Observation(scene.scene_instance_id, agent_id, modalities, data)

    def execute_action(
        self,
        scene: SceneInstance,
        agent_id: str,
        action: str,
        args: dict,
    ) -> ActionResult:
        """Map stable UAV primitive names onto the selected pose strategy."""

        label = self._control_label()
        if action == "connect":
            self._connect()
            return ActionResult(True, action, f"UnrealCV {label} connected")
        if action in {"takeoff", "land"}:
            return ActionResult(
                False,
                action,
                f"UnrealCV does not support {action}",
                {"pose": self._model_pose()},
            )
        if action == "hover":
            pose = self._get_pose()
            self._set_pose(pose)
            return ActionResult(
                True,
                action,
                f"UnrealCV {label} pose held",
                {"pose": self._pose_to_model(pose)},
            )
        if action == "turn_relative":
            pose = self._get_pose()
            pose[4] = _normalize_yaw(
                pose[4]
                + float(args.get("yaw_deg", self.action_config.yaw_step_deg))
            )
            self._set_pose(pose)
            return ActionResult(
                True,
                action,
                f"UnrealCV {label} yaw updated",
                {"pose": self._pose_to_model(pose)},
            )
        if action == "set_pose":
            pose = [
                float(args["x"]),
                float(args["y"]),
                float(args["z"]),
                float(args.get("pitch_deg", 0.0)),
                float(args.get("yaw_deg", 0.0)),
                float(args.get("roll_deg", 0.0)),
            ]
            self._set_pose(pose)
            return ActionResult(
                True,
                action,
                f"UnrealCV {label} pose set",
                {"pose": self._pose_to_model(pose)},
            )
        if action == "move_body":
            pose = self._get_pose()
            duration = float(
                args.get("duration", self.action_config.move_duration_sec)
            )
            vx, vy, vz = self._move_velocity(args)
            delta = body_velocity_to_world_delta(vx, vy, vz, duration, pose[4])
            pose[0] += delta["dx"]
            pose[1] += delta["dy"]
            pose[2] += delta["dz"]
            self._set_pose(pose)
            return ActionResult(
                True,
                action,
                f"UnrealCV {label} location updated",
                {"pose": self._pose_to_model(pose)},
            )
        if action == "ascend":
            pose = self._get_pose()
            speed_mps, duration = self._vertical_args(args)
            pose[2] += speed_mps * duration
            self._set_pose(pose)
            return ActionResult(
                True,
                action,
                f"UnrealCV {label} ascended",
                {"pose": self._pose_to_model(pose)},
            )
        if action == "descend":
            pose = self._get_pose()
            speed_mps, duration = self._vertical_args(args)
            pose[2] -= speed_mps * duration
            self._set_pose(pose)
            return ActionResult(
                True,
                action,
                f"UnrealCV {label} descended",
                {"pose": self._pose_to_model(pose)},
            )
        if action == "get_pose":
            return ActionResult(
                True,
                action,
                f"UnrealCV {label} pose",
                {"pose": self._model_pose()},
            )
        if action == "get_frames":
            return ActionResult(
                True,
                action,
                "UnrealCV camera frames",
                {
                    "frames": self._get_frames(
                        list(args.get("modalities", ["rgb"])),
                        args.get("capture_sources"),
                    )
                },
            )
        raise ValueError(f"Unsupported UnrealCV action: {action}")

    def _connect(self) -> None:
        """Open the client lazily and initialize the configured strategy once."""

        if not self.use_real_client:
            return
        if self.client is not None:
            return

        package = self._require_package()
        from unrealcv import Client

        client = Client((self.host, self.port))
        try:
            client.connect()
            if not client.isconnected():
                raise ConnectionError(
                    f"Could not connect to UnrealCV at {self.host}:{self.port}"
                )
            control = create_pose_control(client, package.unrealcv)
            control.prepare()
            actor = dict(package.unrealcv.get("actor", {}))
            units_per_meter = (
                float(actor.get("units_per_meter", 100.0))
                if package.unrealcv.get("control_target") == "actor"
                else 1.0
            )
            capture = UnrealCvCapture(
                client,
                camera_id=str(package.unrealcv.get("camera_id", 0)),
                units_per_meter=units_per_meter,
                artifact_dir=self.artifact_dir,
                save_artifacts=self.save_artifacts,
                sync_camera=control.sync_camera,
            )
        except Exception:
            client.disconnect()
            raise
        self.client = client
        self.control = control
        self.capture = capture

    def _get_pose(self) -> list[float]:
        if not self.use_real_client:
            return list(self.state.pose)
        self._connect()
        return list(self._require_control().get_pose())

    def _set_pose(self, pose: list[float]) -> None:
        self.state.pose = list(pose)
        if not self.use_real_client:
            return
        self._connect()
        self._require_control().set_pose(pose)

    def _model_pose(self) -> dict[str, float | str]:
        return self._pose_to_model(self._get_pose())

    def _pose_to_model(self, pose: list[float]) -> dict[str, float | str]:
        coordinate_frame = (
            self.control.coordinate_frame
            if self.control is not None
            else (
                "unrealcv_world_m"
                if self._control_target() == "actor"
                else "unrealcv_camera_native"
            )
        )
        return {
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "pitch_deg": float(pose[3]),
            "yaw_deg": float(pose[4]),
            "roll_deg": float(pose[5]),
            "coordinate_frame": coordinate_frame,
        }

    def _get_frames(
        self,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self.use_real_client:
            return {
                modality: {"artifact_path": None, "mock": True}
                for modality in modalities
            }
        self._connect()
        capture = self.capture
        if capture is None:
            raise RuntimeError("UnrealCV capture was not initialized")
        return capture.capture(modalities, capture_sources)

    def _move_velocity(self, args: dict) -> tuple[float, float, float]:
        vx, vy, vz, _duration = move_body_velocity_from_args(
            args,
            self.action_config,
            vertical_sign=UNREALCV_VERTICAL_SIGN,
        )
        return vx, vy, vz

    def _vertical_args(self, args: dict) -> tuple[float, float]:
        speed_mps = abs(
            float(args.get("speed_mps", self.action_config.vertical_velocity_mps))
        )
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

    def _control_target(self) -> str:
        if self.package is None:
            return "camera"
        return str(self.package.unrealcv.get("control_target", "camera")).lower()

    def _control_label(self) -> str:
        return "actor" if self._control_target() == "actor" else "camera"

    def _require_control(self) -> UnrealCvPoseControl:
        if self.control is None:
            raise RuntimeError("UnrealCV pose control was not initialized")
        return self.control

    def _require_package(self) -> MapPackage:
        if self.package is None:
            raise RuntimeError("UnrealCvBackend has not been started")
        return self.package


def _normalize_yaw(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized
