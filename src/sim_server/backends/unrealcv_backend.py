"""Implement a minimal UnrealCV backend for Windows UnrealZoo maps."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

from src.env_client.schemas import ActionResult, MapPackage, Observation, SceneInstance


@dataclass
class UnrealCvState:
    # Store the current soft camera pose for no-op and dry-run operations.
    pose: list[float]


class UnrealCvBackend:
    """Backend for gym_unrealcv / UnrealZoo Windows packaged maps."""

    def __init__(self, use_real_client: bool = False) -> None:
        # Keep real UnrealCV optional so unit tests do not need the package.
        self.use_real_client = use_real_client
        self.client: Any | None = None
        self.package: MapPackage | None = None
        self.state = UnrealCvState([0.0, 0.0, 300.0, 0.0, 0.0, 0.0])

    def start(self, package: MapPackage, scene: SceneInstance) -> SceneInstance:
        # Save package context and mark the scene as ready for UnrealCV calls.
        self.package = package
        scene.status = "running"
        return scene

    def stop(self, scene: SceneInstance) -> None:
        # Disconnect UnrealCV if a real client was opened.
        if self.client is not None:
            self.client.disconnect()
            self.client = None
        scene.status = "stopped"

    def get_observation(
        self, scene: SceneInstance, agent_id: str, modalities: list[str]
    ) -> Observation:
        # Fetch raw camera observations from UnrealCV or dry-run state.
        data: dict[str, Any] = {}
        if "pose" in modalities:
            data["pose"] = self._get_pose()
        frame_modalities = [m for m in modalities if m != "pose"]
        if frame_modalities:
            data["frames"] = self._get_frames(frame_modalities)
        return Observation(scene.scene_instance_id, agent_id, modalities, data)

    def execute_action(
        self, scene: SceneInstance, agent_id: str, action: str, args: dict
    ) -> ActionResult:
        # Map UAV-like smoke-test actions onto UnrealCV camera controls.
        if action == "connect":
            self._connect()
            return ActionResult(True, action, "UnrealCV connected")
        if action in {"takeoff", "hover", "land"}:
            return ActionResult(True, action, f"UnrealCV {action} treated as no-op")
        if action == "turn_relative":
            pose = self._get_pose()
            pose[4] += float(args["yaw_deg"])
            self._set_pose(pose)
            return ActionResult(True, action, "UnrealCV camera yaw updated", {"pose": pose})
        if action == "move_body":
            pose = self._get_pose()
            yaw_rad = math.radians(pose[4])
            duration = float(args.get("duration", 0.0))
            vx = float(args.get("vx", 0.0))
            vy = float(args.get("vy", 0.0))
            pose[0] += (vx * math.cos(yaw_rad) - vy * math.sin(yaw_rad)) * duration
            pose[1] += (vx * math.sin(yaw_rad) + vy * math.cos(yaw_rad)) * duration
            pose[2] += float(args.get("vz", 0.0)) * duration
            self._set_pose(pose)
            return ActionResult(True, action, "UnrealCV camera location updated", {"pose": pose})
        if action == "ascend":
            pose = self._get_pose()
            pose[2] += abs(float(args["meters"]))
            self._set_pose(pose)
            return ActionResult(True, action, "UnrealCV camera ascended", {"pose": pose})
        if action == "descend":
            pose = self._get_pose()
            pose[2] -= abs(float(args["meters"]))
            self._set_pose(pose)
            return ActionResult(True, action, "UnrealCV camera descended", {"pose": pose})
        if action == "get_pose":
            return ActionResult(True, action, "UnrealCV camera pose", {"pose": self._get_pose()})
        if action == "get_frames":
            return ActionResult(
                True,
                action,
                "UnrealCV camera frames",
                {"frames": self._get_frames(list(args.get("modalities", ["rgb"])))},
            )
        raise ValueError(f"Unsupported UnrealCV action: {action}")

    def _connect(self) -> None:
        # Open the UnrealCV client lazily.
        if self.client is not None:
            return
        if not self.use_real_client:
            return
        package = self._require_package()
        from unrealcv import Client

        self.client = Client((package.unrealcv.get("host", "127.0.0.1"), int(package.unrealcv["port"])))
        self.client.connect()
        if not self.client.isconnected():
            raise ConnectionError(f"Could not connect to UnrealCV at {package.unrealcv}")

    def _get_pose(self) -> list[float]:
        # Read current camera pose as [x, y, z, pitch, yaw, roll].
        if not self.use_real_client:
            return list(self.state.pose)
        self._connect()
        cam = self._camera_id()
        location = self._request_float_list(f"vget /camera/{cam}/location")
        rotation = self._request_float_list(f"vget /camera/{cam}/rotation")
        return [*location, *rotation]

    def _set_pose(self, pose: list[float]) -> None:
        # Set camera location and rotation in UnrealCV.
        self.state.pose = list(pose)
        if not self.use_real_client:
            return
        self._connect()
        cam = self._camera_id()
        self.client.request(f"vset /camera/{cam}/location {pose[0]} {pose[1]} {pose[2]}")
        self.client.request(f"vset /camera/{cam}/rotation {pose[3]} {pose[4]} {pose[5]}")

    def _request_float_list(self, command: str) -> list[float]:
        # Parse a numeric UnrealCV response and surface command errors clearly.
        response = self.client.request(command)
        text = str(response)
        if text.startswith("error"):
            raise RuntimeError(f"UnrealCV command failed: {command}: {text}")
        return [float(x) for x in text.split()]

    def _get_frames(self, modalities: list[str]) -> dict[str, Any]:
        # Fetch image bytes as base64 strings for JSON-safe recording.
        if not self.use_real_client:
            return {m: {"artifact": None, "mock": True} for m in modalities}
        self._connect()
        frames: dict[str, Any] = {}
        for modality in modalities:
            command = self._frame_command(modality)
            response = self.client.request(command)
            if isinstance(response, bytes):
                encoded = base64.b64encode(response).decode("ascii")
                frames[modality] = {"base64": encoded, "bytes": len(response), "command": command}
            else:
                frames[modality] = {"text": str(response), "command": command}
        return frames

    def _frame_command(self, modality: str) -> str:
        # Convert FlyAgent modalities into UnrealCV image commands.
        cam = self._camera_id()
        normalized = modality.lower()
        if normalized in {"rgb", "scene", "color"}:
            return f"vget /camera/{cam}/lit png"
        if normalized in {"depth", "depth_vis"}:
            return f"vget /camera/{cam}/depth npy"
        if normalized in {"mask", "segmentation"}:
            return f"vget /camera/{cam}/object_mask png"
        return f"vget /camera/{cam}/lit png"

    def _camera_id(self) -> str:
        # Return configured UnrealCV camera id as a string.
        return str(self._require_package().unrealcv.get("camera_id", 0))

    def _require_package(self) -> MapPackage:
        # Ensure backend has been started with a MapPackage.
        if self.package is None:
            raise RuntimeError("UnrealCvBackend has not been started")
        return self.package
