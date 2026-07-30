"""Provide simulator-agnostic camera and actor pose control for UnrealCV.

The Backend selects one strategy from the MapPackage.  Camera control uses
standard UnrealCV camera commands.  Actor control uses standard object
location/rotation commands unless a packaged map explicitly declares the
Blueprint yaw extension used by its controllable vehicle.

All public strategy poses use FlyAgent's canonical shape:
``[x_m, y_m, z_m, pitch_deg, yaw_deg, roll_deg]``.  Native Unreal units,
rotation ordering, and map-specific yaw offsets remain inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Protocol


class UnrealCvPoseControl(Protocol):
    """Structural interface implemented by each UnrealCV control strategy."""

    control_mode: str
    coordinate_frame: str

    def prepare(self) -> None:
        """Validate the target and apply explicitly requested startup state."""

    def get_pose(self) -> list[float]:
        """Return canonical position in meters and Euler angles in degrees."""

    def set_pose(self, pose: list[float]) -> None:
        """Apply one canonical pose to the configured target."""

    def sync_camera(self) -> None:
        """Align the observation camera when the selected strategy requires it."""


@dataclass(frozen=True)
class ActorControlConfig:
    """Validated actor-specific controls from ``MapPackage.unrealcv``."""

    name: str
    camera_id: str
    units_per_meter: float
    yaw_offset_deg: float
    rotation_method: str
    camera_follow: str
    disable_physics: bool
    yaw_settle_timeout_sec: float
    yaw_tolerance_deg: float
    yaw_poll_interval_sec: float

    @classmethod
    def from_unrealcv(cls, unrealcv: dict[str, Any]) -> "ActorControlConfig":
        """Build the typed configuration after MapPackage schema validation."""

        actor = dict(unrealcv.get("actor", {}))
        return cls(
            name=str(actor["name"]),
            camera_id=str(unrealcv.get("camera_id", 0)),
            units_per_meter=float(actor.get("units_per_meter", 100.0)),
            yaw_offset_deg=float(actor.get("yaw_offset_deg", 0.0)),
            rotation_method=str(actor.get("rotation_method", "standard_pose")),
            camera_follow=str(actor.get("camera_follow", "none")),
            disable_physics=bool(actor.get("disable_physics", False)),
            yaw_settle_timeout_sec=float(
                actor.get("yaw_settle_timeout_sec", 5.0)
            ),
            yaw_tolerance_deg=float(actor.get("yaw_tolerance_deg", 1.0)),
            yaw_poll_interval_sec=float(
                actor.get("yaw_poll_interval_sec", 0.05)
            ),
        )


class CameraPoseControl:
    """Control a free UnrealCV camera with the historical coordinate behavior.

    Camera mode deliberately keeps native location values unchanged.  Existing
    camera-only MapPackages historically treated those values as canonical
    coordinates, so changing their scale here would silently break them.
    Actor mode is the preferred choice when the map exposes a UAV actor and
    needs an explicit units-per-meter conversion.
    """

    control_mode = "camera_pose"
    coordinate_frame = "unrealcv_camera_native"

    def __init__(self, client: Any, camera_id: str) -> None:
        self.client = client
        self.camera_id = str(camera_id)

    def prepare(self) -> None:
        # A camera endpoint has already been proven healthy by the Backend.
        self.get_pose()

    def get_pose(self) -> list[float]:
        responses = _request_batch(
            self.client,
            [
                f"vget /camera/{self.camera_id}/location",
                f"vget /camera/{self.camera_id}/rotation",
            ],
        )
        location = _parse_float_list(
            responses[0], f"vget /camera/{self.camera_id}/location"
        )
        rotation = _parse_float_list(
            responses[1], f"vget /camera/{self.camera_id}/rotation"
        )
        _require_length(location, 3, "camera location")
        _require_length(rotation, 3, "camera rotation")
        # Camera rotation is already [pitch, yaw, roll].
        return [*location, *rotation]

    def set_pose(self, pose: list[float]) -> None:
        _require_length(pose, 6, "canonical pose")
        commands = [
            (
                f"vset /camera/{self.camera_id}/location "
                f"{pose[0]} {pose[1]} {pose[2]}"
            ),
            (
                f"vset /camera/{self.camera_id}/rotation "
                f"{pose[3]} {pose[4]} {pose[5]}"
            ),
        ]
        _check_command_responses(commands, _request_batch(self.client, commands))

    def sync_camera(self) -> None:
        # The controlled target is already the observation camera.
        return


class ActorPoseControl:
    """Control one Unreal actor while exposing canonical meter-based poses."""

    control_mode = "actor_pose"
    coordinate_frame = "unrealcv_world_m"

    def __init__(self, client: Any, config: ActorControlConfig) -> None:
        self.client = client
        self.config = config

    def prepare(self) -> None:
        # Reading the actor first gives a specific missing-actor error instead
        # of silently falling back to camera motion.
        self.get_pose()
        if self.config.disable_physics:
            command = f"vbp {self.config.name} set_phy 0"
            _check_command_responses(
                [command], [self.client.request(command)]
            )
        self.sync_camera()

    def get_pose(self) -> list[float]:
        location_command = f"vget /object/{self.config.name}/location"
        rotation_command = f"vget /object/{self.config.name}/rotation"
        responses = _request_batch(
            self.client, [location_command, rotation_command]
        )
        native_location = _parse_float_list(responses[0], location_command)
        native_rotation = _parse_float_list(responses[1], rotation_command)
        _require_length(native_location, 3, "actor location")
        _require_length(native_rotation, 3, "actor rotation")

        # UnrealCv_API documents object rotation as [roll, yaw, pitch].
        roll_deg, native_yaw_deg, pitch_deg = native_rotation
        scale = self.config.units_per_meter
        return [
            native_location[0] / scale,
            native_location[1] / scale,
            native_location[2] / scale,
            pitch_deg,
            _normalize_yaw(native_yaw_deg - self.config.yaw_offset_deg),
            roll_deg,
        ]

    def set_pose(self, pose: list[float]) -> None:
        _require_length(pose, 6, "canonical pose")
        current = self.get_pose()
        if self.config.rotation_method == "blueprint_yaw":
            # The packaged Blueprint command only controls yaw.  Rejecting a
            # requested pitch/roll change prevents a false successful result.
            if not math.isclose(pose[3], current[3], abs_tol=1e-4) or not math.isclose(
                pose[5], current[5], abs_tol=1e-4
            ):
                raise ValueError(
                    "UnrealCV actor rotation_method=blueprint_yaw cannot change "
                    "pitch_deg or roll_deg"
                )

        scale = self.config.units_per_meter
        native_yaw = _normalize_yaw(pose[4] + self.config.yaw_offset_deg)
        location_command = (
            f"vset /object/{self.config.name}/location "
            f"{pose[0] * scale} {pose[1] * scale} {pose[2] * scale}"
        )
        if self.config.rotation_method == "standard_pose":
            rotation_command = (
                f"vset /object/{self.config.name}/rotation "
                f"{pose[3]} {native_yaw} {pose[5]}"
            )
        else:
            # This Blueprint extension accepts canonical/world yaw and
            # internally applies the Actor mesh offset. It also interpolates,
            # unlike standard UnrealCV object rotation.
            rotation_command = (
                f"vbp {self.config.name} set_rotation "
                f"{_normalize_yaw(pose[4])}"
            )
        commands = [location_command, rotation_command]
        _check_command_responses(commands, _request_batch(self.client, commands))
        if self.config.rotation_method == "blueprint_yaw":
            self._wait_for_yaw(_normalize_yaw(pose[4]))
        self.sync_camera()

    def _wait_for_yaw(self, target_yaw_deg: float) -> None:
        """Wait for a Blueprint-interpolated yaw instead of reporting early."""

        deadline = time.monotonic() + self.config.yaw_settle_timeout_sec
        last_yaw = self.get_pose()[4]
        while abs(_yaw_delta(last_yaw, target_yaw_deg)) > self.config.yaw_tolerance_deg:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "UnrealCV actor yaw did not settle before timeout: "
                    f"target={target_yaw_deg:.3f}, observed={last_yaw:.3f}, "
                    f"tolerance={self.config.yaw_tolerance_deg:.3f}"
                )
            time.sleep(self.config.yaw_poll_interval_sec)
            last_yaw = self.get_pose()[4]

    def sync_camera(self) -> None:
        if self.config.camera_follow == "none":
            return

        location_command = f"vget /object/{self.config.name}/location"
        rotation_command = f"vget /object/{self.config.name}/rotation"
        responses = _request_batch(
            self.client, [location_command, rotation_command]
        )
        location = _parse_float_list(responses[0], location_command)
        rotation = _parse_float_list(responses[1], rotation_command)
        _require_length(location, 3, "actor location")
        _require_length(rotation, 3, "actor rotation")
        roll_deg, yaw_deg, pitch_deg = rotation
        if self.config.camera_follow == "standard_pose":
            commands = [
                (
                    f"vset /camera/{self.config.camera_id}/location "
                    f"{location[0]} {location[1]} {location[2]}"
                ),
                (
                    f"vset /camera/{self.config.camera_id}/rotation "
                    f"{pitch_deg} {yaw_deg} {roll_deg}"
                ),
            ]
            _check_command_responses(
                commands, _request_batch(self.client, commands)
            )
            return
        command = (
            f"vbp {self.config.camera_id} set_cam "
            f"{location[0]} {location[1]} {location[2]} "
            f"{roll_deg} {pitch_deg} {yaw_deg}"
        )
        _check_command_responses([command], [self.client.request(command)])


def create_pose_control(
    client: Any,
    unrealcv: dict[str, Any],
) -> UnrealCvPoseControl:
    """Create the strategy selected by a validated UnrealCV MapPackage."""

    target = str(unrealcv.get("control_target", "camera")).lower()
    if target == "camera":
        return CameraPoseControl(client, str(unrealcv.get("camera_id", 0)))
    if target == "actor":
        return ActorPoseControl(client, ActorControlConfig.from_unrealcv(unrealcv))
    raise ValueError(f"Unsupported UnrealCV control_target: {target}")


def _request_batch(client: Any, commands: list[str]) -> list[Any]:
    """Use UnrealCV's real batch API and keep test clients structurally simple."""

    request_batch = getattr(client, "request_batch", None)
    if callable(request_batch):
        responses = request_batch(commands)
    else:
        responses = client.request(commands)
    if not isinstance(responses, (list, tuple)):
        raise RuntimeError(
            "UnrealCV batch request returned a non-list response for "
            f"{commands!r}: {responses!r}"
        )
    if len(responses) != len(commands):
        raise RuntimeError(
            "UnrealCV batch response count mismatch: "
            f"requested={len(commands)}, received={len(responses)}"
        )
    return list(responses)


def _parse_float_list(response: Any, command: str) -> list[float]:
    text = _response_text(response)
    _raise_for_error(command, text)
    try:
        return [float(value) for value in text.split()]
    except ValueError as exc:
        raise RuntimeError(
            f"UnrealCV command returned non-numeric data: {command}: {text!r}"
        ) from exc


def _check_command_responses(commands: list[str], responses: list[Any]) -> None:
    if len(commands) != len(responses):
        raise RuntimeError(
            "UnrealCV command response count mismatch: "
            f"commands={len(commands)}, responses={len(responses)}"
        )
    for command, response in zip(commands, responses):
        _raise_for_error(command, _response_text(response))


def _response_text(response: Any) -> str:
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace").strip()
    return str(response).strip()


def _raise_for_error(command: str, text: str) -> None:
    if text.lower().startswith("error"):
        raise RuntimeError(f"UnrealCV command failed: {command}: {text}")


def _require_length(values: list[float], expected: int, label: str) -> None:
    if len(values) != expected:
        raise RuntimeError(
            f"UnrealCV {label} returned {len(values)} values; expected {expected}"
        )


def _normalize_yaw(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized


def _yaw_delta(observed: float, target: float) -> float:
    """Return the shortest signed observed-to-target yaw difference."""

    return _normalize_yaw(float(target) - float(observed))
