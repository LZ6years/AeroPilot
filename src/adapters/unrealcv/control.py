"""Provide camera and actor pose control for the UnrealCV adapter package.

The adapter selects a strategy from resolved protocol configuration. Camera control uses
standard UnrealCV camera commands. Actor control uses standard object
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
    startup_hide_objects: tuple[str, ...]
    scale: tuple[float, float, float] | None
    units_per_meter: float
    yaw_offset_deg: float
    rotation_method: str
    camera_follow: str
    activate_viewport: bool
    camera_validation_enabled: bool
    camera_position_mode: str
    camera_expected_body_offset_m: tuple[float, float, float]
    camera_position_tolerance_m: float
    camera_max_distance_m: float
    camera_yaw_tolerance_deg: float
    disable_physics: bool
    yaw_settle_timeout_sec: float
    yaw_tolerance_deg: float
    yaw_poll_interval_sec: float

    @classmethod
    def from_unrealcv(cls, unrealcv: dict[str, Any]) -> "ActorControlConfig":
        """Build the typed configuration after MapPackage schema validation."""

        actor = dict(unrealcv.get("actor", {}))
        startup = dict(unrealcv.get("startup", {}))
        camera_validation = dict(actor.get("camera_validation", {}))
        configured_scale = actor.get("scale")
        return cls(
            name=str(actor["name"]),
            camera_id=str(unrealcv.get("camera_id", 0)),
            startup_hide_objects=tuple(
                str(value) for value in startup.get("hide_objects", ())
            ),
            scale=(
                tuple(float(value) for value in configured_scale)
                if configured_scale is not None
                else None
            ),
            units_per_meter=float(actor.get("units_per_meter", 100.0)),
            yaw_offset_deg=float(actor.get("yaw_offset_deg", 0.0)),
            rotation_method=str(actor.get("rotation_method", "standard_pose")),
            camera_follow=str(actor.get("camera_follow", "none")),
            activate_viewport=bool(actor.get("activate_viewport", False)),
            camera_validation_enabled=bool(
                camera_validation.get("enabled", False)
            ),
            camera_position_mode=str(
                camera_validation.get(
                    "position_mode",
                    "expected_body_offset",
                )
            ),
            camera_expected_body_offset_m=tuple(
                float(value)
                for value in camera_validation.get(
                    "expected_body_offset_m",
                    (0.0, 0.0, 0.0),
                )
            ),
            camera_position_tolerance_m=float(
                camera_validation.get("position_tolerance_m", 0.05)
            ),
            camera_max_distance_m=float(
                camera_validation.get("max_distance_m", 1.5)
            ),
            camera_yaw_tolerance_deg=float(
                camera_validation.get("yaw_tolerance_deg", 1.0)
            ),
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
        self._hide_startup_objects()
        if self.config.scale is not None:
            scale = self.config.scale
            command = (
                f"vset /object/{self.config.name}/scale "
                f"{scale[0]} {scale[1]} {scale[2]}"
            )
            _check_command_responses(
                [command], [self.client.request(command)]
            )
        if self.config.disable_physics:
            command = f"vbp {self.config.name} set_phy 0"
            _check_command_responses(
                [command], [self.client.request(command)]
            )
        if self.config.activate_viewport:
            self._activate_viewport()
        # The packaged map's default Actor pose may place the camera against
        # arbitrary geometry before an episode initial pose is applied. Sync
        # now so the scene is normalized, but validate only once Runtime calls
        # episode set_pose (and again before every capture).
        self._sync_camera(validate=False)

    def _activate_viewport(self) -> None:
        """Bind the visible packaged-game viewport to the controlled Actor.

        Camera synchronization and viewport ownership are separate in
        UnrealCV.  In particular, ``standard_pose`` can correctly feed camera
        0 to perception while the game window remains attached to a default
        character Pawn.  Maps opt into this Blueprint command explicitly so
        generic deployments without a compatible Actor remain unchanged.
        """

        command = f"vbp {self.config.name} set_viewport"
        _check_command_responses(
            [command], [self.client.request(command)]
        )

    def _hide_startup_objects(self) -> None:
        """Hide explicitly declared packaged-map meshes before capture.

        Some packaged environments ship sample pawns whose meshes can leak
        into a reusable camera. The MapPackage must name them explicitly;
        control code never guesses from a benchmark or map identifier.
        Missing objects are treated as already normalized so reconnecting to an
        attached scene remains idempotent. Hiding preserves camera components
        and stable camera ids, unlike destroying a camera-owning pawn.
        """

        if not self.config.startup_hide_objects:
            return
        inventory_command = "vget /objects"
        before_text = _response_text(self.client.request(inventory_command))
        _raise_for_error(inventory_command, before_text)
        present = set(before_text.split())
        commands = [
            f"vset /object/{name}/hide"
            for name in self.config.startup_hide_objects
            if name in present
        ]
        if commands:
            _check_command_responses(
                commands,
                [self.client.request(command) for command in commands],
            )

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
        canonical_yaw = (
            native_yaw_deg
            if self.config.rotation_method == "blueprint_yaw"
            else native_yaw_deg - self.config.yaw_offset_deg
        )
        return [
            native_location[0] / scale,
            native_location[1] / scale,
            native_location[2] / scale,
            pitch_deg,
            _normalize_yaw(canonical_yaw),
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
            # DowntownWest's Blueprint applies its mesh-facing offset inside
            # ``set_rotation``. The command therefore receives the configured
            # native yaw, matching the original UAV-Flow control path. Once
            # interpolation finishes, the queried Actor yaw is the physical
            # camera/world yaw and ``get_pose`` exposes it as canonical yaw.
            rotation_command = (
                f"vbp {self.config.name} set_rotation "
                f"{native_yaw}"
            )
        commands = [location_command, rotation_command]
        _check_command_responses(commands, _request_batch(self.client, commands))
        if self.config.rotation_method == "blueprint_yaw":
            self._wait_for_yaw(_normalize_yaw(pose[4]))
        if self.config.camera_validation_enabled:
            self._validate_actor_pose(pose)
        # Runtime preflight takeoff also uses set_pose. Validate on capture,
        # after the episode initial pose has replaced that scene-level pose.
        self._sync_camera(validate=False)

    def _validate_actor_pose(self, expected_pose: list[float]) -> None:
        """Reject acknowledged Actor writes that did not reach the requested pose."""

        observed_pose = self.get_pose()
        position_error_m = math.sqrt(
            sum(
                (observed_pose[index] - expected_pose[index]) ** 2
                for index in range(3)
            )
        )
        yaw_error_deg = abs(_yaw_delta(observed_pose[4], expected_pose[4]))
        if (
            position_error_m <= self.config.camera_position_tolerance_m
            and yaw_error_deg <= self.config.camera_yaw_tolerance_deg
        ):
            return
        raise RuntimeError(
            "UnrealCV actor pose validation failed after acknowledged write: "
            f"expected_xyz_m={tuple(float(value) for value in expected_pose[:3])}, "
            f"observed_xyz_m={tuple(float(value) for value in observed_pose[:3])}, "
            f"position_error_m={position_error_m:.6f}, "
            "position_tolerance_m="
            f"{self.config.camera_position_tolerance_m:.6f}, "
            f"expected_yaw_deg={float(expected_pose[4]):.3f}, "
            f"observed_yaw_deg={float(observed_pose[4]):.3f}, "
            f"yaw_error_deg={yaw_error_deg:.3f}, "
            f"yaw_tolerance_deg={self.config.camera_yaw_tolerance_deg:.3f}"
        )

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
        # Capture is the observable boundary that matters to callers. Actor
        # pose writes sync without validation because scene-level takeoff also
        # uses set_pose before Runtime applies the episode initial pose.
        self._sync_camera(validate=True)

    def _sync_camera(self, *, validate: bool) -> None:
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
            if validate:
                self._validate_camera_alignment(location, rotation)
            return
        if self.config.camera_follow == "blueprint_pose":
            # The historical UAV-Flow batch helper copies the controlled
            # actor's absolute pose into the Blueprint camera on every update.
            # ``vbp`` is addressed by the camera-owning actor; the numeric
            # capture camera id is not an Actor id.
            commands = [
                (
                    f"vbp {self.config.name} set_cam "
                    f"{location[0]} {location[1]} {location[2]} "
                    f"{roll_deg} {pitch_deg} {yaw_deg}"
                ),
            ]
            # Preserve the historical blueprint_pose contract for maps that
            # have not opted into one-time viewport activation during prepare.
            if not self.config.activate_viewport:
                commands.append(f"vbp {self.config.name} set_viewport")
            _check_command_responses(
                commands, [self.client.request(command) for command in commands]
            )
            if validate:
                self._validate_camera_alignment(location, rotation)
            return
        raise ValueError(
            f"Unsupported actor camera_follow strategy: {self.config.camera_follow}"
        )

    def _validate_camera_alignment(
        self,
        actor_location: list[float],
        actor_rotation: list[float],
    ) -> None:
        """Reject camera commands that acknowledged but did not take effect.

        Some packaged Blueprints return ``ok`` for a syntactically accepted
        command without moving the capture camera.  The map-level validation
        contract therefore checks the observable camera pose rather than
        trusting the command response.
        """

        if not self.config.camera_validation_enabled:
            return
        location_command = (
            f"vget /camera/{self.config.camera_id}/location"
        )
        rotation_command = (
            f"vget /camera/{self.config.camera_id}/rotation"
        )
        responses = _request_batch(
            self.client, [location_command, rotation_command]
        )
        camera_location = _parse_float_list(
            responses[0], location_command
        )
        camera_rotation = _parse_float_list(
            responses[1], rotation_command
        )
        _require_length(camera_location, 3, "camera location")
        _require_length(camera_rotation, 3, "camera rotation")
        actor_yaw_deg = float(actor_rotation[1])
        camera_yaw_deg = float(camera_rotation[1])
        scale = self.config.units_per_meter
        world_dx_m = (
            camera_location[0] - actor_location[0]
        ) / scale
        world_dy_m = (
            camera_location[1] - actor_location[1]
        ) / scale
        world_dz_m = (
            camera_location[2] - actor_location[2]
        ) / scale
        yaw_rad = math.radians(actor_yaw_deg)
        observed_body_offset_m = (
            world_dx_m * math.cos(yaw_rad)
            + world_dy_m * math.sin(yaw_rad),
            -world_dx_m * math.sin(yaw_rad)
            + world_dy_m * math.cos(yaw_rad),
            world_dz_m,
        )
        expected_body_offset_m = (
            self.config.camera_expected_body_offset_m
        )
        offset_error_m = math.sqrt(
            sum(
                (
                    observed_body_offset_m[index]
                    - expected_body_offset_m[index]
                )
                ** 2
                for index in range(3)
            )
        )
        camera_distance_m = math.sqrt(
            sum(value**2 for value in observed_body_offset_m)
        )
        if self.config.camera_position_mode == "expected_body_offset":
            position_ok = (
                offset_error_m
                <= self.config.camera_position_tolerance_m
            )
        elif self.config.camera_position_mode == "max_distance":
            position_ok = (
                camera_distance_m
                <= self.config.camera_max_distance_m
            )
        else:  # Defensive guard for direct construction outside MapPackage.
            raise ValueError(
                "Unsupported camera validation position mode: "
                f"{self.config.camera_position_mode}"
            )
        yaw_error_deg = abs(_yaw_delta(camera_yaw_deg, actor_yaw_deg))
        if (
            position_ok
            and yaw_error_deg <= self.config.camera_yaw_tolerance_deg
        ):
            return
        raise RuntimeError(
            "UnrealCV camera follow validation failed: "
            f"strategy={self.config.camera_follow}, "
            f"actor={self.config.name}, camera_id={self.config.camera_id}, "
            f"actor_location={actor_location}, "
            f"camera_location={camera_location}, "
            f"position_mode={self.config.camera_position_mode}, "
            f"expected_body_offset_m={expected_body_offset_m}, "
            f"observed_body_offset_m={observed_body_offset_m}, "
            f"offset_error_m={offset_error_m:.6f}, "
            "position_tolerance_m="
            f"{self.config.camera_position_tolerance_m:.6f}, "
            f"camera_distance_m={camera_distance_m:.6f}, "
            f"max_distance_m={self.config.camera_max_distance_m:.6f}, "
            f"actor_yaw_deg={actor_yaw_deg:.6f}, "
            f"camera_yaw_deg={camera_yaw_deg:.6f}, "
            f"yaw_error_deg={yaw_error_deg:.6f}, "
            f"yaw_tolerance_deg={self.config.camera_yaw_tolerance_deg:.6f}"
        )


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
