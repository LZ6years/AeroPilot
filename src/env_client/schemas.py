"""Define shared request and response schemas for environment C/S calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class PortPolicy:
    control_port_start: int = 30001
    max_scene_num: int = 8


@dataclass(frozen=True)
class MapPackage:
    map_id: str
    provider: str
    scene_id: str
    root: str | None
    launch: dict[str, Any]
    airsim: dict[str, Any]
    control: dict[str, Any] = field(default_factory=dict)
    unrealcv: dict[str, Any] = field(default_factory=dict)
    connection: dict[str, Any] = field(default_factory=dict)
    ssh: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapPackage":
        # Validate and normalize a map configuration dictionary.
        missing = [key for key in ("map_id", "provider", "scene_id") if key not in data]
        if missing:
            raise ValueError(f"MapPackage missing required fields: {', '.join(missing)}")
        provider = str(data["provider"])
        airsim = dict(data.get("airsim", {}))
        unrealcv = dict(data.get("unrealcv", {}))
        control = dict(data.get("control", {}))
        if not control.get("protocol"):
            raise ValueError(
                f"MapPackage {data['map_id']} missing control.protocol"
            )
        protocol = str(control["protocol"]).lower()
        control["protocol"] = protocol
        if protocol == "airsim":
            for key in ("vehicle_name", "camera_name"):
                if key not in airsim:
                    raise ValueError(f"MapPackage {data['map_id']} missing airsim.{key}")
            raw_mode = str(airsim.get("mode", "Multirotor")).strip().lower()
            mode_aliases = {
                "computervision": "ComputerVision",
                "computer_vision": "ComputerVision",
                "cv": "ComputerVision",
                "multirotor": "Multirotor",
            }
            try:
                mode = mode_aliases[raw_mode]
            except KeyError as exc:
                raise ValueError(
                    f"MapPackage {data['map_id']} has unsupported "
                    f"airsim.mode: {airsim.get('mode')}"
                ) from exc
            airsim["mode"] = mode

            # AirSim's simulator mode and FlyAgent's control semantics are
            # related but intentionally explicit. Missing values adopt the new
            # two-strategy contract, so legacy Multirotor packages migrate to
            # paused physical stepping instead of the removed live controller.
            inferred_strategy = (
                "pose_kinematic"
                if mode == "ComputerVision"
                else "paused_multirotor"
            )
            strategy = str(
                airsim.get("control_strategy", inferred_strategy)
            ).strip().lower()
            expected_mode = {
                "pose_kinematic": "ComputerVision",
                "paused_multirotor": "Multirotor",
            }
            if strategy not in expected_mode:
                raise ValueError(
                    f"MapPackage {data['map_id']} has unsupported "
                    f"airsim.control_strategy: {strategy}"
                )
            if mode != expected_mode[strategy]:
                raise ValueError(
                    f"MapPackage {data['map_id']} requires airsim.mode="
                    f"{expected_mode[strategy]} when "
                    f"airsim.control_strategy={strategy}; got {mode}"
                )
            airsim["control_strategy"] = strategy
            source = str(airsim.get("settings_template_source", "local"))
            if source not in {"local", "remote"}:
                raise ValueError(
                    f"MapPackage {data['map_id']} has unsupported "
                    f"airsim.settings_template_source: {source}"
                )
        if protocol == "unrealcv":
            required = ("port", "camera_id")
            for key in required:
                if key not in unrealcv:
                    raise ValueError(f"MapPackage {data['map_id']} missing unrealcv.{key}")
            control_target = str(
                unrealcv.get("control_target", "camera")
            ).strip().lower()
            if control_target not in {"camera", "actor"}:
                raise ValueError(
                    f"MapPackage {data['map_id']} has unsupported "
                    f"unrealcv.control_target: {control_target}"
                )
            unrealcv["control_target"] = control_target
            if control_target == "actor":
                actor = unrealcv.get("actor")
                if not isinstance(actor, dict):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires unrealcv.actor "
                        "when control_target=actor"
                    )
                actor = dict(actor)
                name = str(actor.get("name", "")).strip()
                if not name:
                    raise ValueError(
                        f"MapPackage {data['map_id']} missing unrealcv.actor.name"
                    )
                actor["name"] = name
                startup = unrealcv.get("startup", {})
                if not isinstance(startup, dict):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires "
                        "unrealcv.startup to be an object"
                    )
                startup = dict(startup)
                hide_objects = startup.get("hide_objects", [])
                if not isinstance(hide_objects, (list, tuple)):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires "
                        "unrealcv.startup.hide_objects to be a list"
                    )
                normalized_hide_objects = [
                    str(value).strip() for value in hide_objects
                ]
                if any(not value for value in normalized_hide_objects):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires non-empty "
                        "unrealcv.startup.hide_objects names"
                    )
                if len(set(normalized_hide_objects)) != len(
                    normalized_hide_objects
                ):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires unique "
                        "unrealcv.startup.hide_objects names"
                    )
                # Hiding an Actor only disables its rendered mesh in UnrealCV;
                # it does not delete the Actor or remove pose control. Allowing
                # the controlled Actor here is required by egocentric vehicle
                # deployments that must keep self geometry out of observations.
                startup["hide_objects"] = normalized_hide_objects
                unrealcv["startup"] = startup
                scale = actor.get("scale")
                if scale is not None:
                    if not isinstance(scale, (list, tuple)) or len(scale) != 3:
                        raise ValueError(
                            f"MapPackage {data['map_id']} requires "
                            "unrealcv.actor.scale to contain three values"
                        )
                    normalized_scale = [float(value) for value in scale]
                    if any(
                        not math.isfinite(value) or value <= 0.0
                        for value in normalized_scale
                    ):
                        raise ValueError(
                            f"MapPackage {data['map_id']} requires positive "
                            "finite unrealcv.actor.scale values"
                        )
                    actor["scale"] = normalized_scale
                units_per_meter = float(actor.get("units_per_meter", 100.0))
                if not math.isfinite(units_per_meter) or units_per_meter <= 0.0:
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires a positive finite "
                        "unrealcv.actor.units_per_meter"
                    )
                actor["units_per_meter"] = units_per_meter
                yaw_offset_deg = float(actor.get("yaw_offset_deg", 0.0))
                if not math.isfinite(yaw_offset_deg):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires a finite "
                        "unrealcv.actor.yaw_offset_deg"
                    )
                actor["yaw_offset_deg"] = yaw_offset_deg
                if "landed_origin_z_m" in actor:
                    landed_origin_z_m = float(actor["landed_origin_z_m"])
                    if not math.isfinite(landed_origin_z_m):
                        raise ValueError(
                            f"MapPackage {data['map_id']} requires a finite "
                            "unrealcv.actor.landed_origin_z_m"
                        )
                    actor["landed_origin_z_m"] = landed_origin_z_m
                rotation_method = str(
                    actor.get("rotation_method", "standard_pose")
                ).strip().lower()
                if rotation_method not in {"standard_pose", "blueprint_yaw"}:
                    raise ValueError(
                        f"MapPackage {data['map_id']} has unsupported "
                        "unrealcv.actor.rotation_method: "
                        f"{rotation_method}"
                    )
                actor["rotation_method"] = rotation_method
                camera_follow = str(
                    actor.get("camera_follow", "none")
                ).strip().lower()
                if camera_follow not in {
                    "none",
                    "standard_pose",
                    "blueprint_pose",
                }:
                    raise ValueError(
                        f"MapPackage {data['map_id']} has unsupported "
                        "unrealcv.actor.camera_follow: "
                        f"{camera_follow}"
                    )
                actor["camera_follow"] = camera_follow
                activate_viewport = actor.get("activate_viewport", False)
                if not isinstance(activate_viewport, bool):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires "
                        "unrealcv.actor.activate_viewport to be boolean"
                    )
                actor["activate_viewport"] = activate_viewport
                camera_validation = actor.get("camera_validation", {})
                if not isinstance(camera_validation, dict):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires "
                        "unrealcv.actor.camera_validation to be an object"
                    )
                camera_validation = dict(camera_validation)
                validation_enabled = camera_validation.get("enabled", False)
                if not isinstance(validation_enabled, bool):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires "
                        "unrealcv.actor.camera_validation.enabled to be boolean"
                    )
                camera_validation["enabled"] = validation_enabled
                position_mode = str(
                    camera_validation.get(
                        "position_mode",
                        "expected_body_offset",
                    )
                ).strip().lower()
                if position_mode not in {
                    "expected_body_offset",
                    "max_distance",
                }:
                    raise ValueError(
                        f"MapPackage {data['map_id']} has unsupported "
                        "unrealcv.actor.camera_validation.position_mode: "
                        f"{position_mode}"
                    )
                camera_validation["position_mode"] = position_mode
                expected_offset = camera_validation.get(
                    "expected_body_offset_m",
                    [0.0, 0.0, 0.0],
                )
                if (
                    not isinstance(expected_offset, (list, tuple))
                    or len(expected_offset) != 3
                ):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires "
                        "unrealcv.actor.camera_validation."
                        "expected_body_offset_m to contain three values"
                    )
                normalized_offset = [
                    float(value) for value in expected_offset
                ]
                if any(
                    not math.isfinite(value)
                    for value in normalized_offset
                ):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires finite "
                        "unrealcv.actor.camera_validation."
                        "expected_body_offset_m values"
                    )
                camera_validation[
                    "expected_body_offset_m"
                ] = normalized_offset
                for key, default in (
                    ("position_tolerance_m", 0.05),
                    ("max_distance_m", 1.5),
                    ("yaw_tolerance_deg", 1.0),
                ):
                    value = float(camera_validation.get(key, default))
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(
                            f"MapPackage {data['map_id']} requires a positive "
                            "finite unrealcv.actor.camera_validation."
                            f"{key}"
                        )
                    camera_validation[key] = value
                actor["camera_validation"] = camera_validation
                disable_physics = actor.get("disable_physics", False)
                if not isinstance(disable_physics, bool):
                    raise ValueError(
                        f"MapPackage {data['map_id']} requires "
                        "unrealcv.actor.disable_physics to be boolean"
                    )
                actor["disable_physics"] = disable_physics
                for key, default in (
                    ("yaw_settle_timeout_sec", 5.0),
                    ("yaw_tolerance_deg", 1.0),
                    ("yaw_poll_interval_sec", 0.05),
                ):
                    value = float(actor.get(key, default))
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError(
                            f"MapPackage {data['map_id']} requires a positive "
                            f"finite unrealcv.actor.{key}"
                        )
                    actor[key] = value
                unrealcv["actor"] = actor
        raw_launch = data.get("launch", {})
        if not isinstance(raw_launch, dict):
            raise ValueError(
                f"MapPackage {data['map_id']} requires launch to be an object"
            )
        launch = dict(raw_launch)
        existing_process_policy = str(
            launch.get("existing_process_policy", "attach")
        ).strip().lower()
        if existing_process_policy not in {"attach", "reject"}:
            raise ValueError(
                f"MapPackage {data['map_id']} has unsupported "
                "launch.existing_process_policy: "
                f"{existing_process_policy}"
            )
        # Attaching remains the compatibility default. Benchmarks that require
        # a provably fresh map can opt into rejection without changing the
        # public EnvClient method surface.
        launch["existing_process_policy"] = existing_process_policy
        connection = dict(data.get("connection", {}))
        if not connection.get("mode"):
            raise ValueError(
                f"MapPackage {data['map_id']} missing connection.mode"
            )
        connection_mode = str(connection["mode"]).lower()
        connection["mode"] = connection_mode
        if connection_mode not in {"local", "remote"}:
            raise ValueError(
                f"MapPackage {data['map_id']} has unsupported connection.mode: "
                f"{connection_mode}"
            )
        connection.setdefault(
            "data_transport",
            "ssh_tunnel" if connection_mode == "remote" else "direct",
        )
        ssh = dict(data.get("ssh", {}))
        if connection_mode == "remote":
            for key in ("host", "user", "key_path"):
                if not ssh.get(key):
                    raise ValueError(f"MapPackage {data['map_id']} missing ssh.{key}")
            if connection.get("data_transport") != "ssh_tunnel":
                raise ValueError(
                    f"MapPackage {data['map_id']} remote connection requires "
                    "connection.data_transport=ssh_tunnel"
                )
            managed_launch = bool(launch.get("linux_executable")) and not bool(
                launch.get("manual", False)
            )
            if (
                managed_launch
                and protocol == "airsim"
                and not airsim.get("settings_template")
            ):
                raise ValueError(
                    f"MapPackage {data['map_id']} missing airsim.settings_template"
                )
        elif (
            protocol == "airsim"
            and airsim.get("settings_template_source") == "remote"
        ):
            raise ValueError(
                f"MapPackage {data['map_id']} local connection cannot use "
                "airsim.settings_template_source=remote"
            )
        root = data.get("root")
        if (
            connection_mode == "local"
            and not launch.get("manual", False)
            and not root
        ):
            raise ValueError(
                f"MapPackage {data['map_id']} requires root unless launch.manual=true"
            )
        if protocol not in {"airsim", "unrealcv"}:
            raise ValueError(f"MapPackage {data['map_id']} has unsupported control.protocol: {protocol}")
        return cls(
            map_id=str(data["map_id"]),
            provider=provider,
            scene_id=str(data["scene_id"]),
            root=str(root) if root is not None else None,
            launch=dict(launch),
            airsim=dict(airsim),
            control=control,
            unrealcv=dict(unrealcv),
            connection=connection,
            ssh=ssh,
        )

    def to_dict(self) -> dict[str, Any]:
        # Convert the package into JSON-serializable data.
        return asdict(self)

    @property
    def control_protocol(self) -> str:
        # Return the runtime control protocol used by the simulation backend.
        return str(self.control["protocol"])

    @property
    def connection_mode(self) -> str:
        """Return how the scene process is reached and managed."""

        return str(self.connection["mode"])


@dataclass(frozen=True)
class SceneCapabilities:
    """Describe simulator behavior that Runtime may safely rely on."""

    protocol: str
    control_mode: str
    supported_actions: tuple[str, ...]
    observation_modalities: tuple[str, ...]
    requires_takeoff: bool
    supports_soft_reset: bool

    def supports(self, action: str) -> bool:
        """Return whether the backend advertises one primitive action."""

        return action in self.supported_actions


@dataclass
class SceneInstance:
    scene_instance_id: str
    map_id: str
    scene_id: str
    machine_id: str
    control_host: str
    control_port: int
    settings_path: str | None
    process_id: int | None
    status: str
    process_ownership: str = "local"
    remote_process_id: int | None = None
    settings_sha256: str | None = None
    capabilities: SceneCapabilities | None = None

    def to_dict(self) -> dict[str, Any]:
        # Convert the scene instance into JSON-serializable data.
        return asdict(self)


@dataclass(frozen=True)
class ActionRequest:
    scene_instance_id: str
    agent_id: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    action: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Convert the action result into JSON-serializable data.
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    scene_instance_id: str
    agent_id: str
    modalities: list[str]
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        # Convert the observation into JSON-serializable data.
        return asdict(self)
