"""Document the AirNavigation primitive surface without importing it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AirNavigationPrimitiveSurface:
    """Reference-only list of AirNavigation primitive capabilities.

    AirNavigation is not imported here on purpose. Its current controller pulls
    in project-specific map and UI dependencies, while FlyAgent only wants the
    primitive capability surface as a compatibility reference.
    """

    connection: tuple[str, ...] = ("connect", "disconnect")
    lifecycle: tuple[str, ...] = ("take_off", "hover", "land")
    motion: tuple[str, ...] = (
        "move_by_velocity_face_direction",
        "move_by_velocity_with_same_direction",
        "move_by_velocity_new",
        "fly_to_position",
    )
    camera: tuple[str, ...] = ("set_camera_rotate", "get_camera_rotation", "rotate_camera")
    telemetry: tuple[str, ...] = (
        "get_body_position",
        "get_body_eularian_angle",
        "get_velocity",
        "get_angular_velocity",
    )
    observation: tuple[str, ...] = ("get_origin_frame", "set_capture_type", "get_all_frame")


AIRNAVIGATION_PRIMITIVE_SURFACE = AirNavigationPrimitiveSurface()
