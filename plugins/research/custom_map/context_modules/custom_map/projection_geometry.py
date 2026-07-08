"""Shared depth projection geometry for custom_map and STMR debug artifacts.

This module owns the coordinate math used by standalone research maps. It keeps
the production map path stable while exposing enough camera/world/local evidence
to diagnose whether a bad-looking point cloud comes from depth interpretation,
camera/body axes, pose/yaw, or only HTML rendering.
"""

from __future__ import annotations

import math
from typing import Any

from .schemas import PosePoint

DEPTH_MODE_OPTICAL_Z = "depth_as_optical_z"
DEPTH_MODE_RAY_DISTANCE = "depth_as_ray_distance"
DEPTH_COMPARISON_SAMPLE_LIMIT = 250


def camera_intrinsics(width: int, height: int, fov_degrees: float) -> tuple[float, float, float, float]:
    """Return pinhole intrinsics for the square-FOV UAV debug camera."""

    fov_rad = math.radians(float(fov_degrees))
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def project_depth_pixel(
    *,
    pixel_u: int,
    pixel_v: int,
    raw_depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pose: PosePoint,
    center_pose: PosePoint,
    depth_mode: str,
) -> dict[str, Any]:
    """Project one depth pixel into camera, world, and UAV-local coordinates."""

    if depth_mode == DEPTH_MODE_OPTICAL_Z:
        camera_x = (pixel_u - cx) * raw_depth_m / fx
        camera_y = (pixel_v - cy) * raw_depth_m / fy
        camera_z = raw_depth_m
    elif depth_mode == DEPTH_MODE_RAY_DISTANCE:
        ray_x = (pixel_u - cx) / fx
        ray_y = (pixel_v - cy) / fy
        norm = math.sqrt(ray_x * ray_x + ray_y * ray_y + 1.0)
        camera_x = raw_depth_m * ray_x / norm
        camera_y = raw_depth_m * ray_y / norm
        camera_z = raw_depth_m / norm
    else:
        raise ValueError(f"Unsupported depth projection mode: {depth_mode}")

    forward = camera_z
    right = camera_x
    up = -camera_y
    yaw_rad = math.radians(pose.yaw_deg or 0.0)
    world_x = pose.x + forward * math.cos(yaw_rad) - right * math.sin(yaw_rad)
    world_y = pose.y + forward * math.sin(yaw_rad) + right * math.cos(yaw_rad)
    world_z = (pose.z or 0.0) + up
    local = world_to_local(world_x, world_y, world_z, center_pose)
    distance_m = math.sqrt(forward * forward + right * right + up * up)
    return {
        "depth_mode": depth_mode,
        "pixel_u": int(pixel_u),
        "pixel_v": int(pixel_v),
        "raw_depth_m": round(float(raw_depth_m), 4),
        "camera_x": round(camera_x, 4),
        "camera_y": round(camera_y, 4),
        "camera_z": round(camera_z, 4),
        "camera_forward_m": round(forward, 4),
        "camera_right_m": round(right, 4),
        "camera_up_m": round(up, 4),
        "world_x": round(world_x, 4),
        "world_y": round(world_y, 4),
        "world_z": round(world_z, 4),
        "local_forward_m": round(local["local_forward_m"], 4),
        "local_right_m": round(local["local_right_m"], 4),
        "local_up_m": round(local["local_up_m"], 4),
        "distance_m": round(distance_m, 4),
    }


def world_to_local(
    world_x: float,
    world_y: float,
    world_z: float | None,
    center_pose: PosePoint,
) -> dict[str, float]:
    """Convert world coordinates to current-UAV local forward/right/up."""

    dx = world_x - center_pose.x
    dy = world_y - center_pose.y
    yaw_rad = math.radians(center_pose.yaw_deg or 0.0)
    forward_m = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
    right_m = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    up_m = 0.0 if world_z is None else world_z - (center_pose.z or 0.0)
    return {
        "local_forward_m": forward_m,
        "local_right_m": right_m,
        "local_up_m": up_m,
    }


def local_to_centered_cell(
    local: dict[str, float],
    grid_size: int,
    meters_per_cell: float,
) -> dict[str, int] | None:
    """Map UAV-local forward/right meters to a heading-up grid cell."""

    center = grid_size // 2
    row = int(round(center - local["local_forward_m"] / meters_per_cell))
    col = int(round(center + local["local_right_m"] / meters_per_cell))
    if row < 0 or col < 0 or row >= grid_size or col >= grid_size:
        return None
    return {"row": row, "col": col}


def world_to_centered_cell(
    world_x: float,
    world_y: float,
    center_pose: PosePoint,
    grid_size: int,
    meters_per_cell: float,
) -> dict[str, int] | None:
    return local_to_centered_cell(world_to_local(world_x, world_y, None, center_pose), grid_size, meters_per_cell)


def new_range_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "world_x": [math.inf, -math.inf],
        "world_y": [math.inf, -math.inf],
        "world_z": [math.inf, -math.inf],
        "local_forward_m": [math.inf, -math.inf],
        "local_right_m": [math.inf, -math.inf],
        "local_up_m": [math.inf, -math.inf],
        "camera_x": [math.inf, -math.inf],
        "camera_y": [math.inf, -math.inf],
        "camera_z": [math.inf, -math.inf],
        "raw_depth_m": [math.inf, -math.inf],
    }


def update_range_accumulator(accumulator: dict[str, Any], point: dict[str, Any]) -> None:
    accumulator["count"] = int(accumulator.get("count", 0)) + 1
    for key in (
        "world_x",
        "world_y",
        "world_z",
        "local_forward_m",
        "local_right_m",
        "local_up_m",
        "camera_x",
        "camera_y",
        "camera_z",
        "raw_depth_m",
    ):
        value = point.get(key)
        if value is None:
            continue
        accumulator[key][0] = min(float(accumulator[key][0]), float(value))
        accumulator[key][1] = max(float(accumulator[key][1]), float(value))


def finish_range_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    if int(accumulator.get("count", 0)) <= 0:
        return {"count": 0}
    result: dict[str, Any] = {"count": int(accumulator["count"])}
    for key, value in accumulator.items():
        if key == "count":
            continue
        result[key] = [round(float(value[0]), 3), round(float(value[1]), 3)]
    return result


def summarize_frame_projection_drift(
    frame_point_ranges: list[dict[str, Any]],
    *,
    depth_mode: str = DEPTH_MODE_OPTICAL_Z,
) -> dict[str, Any]:
    """Summarize coarse multi-frame drift in local projected point ranges.

    This is a diagnostic signal, not a correctness verdict. It helps spot large
    pose/yaw/depth inconsistencies by comparing the center of each frame's local
    projected point range.
    """

    centers = []
    for frame in frame_point_ranges:
        ranges = frame.get(depth_mode)
        if not isinstance(ranges, dict) or int(ranges.get("count", 0)) <= 0:
            continue
        try:
            center = {
                "observation_id": frame.get("observation_id"),
                "local_forward_m": _range_center(ranges["local_forward_m"]),
                "local_right_m": _range_center(ranges["local_right_m"]),
                "local_up_m": _range_center(ranges["local_up_m"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        centers.append(center)
    max_delta = 0.0
    for index, left in enumerate(centers):
        for right in centers[index + 1:]:
            delta = math.sqrt(
                (float(left["local_forward_m"]) - float(right["local_forward_m"])) ** 2
                + (float(left["local_right_m"]) - float(right["local_right_m"])) ** 2
                + (float(left["local_up_m"]) - float(right["local_up_m"])) ** 2
            )
            max_delta = max(max_delta, delta)
    return {
        "depth_mode": depth_mode,
        "frame_count": len(centers),
        "max_range_center_delta_m": round(max_delta, 3),
        "range_centers": [
            {
                "observation_id": item["observation_id"],
                "local_forward_m": round(float(item["local_forward_m"]), 3),
                "local_right_m": round(float(item["local_right_m"]), 3),
                "local_up_m": round(float(item["local_up_m"]), 3),
            }
            for item in centers
        ],
    }


def _range_center(value: list[float]) -> float:
    return (float(value[0]) + float(value[1])) / 2.0
