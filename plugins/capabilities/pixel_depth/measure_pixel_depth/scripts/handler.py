"""Measure the exact selected RGB pixel using its observation's metric depth.

The plugin owns coordinates, validity and visual evidence. It never retargets
an invalid pixel, consults benchmark goals or loads detector/segmenter weights.
Public data is compact; sensor provenance and nearby samples stay in the audit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import ImageDraw

from src.memory.context.media_refs import image_media_ref
from src.perception.frame_payload import frame_metadata, load_depth_frame, load_rgb_frame
from src.plugins.contracts.skill import SkillCapabilityResult


def measure_pixel_depth(observation: dict, observation_id: str, pixel_x: int, pixel_y: int,
                        *, run_dir: str | None = None, tool_call_id: str | None = None) -> SkillCapabilityResult:
    """Return a 0.1 m sensor sample; mismatched or stale inputs fail explicitly."""
    if not observation_id or observation_id != observation.get("observation_id"):
        raise ValueError("pixel depth requires the latest observation_id")
    if any(type(v) is not int for v in (pixel_x, pixel_y)):
        raise ValueError("pixel coordinates must be integers")
    frames = observation.get("frames") or {}
    rgb = frames.get("rgb")
    if not isinstance(rgb, dict):
        raise ValueError("latest observation has no original RGB frame")
    image = load_rgb_frame(rgb)
    width, height = image.size
    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        raise ValueError(f"pixel outside original scene bounds: x:[0,{width}), y:[0,{height})")
    data = {"observation_id": observation_id, "pixel_x": pixel_x, "pixel_y": pixel_y,
            "scene_size": [width, height], "distance_m": None,
            "distance_status": "metric_depth_unavailable", "depth_geometry": "sensor_depth",
            "measurement_semantics": "Depth at the selected visible pixel, not horizontal or target-center distance."}
    audit = {"rgb_frame_id": rgb.get("id"), "depth_frame_id": None}
    raw_depth = frames.get("depth")
    if isinstance(raw_depth, dict):
        audit["depth_frame_id"] = raw_depth.get("id")
        for key in ("camera_name", "camera_id"):
            a, b = frame_metadata(rgb).get(key), frame_metadata(raw_depth).get(key)
            if a is not None and b is not None and a != b:
                raise ValueError("RGB and depth originate from different cameras")
        try:
            depth = load_depth_frame(raw_depth, require_metric=True)
        except (ValueError, OSError, EOFError, TypeError) as exc:
            audit["depth_error"] = f"{type(exc).__name__}: {exc}"
        else:
            if depth.shape != (height, width):
                raise ValueError("RGB and metric-depth dimensions do not match")
            source = frame_metadata(raw_depth).get("capture_source")
            data["depth_geometry"] = {"depth_planar": "optical_axis_depth", "depth_perspective": "ray_range"}.get(source, "sensor_depth")
            value = float(depth[pixel_y, pixel_x])
            audit["capture_source"] = source
            if np.isfinite(value) and value > 0:
                data.update(distance_m=round(value, 1), distance_status="ok")
                audit["raw_depth_m"] = value
            else:
                data["distance_status"] = "invalid_depth_at_point"
            # Nearby values describe sensitivity to pixel placement, never
            # replace the requested pixel or reject a sloped object region.
            patch = depth[max(0, pixel_y - 1):pixel_y + 2, max(0, pixel_x - 1):pixel_x + 2]
            values = patch[np.isfinite(patch) & (patch > 0)]
            audit["nearby_valid_samples"] = int(values.size)
            if values.size:
                audit.update(nearby_min_m=float(values.min()), nearby_max_m=float(values.max()))
    else:
        audit["depth_error"] = "latest observation has no metric-depth frame"
    data["audit_details"] = audit
    artifacts, inputs = (), ()
    if run_dir:
        identity = hashlib.sha256(f"{observation_id}|{tool_call_id}|{pixel_x}|{pixel_y}".encode()).hexdigest()[:20]
        path = Path(run_dir) / "artifacts/tool_evidence" / f"pixel_depth_{identity}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        draw = ImageDraw.Draw(image)
        color = (255, 40, 70)
        draw.ellipse((pixel_x - 3, pixel_y - 3, pixel_x + 3, pixel_y + 3), outline=color, width=2)
        draw.text((max(0, min(pixel_x + 6, width - 78)), max(0, min(pixel_y + 6, height - 14))),
                  f"({pixel_x},{pixel_y})", fill=color)
        image.save(path, format="PNG")
        artifacts = (str(path),)
        inputs = (image_media_ref(path, run_dir=run_dir, observation_id=observation_id,
                                 frame_id=f"pixel_depth_{identity}", role="selected_pixel_evidence",
                                 usage="The red marker is the exact sampled point in the associated original RGB. Verify its visible surface.").to_model_input(),)
    message = (f"Selected pixel ({pixel_x}, {pixel_y}) depth: {data['distance_m']} m ({data['depth_geometry']})."
               if data["distance_status"] == "ok" else
               f"No numeric depth at selected pixel: {data['distance_status']}. Reassess the point or viewpoint.")
    return SkillCapabilityResult(message, data, artifacts, inputs)


def handle(payload: dict) -> SkillCapabilityResult:
    """Adapt the generic plugin runtime context without accessing Runtime."""
    args = payload.get("args") or {}
    context = payload.get("runtime_context") or {}
    return measure_pixel_depth(context.get("latest_observation") or {}, str(args.get("observation_id") or ""),
                               args.get("pixel_x"), args.get("pixel_y"),
                               run_dir=context.get("run_dir"), tool_call_id=context.get("tool_call_id"))
