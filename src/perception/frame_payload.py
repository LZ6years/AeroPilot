"""Read validated frame pixels from artifacts or in-memory transport payloads.

No simulator, Runtime or benchmark dependency belongs here. DepthVis remains
normalized; callers asking for metric depth must explicitly require meters.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def frame_metadata(frame: dict[str, Any]) -> dict[str, Any]:
    return {**dict(frame.get("metadata") or {}), **{k: v for k, v in frame.items() if v is not None}}


def frame_bytes(frame: dict[str, Any]) -> bytes:
    """Prefer existing artifacts, then payload; corrupt payloads fail explicitly."""
    meta = frame_metadata(frame)
    for key in ("artifact_path", "path", "artifact_ref"):
        raw = meta.get(key)
        if isinstance(raw, str) and raw and Path(raw).is_file():
            return Path(raw).read_bytes()
    raw = meta.get("base64")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Frame has no readable artifact or base64 payload")
    try:
        payload = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid frame base64 payload") from exc
    if not payload:
        raise ValueError("Empty frame payload")
    return payload


def _validate_dimensions(frame: dict[str, Any], width: int, height: int) -> None:
    meta = frame_metadata(frame)
    for key, actual in (("width", width), ("height", height)):
        if meta.get(key) is not None and int(meta[key]) != actual:
            raise ValueError(f"Frame {key} disagrees with decoded payload: {meta[key]} != {actual}")


def load_rgb_frame(frame: dict[str, Any]) -> Image.Image:
    """Return detached RGB pixels, preserving the original scene dimensions."""
    with Image.open(BytesIO(frame_bytes(frame))) as source:
        image = source.convert("RGB")
    _validate_dimensions(frame, *image.size)
    return image


def load_depth_frame(frame: dict[str, Any], *, require_metric: bool = False) -> np.ndarray:
    meta = frame_metadata(frame)
    if require_metric and meta.get("depth_unit") != "meter":
        raise ValueError("latest depth frame is not declared in meters")
    if meta.get("capture_source") == "depth_vis" and require_metric:
        raise ValueError("DepthVis is normalized, not metric depth")
    values = np.load(BytesIO(frame_bytes(frame)), allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "fiu":
        raise ValueError("Depth payload must be a numeric NPY array")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
        if width <= 0 or height <= 0 or values.size != width * height:
            raise ValueError("Flat depth payload has invalid dimensions")
        values = values.reshape(height, width)
    if values.ndim != 2 or not values.size:
        raise ValueError("Depth payload must be a non-empty 2D array")
    _validate_dimensions(frame, values.shape[1], values.shape[0])
    return values
