"""Read validated frame pixels from artifacts or in-memory transport payloads.

No simulator, Runtime or benchmark dependency belongs here. DepthVis remains
normalized; callers asking for metric depth must explicitly require meters.
"""

import base64
from collections import OrderedDict
import hashlib
from io import BytesIO
from pathlib import Path
from threading import RLock
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


def rgb_data_url(frame: dict[str, Any]) -> tuple[str, tuple[int, int]]:
    image = load_rgb_frame(frame)
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"), image.size


class RgbEncodingCache:
    """Bounded instance-owned RGB encoding cache, never a source-byte cache.

    Source precedence and validation stay identical to the uncached path. The
    byte budget counts retained ASCII Data URL payloads, not total process RSS.
    A lock covers lookup through insertion so concurrent same-image requests
    cannot duplicate encoding or corrupt LRU accounting. No GPU objects live here.
    """

    def __init__(self, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("RGB encoding cache bytes must be a non-negative integer")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, tuple[str, tuple[int, int]]] = OrderedDict()
        self._bytes = 0
        self._namespace: Any = None
        self._lock = RLock()
        self._hits = self._misses = self._encodes = self._evictions = 0

    def clear(self) -> None:
        """Release retained results; diagnostic counters remain cumulative."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._namespace = None

    def stats(self) -> dict[str, int]:
        """Human diagnostics only; callers must not insert these into context."""
        with self._lock:
            return dict(hits=self._hits, misses=self._misses, encodes=self._encodes,
                        evictions=self._evictions, entries=len(self._entries), payload_bytes=self._bytes)

    def set_namespace(self, namespace: Any) -> None:
        """Clear on lifecycle changes even when the new build has no image."""
        with self._lock:
            if self._namespace != namespace:
                self.clear()
                self._namespace = namespace

    def encode(self, frame: dict[str, Any], *, namespace: Any = None) -> tuple[str, tuple[int, int]]:
        if not self.max_bytes:
            return rgb_data_url(frame)
        # Always re-read using the existing source priority. Same path/stat is
        # insufficient: even equal-size same-timestamp replacements must miss.
        raw = frame_bytes(frame)
        key = hashlib.sha256(raw).hexdigest()
        with self._lock:
            self.set_namespace(namespace)
            if key in self._entries:
                result = self._entries[key]
                _validate_dimensions(frame, *result[1])
                self._entries.move_to_end(key)
                self._hits += 1
                return result
            self._misses += 1
            with Image.open(BytesIO(raw)) as source:
                image = source.convert("RGB")
            _validate_dimensions(frame, *image.size)
            output = BytesIO()
            image.save(output, format="PNG")
            result = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii"), image.size
            self._encodes += 1
            size = len(result[0])  # Data URLs contain ASCII characters only.
            if size <= self.max_bytes:
                while self._entries and (len(self._entries) >= 4 or self._bytes + size > self.max_bytes):
                    _, removed = self._entries.popitem(last=False)
                    self._bytes -= len(removed[0])
                    self._evictions += 1
                self._entries[key] = result
                self._bytes += size
            return result
