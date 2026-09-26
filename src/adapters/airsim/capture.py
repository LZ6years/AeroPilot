"""AirSim image transport: validate and encode payloads independently of persistence."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from ..uav_primitives import PrimitiveResult
from .performance import AirSimPerformance

@dataclass(frozen=True)
class _AirSimCaptureSpec:
    """Translate one generic capture source into stable AirSim request flags."""

    image_type_attribute: str
    pixels_as_float: bool
    compress: bool
    payload_kind: str


_AIRSIM_CAPTURE_SPECS = {
    # Request formats are intrinsic to the capture source.  Saving artifacts
    # must never change the simulator RPC contract.
    "scene": _AirSimCaptureSpec("Scene", False, False, "raw_color"),
    "depth_vis": _AirSimCaptureSpec("DepthVis", False, True, "depth_vis"),
    "depth_perspective": _AirSimCaptureSpec(
        "DepthPerspective", True, False, "float_depth"
    ),
    "segmentation": _AirSimCaptureSpec(
        "Segmentation", False, False, "raw_color"
    ),
}

_DEFAULT_CAPTURE_SOURCE = {
    "rgb": "scene",
    "scene": "scene",
    "color": "scene",
    "depth": "depth_perspective",
    "depth_perspective": "depth_perspective",
    "depth_vis": "depth_vis",
    "segmentation": "segmentation",
    "mask": "segmentation",
}


class AirSimCapture:
    """Scene-owned capture routing and optional artifact writer."""

    def __init__(self, camera_name: str, vehicle_name: str, artifact_dir=None, save_artifacts=False, *, performance=None):
        self.performance = performance or AirSimPerformance()
        self.camera_name = camera_name
        self.vehicle_name = vehicle_name
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = save_artifacts
        self._frame_index = 0

    def get_frames(
        self,
        client: Any,
        airsim: Any,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> PrimitiveResult:
        """Capture canonical frames using caller-selected generic sources.

        ``capture_sources`` is supplied by the run-level ObservationProfile.
        No map ID, scene ID, or benchmark name participates in routing.
        """

        requests = []
        resolved: list[tuple[str, str, _AirSimCaptureSpec]] = []
        for modality in modalities:
            source = self._capture_source(modality, capture_sources)
            spec = self._capture_spec(source)
            image_type = getattr(airsim.ImageType, spec.image_type_attribute)
            requests.append(
                airsim.ImageRequest(
                    self.camera_name,
                    image_type,
                    spec.pixels_as_float,
                    spec.compress,
                )
            )
            resolved.append((modality, source, spec))
        responses = client.simGetImages(
            requests, vehicle_name=self.vehicle_name
        )
        if len(responses) != len(resolved):
            raise RuntimeError(
                "AirSim returned an unexpected response count: "
                f"requested={len(resolved)}, received={len(responses)}"
            )
        with self.performance.span("capture.decode_and_write"):
            frames = {
                modality: self._frame_metadata(modality, source, spec, response)
                for (modality, source, spec), response in zip(resolved, responses)
            }
        return PrimitiveResult(True, "AirSim UAV frames", {"frames": frames})

    def _frame_metadata(self, modality: str, capture_source: str,
                        spec: _AirSimCaptureSpec, response: Any) -> dict[str, Any]:
        """Decode every response once; persistence never gates validation."""
        import numpy as np
        from PIL import Image

        width, height = int(response.width), int(response.height)
        if width <= 0 or height <= 0:
            raise ValueError(f"{capture_source} returned invalid dimensions {width}x{height}")
        raw = bytes(response.image_data_uint8 or b"")
        floats = response.image_data_float
        metadata = {
            "width": width, "height": height, "capture_source": capture_source,
            "image_type": spec.image_type_attribute, "pixels_as_float": spec.pixels_as_float,
            "compressed": spec.compress, "bytes": len(raw),
            "float_count": len(floats) if floats is not None else 0,
        }
        preview = None
        if spec.payload_kind == "raw_color":
            channels, remainder = divmod(len(raw), width * height)
            if remainder or channels not in {3, 4}:
                raise ValueError(f"{capture_source} raw payload has {len(raw)} bytes for {width}x{height}; expected 3 or 4 channels")
            pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)
            buffer = BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            payload, suffix = buffer.getvalue(), "png"
            folder = "rgb" if modality in {"rgb", "scene", "color"} else modality
        else:
            if spec.payload_kind == "float_depth" or metadata["float_count"]:
                values = np.asarray(floats if floats is not None else [], dtype=np.float32)
                self._validate_pixel_count(capture_source, values.size, width * height)
                values = values.reshape(height, width)
            else:
                if not raw:
                    raise ValueError(f"{capture_source} returned empty encoded payload")
                with Image.open(BytesIO(raw)) as image:
                    color = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if color.shape[:2] != (height, width):
                    raise ValueError(f"{capture_source} dimensions disagree with payload")
                values = color[:, :, 1].astype(np.float32) / 255.0
            # DepthVis is a normalized display value, never a metric distance.
            metadata["depth_unit"] = "normalized" if capture_source == "depth_vis" else "meter"
            if capture_source == "depth_vis":
                preview = Image.fromarray(np.round(np.clip(values, 0, 1) * 255).astype(np.uint8))
            buffer = BytesIO()
            np.save(buffer, values, allow_pickle=False)
            payload, suffix, folder = buffer.getvalue(), "npy", "depth"
        metadata.update(base64=base64.b64encode(payload).decode("ascii"), encoding=suffix)
        if self.save_artifacts:
            path = self._next_artifact_path(folder, suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.performance.span("capture.artifact_write"):
                path.write_bytes(payload)
            metadata["artifact_path"] = str(path)
            if preview is not None:
                preview_path = self._depth_preview_path(path)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                with self.performance.span("capture.artifact_write"):
                    preview.save(preview_path, format="PNG")
                metadata["preview_path"] = str(preview_path)
        return metadata

    def _depth_preview_path(self, depth_path: Path) -> Path:
        if self.artifact_dir is None:
            raise RuntimeError("artifact_dir is required when saving artifacts")
        return self.artifact_dir / "depth_preview" / f"{depth_path.stem}.png"

    @staticmethod
    def _validate_pixel_count(
        capture_source: str, actual: int, expected: int
    ) -> None:
        if expected <= 0 or actual != expected:
            raise ValueError(
                f"{capture_source} returned {actual} values; expected {expected}"
            )

    def _next_artifact_path(self, folder: str, suffix: str) -> Path:
        # Adapter instances can be recreated inside one run, so never assume an
        # in-memory frame counter is globally unique for the run directory.
        if self.artifact_dir is None:
            raise RuntimeError("artifact_dir is required when saving artifacts")
        while True:
            self._frame_index += 1
            path = self.artifact_dir / folder / f"frame_{self._frame_index:06d}.{suffix}"
            if not path.exists():
                return path

    @staticmethod
    def _capture_source(
        modality: str,
        capture_sources: dict[str, str] | None,
    ) -> str:
        key = modality.strip().lower()
        if capture_sources and key in capture_sources:
            return str(capture_sources[key]).strip().lower()
        try:
            return _DEFAULT_CAPTURE_SOURCE[key]
        except KeyError as exc:
            raise ValueError(
                f"AirSim has no default capture source for modality {modality!r}"
            ) from exc

    @staticmethod
    def _capture_spec(capture_source: str) -> _AirSimCaptureSpec:
        try:
            return _AIRSIM_CAPTURE_SPECS[capture_source]
        except KeyError as exc:
            available = ", ".join(sorted(_AIRSIM_CAPTURE_SPECS))
            raise ValueError(
                f"Unsupported AirSim capture source {capture_source!r}; "
                f"available: {available}"
            ) from exc
