"""Capture and decode observation modalities for the UnrealCV adapter package.

Observation choice comes from the run-level ``ObservationProfile``.  This
module only translates generic capture-source names into UnrealCV commands,
performs one synchronized batch request, and optionally persists artifacts.
It never inspects benchmark, map, or scene names.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class UnrealCvCaptureSpec:
    """Describe one supported generic capture source."""

    command_view: str
    command_format: str
    payload_kind: str


CAPTURE_SOURCES = {
    "scene": UnrealCvCaptureSpec("lit", "png", "color_png"),
    "depth_perspective": UnrealCvCaptureSpec("depth", "npy", "depth_npy"),
    "segmentation": UnrealCvCaptureSpec("object_mask", "png", "color_png"),
}

DEFAULT_SOURCE_BY_MODALITY = {
    "rgb": "scene",
    "depth": "depth_perspective",
    "segmentation": "segmentation",
    "mask": "segmentation",
}


class UnrealCvCapture:
    """Issue aligned image requests and build JSON-safe frame metadata."""

    def __init__(
        self,
        client: Any,
        *,
        camera_id: str,
        units_per_meter: float,
        artifact_dir: str | Path | None,
        save_artifacts: bool,
        sync_camera: Callable[[], None],
    ) -> None:
        self.client = client
        self.camera_id = str(camera_id)
        self.units_per_meter = float(units_per_meter)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self.save_artifacts = bool(save_artifacts)
        self.sync_camera = sync_camera
        self._capture_index = 0

    def capture(
        self,
        modalities: list[str],
        capture_sources: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Capture all requested image modalities in one UnrealCV batch."""

        resolved: list[tuple[str, str, UnrealCvCaptureSpec, str]] = []
        for modality in modalities:
            normalized = str(modality).strip().lower()
            source = self._capture_source(normalized, capture_sources)
            try:
                spec = CAPTURE_SOURCES[source]
            except KeyError as exc:
                available = ", ".join(sorted(CAPTURE_SOURCES))
                raise ValueError(
                    f"Unsupported UnrealCV capture source {source!r} for "
                    f"{normalized!r}; available: {available}"
                ) from exc
            command = (
                f"vget /camera/{self.camera_id}/"
                f"{spec.command_view} {spec.command_format}"
            )
            resolved.append((normalized, source, spec, command))

        self.sync_camera()
        commands = [entry[3] for entry in resolved]
        responses = self.client.request_batch(commands)
        if not isinstance(responses, (list, tuple)) or len(responses) != len(resolved):
            received = len(responses) if isinstance(responses, (list, tuple)) else "non-list"
            raise RuntimeError(
                "UnrealCV capture response count mismatch: "
                f"requested={len(resolved)}, received={received}"
            )

        self._capture_index += 1
        capture_group = f"unrealcv_{self._capture_index:06d}"
        return {
            modality: self._frame_metadata(
                modality,
                source,
                spec,
                command,
                response,
                capture_group,
            )
            for (modality, source, spec, command), response in zip(
                resolved, responses
            )
        }

    def _capture_source(
        self,
        modality: str,
        capture_sources: dict[str, str] | None,
    ) -> str:
        try:
            default = DEFAULT_SOURCE_BY_MODALITY[modality]
        except KeyError as exc:
            available = ", ".join(sorted(DEFAULT_SOURCE_BY_MODALITY))
            raise ValueError(
                f"Unsupported UnrealCV observation modality {modality!r}; "
                f"available: {available}"
            ) from exc
        if capture_sources is None:
            return default
        return str(capture_sources.get(modality, default)).strip().lower()

    def _frame_metadata(
        self,
        modality: str,
        source: str,
        spec: UnrealCvCaptureSpec,
        command: str,
        response: Any,
        capture_group: str,
    ) -> dict[str, Any]:
        payload = _require_bytes(response, command)
        metadata: dict[str, Any] = {
            "capture_source": source,
            "command": command,
            "capture_group": capture_group,
            "bytes": len(payload),
            "base64": base64.b64encode(payload).decode("ascii"),
        }

        if spec.payload_kind == "color_png":
            image = _decode_png(payload, command)
            metadata.update({"width": image.width, "height": image.height})
            if self.save_artifacts:
                folder = "rgb" if modality == "rgb" else "segmentation"
                path = self._artifact_path(folder, "png")
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="PNG")
                metadata["artifact_path"] = str(path)
            return metadata

        if spec.payload_kind == "depth_npy":
            import numpy as np

            try:
                native_depth = np.asarray(
                    np.load(BytesIO(payload), allow_pickle=False),
                    dtype=np.float32,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not decode UnrealCV depth response for {command}"
                ) from exc
            if native_depth.ndim != 2 or not native_depth.size:
                raise RuntimeError(
                    "UnrealCV depth response must be a non-empty 2D array; "
                    f"received shape={native_depth.shape}"
                )
            depth_m = native_depth / self.units_per_meter
            metadata.update(
                {
                    "width": int(depth_m.shape[1]),
                    "height": int(depth_m.shape[0]),
                    "depth_unit": "meter",
                    "native_units_per_meter": self.units_per_meter,
                    "finite": bool(np.isfinite(depth_m).all()),
                }
            )
            converted = BytesIO()
            np.save(converted, depth_m)
            metadata["base64"] = base64.b64encode(converted.getvalue()).decode("ascii")
            if self.save_artifacts:
                path = self._artifact_path("depth", "npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, depth_m)
                preview_path = self._depth_preview_path(path)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                _save_depth_preview(depth_m, preview_path)
                metadata["artifact_path"] = str(path)
                metadata["preview_path"] = str(preview_path)
            return metadata

        raise ValueError(f"Unsupported UnrealCV payload kind: {spec.payload_kind}")

    def _artifact_path(self, folder: str, suffix: str) -> Path:
        if self.artifact_dir is None:
            raise RuntimeError("artifact_dir is required when saving artifacts")
        base = self.artifact_dir / folder / f"frame_{self._capture_index:06d}.{suffix}"
        if not base.exists():
            return base
        collision = 1
        while True:
            candidate = (
                self.artifact_dir
                / folder
                / f"frame_{self._capture_index:06d}_{collision:02d}.{suffix}"
            )
            if not candidate.exists():
                return candidate
            collision += 1

    def _depth_preview_path(self, depth_path: Path) -> Path:
        if self.artifact_dir is None:
            raise RuntimeError("artifact_dir is required when saving artifacts")
        return self.artifact_dir / "depth_preview" / f"{depth_path.stem}.png"


def _require_bytes(response: Any, command: str) -> bytes:
    if not isinstance(response, (bytes, bytearray)):
        text = str(response)
        if text.lower().startswith("error"):
            raise RuntimeError(f"UnrealCV command failed: {command}: {text}")
        raise RuntimeError(
            f"UnrealCV image command returned non-bytes data: {command}: {text!r}"
        )
    return bytes(response)


def _decode_png(payload: bytes, command: str):
    from PIL import Image

    try:
        with Image.open(BytesIO(payload)) as encoded:
            return encoded.convert("RGB").copy()
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode UnrealCV PNG response for {command}"
        ) from exc


def _save_depth_preview(depth_m, output_path: Path) -> None:
    """Create a robust human preview without altering the numeric artifact."""

    import numpy as np
    from PIL import Image

    finite = np.asarray(depth_m, dtype=np.float32)
    mask = np.isfinite(finite)
    if not mask.any():
        preview = np.zeros(finite.shape, dtype=np.uint8)
    else:
        values = finite[mask]
        low, high = np.percentile(values, [1.0, 99.0])
        if high <= low:
            preview = np.zeros(finite.shape, dtype=np.uint8)
        else:
            normalized = np.clip((finite - low) / (high - low), 0.0, 1.0)
            normalized[~mask] = 0.0
            preview = np.round(normalized * 255.0).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(output_path, format="PNG")
