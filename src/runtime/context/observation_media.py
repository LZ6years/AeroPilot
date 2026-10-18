"""Own original observation binaries independently of simulator scene storage.

A scene can outlive several Runtime runs. Import its media before publishing any
observation so RGB context, paired depth and historical place photos all refer
to the receiving run. This module preserves encoded bytes and sensor metadata;
it neither derives images nor changes the run-relative model-media contract.
"""

from __future__ import annotations

import base64
import copy
import hashlib
from io import BytesIO
import os
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image

from src.perception.frame_payload import (
    frame_bytes,
    frame_metadata,
    load_depth_frame,
    load_rgb_frame,
)


class ObservationMediaStore:
    """Import real frame payloads into a run, reusing identical local bytes.

    ``run_dir`` belongs to the receiving Runtime, not the simulator service.
    Backend frames remain untouched. Original external paths are retained only
    as provenance; authoritative artifact paths point at the receiving run.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.root = self.run_dir / "artifacts" / "observations"

    def materialize_frames(self, frames: dict[str, Any]) -> dict[str, Any]:
        """Return isolated frame metadata after importing every supplied binary.

        Mock and metadata-only frames retain their existing unavailable-image
        semantics. A supplied but unreadable payload is an observation failure,
        rather than a valid observation with silently discarded visual input.
        """
        result = copy.deepcopy(frames)
        for frame in result.values():
            if not isinstance(frame, dict):
                continue
            metadata = frame_metadata(frame)
            if metadata.get("mock") is True:
                continue
            source = self._source_path(metadata)
            if source is None and not any(metadata.get(key) for key in ("artifact_path", "path", "base64")):
                continue

            payload = frame_bytes(frame)
            suffix = self._validate_payload(payload, metadata)
            path, digest = self._persist(payload, suffix, source)
            if source is not None and path != source:
                frame["source_artifact_path"] = str(source)
            frame.update(artifact_path=str(path), path=str(path), sha256=digest)

            preview = metadata.get("preview_path")
            if preview:
                preview_source = Path(preview).resolve()
                preview_bytes = frame_bytes({"path": str(preview_source)})
                preview_suffix = self._validate_payload(preview_bytes, {})
                preview_path, _ = self._persist(preview_bytes, preview_suffix, preview_source)
                if preview_path != preview_source:
                    frame["source_preview_path"] = str(preview_source)
                frame["preview_path"] = str(preview_path)
        return result

    @staticmethod
    def _source_path(metadata: dict[str, Any]) -> Path | None:
        """Match the existing frame reader's artifact-before-payload ordering."""
        for key in ("artifact_path", "path", "artifact_ref"):
            value = metadata.get(key)
            if isinstance(value, str) and value and Path(value).is_file():
                return Path(value).resolve()
        return None

    @staticmethod
    def _validate_payload(payload: bytes, metadata: dict[str, Any]) -> str:
        """Validate the exact bytes being imported with shared sensor readers.

        The transient inline view avoids rereading a possibly reused capture
        filename during validation. No decoded/re-encoded image is persisted.
        """
        view = {key: metadata[key] for key in ("width", "height", "capture_source", "depth_unit")
                if key in metadata}
        view["base64"] = base64.b64encode(payload).decode("ascii")
        if payload.startswith(b"\x93NUMPY"):
            load_depth_frame(view)
            return ".npy"
        load_rgb_frame(view)
        with Image.open(BytesIO(payload)) as image:
            return "." + str(image.format).lower()

    def _persist(self, payload: bytes, suffix: str, source: Path | None) -> tuple[Path, str]:
        """Publish complete content-addressed files; never rewrite source media."""
        digest = hashlib.sha256(payload).hexdigest()
        if source is not None and source.is_relative_to(self.run_dir):
            return source, digest
        path = (self.root / digest[:2] / f"{digest}{suffix}").resolve()
        if not path.is_relative_to(self.run_dir):
            raise ValueError("observation media destination is outside run_dir")
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("existing observation media digest mismatch")
            return path, digest
        path.parent.mkdir(parents=True, exist_ok=True)
        # A failed disk write must not leave a partial file at its claimed hash.
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path, digest
