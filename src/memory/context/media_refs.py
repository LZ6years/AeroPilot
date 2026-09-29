"""Construct binary-free MediaRefs from module-owned image artifacts."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.memory.context.contracts import MediaRef


def run_relative_media_ref(path: str | Path, run_dir: str | Path) -> str:
    """Return a portable POSIX media path contained by one Runtime run.

    Capture adapters may report absolute paths or paths relative to the process
    working directory.  Context V2 persists neither form: durable model media
    is always addressed relative to ``run_dir`` so a complete run can be moved
    without rewriting checkpoints or Transcript records.
    """

    root = Path(run_dir).resolve()
    artifact = Path(path).resolve()
    try:
        relative = artifact.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"media artifact is outside run_dir: {path}") from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"invalid run-relative media artifact: {path}")
    return relative.as_posix()


def image_media_ref(
    path: str | Path,
    *,
    run_dir: str | Path | None = None,
    observation_id: str,
    frame_id: str,
    role: str,
    usage: str = "",
    scene_width: int | None = None,
    scene_height: int | None = None,
) -> MediaRef:
    """Read one selected artifact once to establish identity and a durable ref.

    Runtime producers pass ``run_dir`` and receive a portable run-relative
    reference.  Standalone diagnostics may omit it; their transient reference
    is absolute and is valid only with a base-less ``MediaHydrator``.
    """

    artifact = Path(path)
    try:
        raw = artifact.read_bytes()
        with Image.open(BytesIO(raw)) as source:
            width, height = source.size
    except (OSError, EOFError, UnidentifiedImageError) as exc:
        raise ValueError(f"unreadable image artifact: {artifact}") from exc
    media_ref = (
        run_relative_media_ref(artifact, run_dir)
        if run_dir is not None
        else artifact.resolve().as_posix()
    )
    return MediaRef(
        ref=media_ref,
        sha256=hashlib.sha256(raw).hexdigest(),
        mime_type="image/png",
        observation_id=observation_id,
        frame_id=frame_id,
        role=role,
        usage=usage,
        width=scene_width or width,
        height=scene_height or height,
        display_width=width,
        display_height=height,
    )
