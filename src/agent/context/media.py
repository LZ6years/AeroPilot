"""Hydrate selected MediaRefs into bounded, transient provider payloads."""

from __future__ import annotations

import base64
import hashlib
from collections import OrderedDict
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.memory.context.contracts import MediaRef, PreparedMedia


class MediaHydrationError(ValueError):
    """Raised when selected media cannot be read or verified."""


class MediaHydrator:
    """Instruction-scoped, bounded SHA-LRU for transient Provider media.

    This is the only model-request encoding cache. Durable Context and
    Transcript records retain ``MediaRef`` values, while hydrated data URLs
    exist only for one active instruction and are released by ``clear()``.
    Production instances receive the current ``run_dir`` as ``base_dir`` and
    accept only portable run-relative POSIX refs.  Base-less instances exist
    only for standalone diagnostics and require absolute refs, avoiding any
    implicit dependency on the process working directory.
    """

    def __init__(
        self,
        *,
        base_dir: str | Path | None = None,
        max_entries: int = 32,
        max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("media cache limits must be positive")
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[str, PreparedMedia] = OrderedDict()
        self._bytes = 0
        self._lock = RLock()
        self._reads = self._encodes = self._hits = self._evictions = 0

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "payload_bytes": self._bytes,
                "reads": self._reads,
                "encodes": self._encodes,
                "hits": self._hits,
                "evictions": self._evictions,
            }

    def hydrate(self, value: MediaRef | dict[str, Any]) -> PreparedMedia:
        ref = value if isinstance(value, MediaRef) else MediaRef(**value)
        with self._lock:
            cached = self._entries.get(ref.sha256)
            if cached is not None:
                self._entries.move_to_end(ref.sha256)
                self._hits += 1
                return PreparedMedia(ref, cached.data_url, cached.provider_sha256)

            path = self._resolve(ref.ref)
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise MediaHydrationError(f"media unavailable: {ref.ref}") from exc
            self._reads += 1
            if hashlib.sha256(raw).hexdigest() != ref.sha256:
                raise MediaHydrationError(f"media digest mismatch: {ref.ref}")
            try:
                with Image.open(BytesIO(raw)) as source:
                    image = source.convert("RGB")
                output = BytesIO()
                image.save(output, format="PNG")
                provider_bytes = output.getvalue()
            except (OSError, EOFError, UnidentifiedImageError) as exc:
                raise MediaHydrationError(f"invalid image media: {ref.ref}") from exc
            self._encodes += 1
            provider_digest = hashlib.sha256(provider_bytes).hexdigest()
            data_url = "data:image/png;base64," + base64.b64encode(
                provider_bytes
            ).decode("ascii")
            prepared = PreparedMedia(ref, data_url, provider_digest)
            size = len(data_url)
            if size <= self.max_bytes:
                while self._entries and (
                    len(self._entries) >= self.max_entries
                    or self._bytes + size > self.max_bytes
                ):
                    _, removed = self._entries.popitem(last=False)
                    self._bytes -= len(removed.data_url)
                    self._evictions += 1
                self._entries[ref.sha256] = prepared
                self._bytes += size
            return prepared

    def _resolve(self, value: str) -> Path:
        """Resolve one ref through exactly one explicit path contract."""

        if self.base_dir is None:
            path = Path(value)
            if not path.is_absolute():
                raise MediaHydrationError(
                    f"standalone media ref must be absolute: {value}"
                )
            return path

        if "\\" in value:
            raise MediaHydrationError(
                f"run-relative media ref must use POSIX separators: {value}"
            )
        if Path(value).is_absolute():
            raise MediaHydrationError(f"invalid run-relative media ref: {value}")
        relative = PurePosixPath(value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise MediaHydrationError(f"invalid run-relative media ref: {value}")
        root = self.base_dir.resolve()
        path = root.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(root):
            raise MediaHydrationError(f"media ref escapes run_dir: {value}")
        return path
