"""Latest-frame preview sampling; slow viewers cannot backpressure flight."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from .base import PreviewConfig, PreviewFrame, PreviewSource


class PreviewService:
    """Own a source in one capture thread and retain only its newest JPEG."""

    def __init__(self, source_factory: Callable[[], PreviewSource], config: PreviewConfig):
        self.config, self._source_factory = config, source_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: PreviewFrame | None = None
        self._error: str | None = None
        self._samples: deque[tuple[float, float]] = deque(maxlen=120)
        self._thread = threading.Thread(target=self._run, name="flyagent-ui-preview", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> PreviewFrame | None:
        with self._lock:
            return self._latest

    def stats(self) -> dict[str, Any]:
        with self._lock:
            samples, frame, error = list(self._samples), self._latest, self._error
        duration = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0
        timings = sorted(sample[1] for sample in samples)
        return {
            "status": "error" if error else ("streaming" if frame and self._thread.is_alive() else "starting"),
            "error": error, "camera_name": self.config.camera_name,
            "width": self.config.width, "height": self.config.height,
            "target_fps": self.config.fps,
            "capture_fps": (len(samples) - 1) / duration if duration > 0 else 0,
            "capture_p95_ms": timings[min(len(timings) - 1, int(len(timings) * .95))] if timings else None,
            "frame_id": frame.frame_id if frame else None,
            "captured_at": frame.captured_at if frame else None,
            "age_ms": (time.monotonic() - frame.captured_monotonic) * 1000 if frame else None,
        }

    def _run(self) -> None:
        source = None
        try:
            source = self._source_factory()
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    frame = source.get_frame()
                    with self._lock:
                        self._latest, self._error = frame, None
                        self._samples.append((frame.captured_monotonic, frame.capture_ms))
                except Exception as exc:
                    with self._lock:
                        self._error = f"{type(exc).__name__}: {exc}"
                    self._stop.wait(0.5)
                self._stop.wait(max(0, 1 / self.config.fps - (time.monotonic() - started)))
        except Exception as exc:
            with self._lock:
                self._error = f"preview source failed: {exc}"
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    with self._lock:
                        self._error = f"preview cleanup failed: {exc}"

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.config.timeout_sec + 3)
        if self._thread.is_alive():
            raise RuntimeError("Preview client did not stop before scene cleanup")
        if self._error and self._error.startswith("preview cleanup failed"):
            raise RuntimeError(self._error)
