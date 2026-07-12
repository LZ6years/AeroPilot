"""Low-overhead stage profiling for standalone STMR experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


_BYTES_PER_MIB = 1024.0 * 1024.0


@dataclass
class StageContext:
    """Mutable metadata returned to a measured block."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, **values: Any) -> None:
        self.metadata.update(values)


class StageProfiler:
    """Measure wall time, process RSS, and optional CUDA memory per stage."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        device: str = "auto",
        context: dict[str, Any] | None = None,
        torch_module: Any | None = None,
        process: Any | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.context = dict(context or {})
        self.records: list[dict[str, Any]] = []
        self._torch_module = torch_module
        self._process = process

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        cold_start: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[StageContext]:
        """Measure one stage and append a serializable record."""

        token = StageContext(dict(metadata or {}))
        if not self.enabled:
            yield token
            return

        torch = self._torch()
        cuda_enabled = self._cuda_enabled(torch)
        if cuda_enabled:
            self._cuda_synchronize(torch)
            try:
                torch.cuda.reset_peak_memory_stats()
            except (RuntimeError, TypeError):
                pass
        before = self._memory_snapshot(torch, cuda_enabled)
        started = perf_counter()
        status = "ok"
        error_type: str | None = None
        try:
            yield token
        except BaseException as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            if cuda_enabled:
                self._cuda_synchronize(torch)
            elapsed_ms = (perf_counter() - started) * 1000.0
            after = self._memory_snapshot(torch, cuda_enabled)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **self.context,
                "stage": str(name),
                "run_phase": "cold_start" if cold_start else "warm",
                "status": status,
                "elapsed_ms": round(elapsed_ms, 4),
                "cpu_rss_before_mb": before["cpu_rss_mb"],
                "cpu_rss_after_mb": after["cpu_rss_mb"],
                "cpu_rss_delta_mb": _difference(after["cpu_rss_mb"], before["cpu_rss_mb"]),
                "cuda_allocated_before_mb": before["cuda_allocated_mb"],
                "cuda_allocated_after_mb": after["cuda_allocated_mb"],
                "cuda_allocated_delta_mb": _difference(
                    after["cuda_allocated_mb"], before["cuda_allocated_mb"]
                ),
                "cuda_reserved_after_mb": after["cuda_reserved_mb"],
                "cuda_peak_allocated_mb": after["cuda_peak_allocated_mb"],
                **token.metadata,
            }
            if error_type is not None:
                record["error_type"] = error_type
            self.records.append(record)

    def child(self, **context: Any) -> "StageProfiler":
        """Create a profiler view that writes into the same record list."""

        child = StageProfiler(
            enabled=self.enabled,
            device=self.device,
            context={**self.context, **context},
            torch_module=self._torch_module,
            process=self._process,
        )
        child.records = self.records
        return child

    def summary(self) -> dict[str, Any]:
        """Aggregate stage samples with stable P50/P95 metrics."""

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self.records:
            grouped.setdefault(str(record.get("stage", "unknown")), []).append(record)
        stages: dict[str, dict[str, Any]] = {}
        for name, records in sorted(grouped.items()):
            elapsed = [float(item["elapsed_ms"]) for item in records if item.get("elapsed_ms") is not None]
            cuda_peaks = [
                float(item["cuda_peak_allocated_mb"])
                for item in records
                if item.get("cuda_peak_allocated_mb") is not None
            ]
            rss = [float(item["cpu_rss_after_mb"]) for item in records if item.get("cpu_rss_after_mb") is not None]
            stages[name] = {
                "sample_count": len(records),
                "cold_start_count": sum(item.get("run_phase") == "cold_start" for item in records),
                "error_count": sum(item.get("status") != "ok" for item in records),
                "total_ms": round(sum(elapsed), 4),
                "mean_ms": round(sum(elapsed) / len(elapsed), 4) if elapsed else 0.0,
                "p50_ms": round(_percentile(elapsed, 0.50), 4) if elapsed else 0.0,
                "p95_ms": round(_percentile(elapsed, 0.95), 4) if elapsed else 0.0,
                "max_ms": round(max(elapsed), 4) if elapsed else 0.0,
                "cuda_peak_allocated_mb": round(max(cuda_peaks), 4) if cuda_peaks else None,
                "cpu_rss_peak_mb": round(max(rss), 4) if rss else None,
            }
        return {
            "sample_count": len(self.records),
            "stage_count": len(stages),
            "stages": stages,
        }

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def write_summary(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _torch(self) -> Any | None:
        if self._torch_module is not None:
            return self._torch_module
        try:
            import torch
        except ModuleNotFoundError:
            return None
        self._torch_module = torch
        return torch

    def _cuda_enabled(self, torch: Any | None) -> bool:
        if torch is None or not hasattr(torch, "cuda"):
            return False
        if self.device == "cpu":
            return False
        try:
            return bool(torch.cuda.is_available())
        except (RuntimeError, AttributeError):
            return False

    def _cuda_synchronize(self, torch: Any) -> None:
        try:
            torch.cuda.synchronize()
        except (RuntimeError, TypeError):
            pass

    def _memory_snapshot(self, torch: Any | None, cuda_enabled: bool) -> dict[str, float | None]:
        cpu_rss_mb = None
        process = self._psutil_process()
        if process is not None:
            try:
                cpu_rss_mb = round(float(process.memory_info().rss) / _BYTES_PER_MIB, 4)
            except (AttributeError, OSError):
                cpu_rss_mb = None
        snapshot: dict[str, float | None] = {
            "cpu_rss_mb": cpu_rss_mb,
            "cuda_allocated_mb": None,
            "cuda_reserved_mb": None,
            "cuda_peak_allocated_mb": None,
        }
        if cuda_enabled and torch is not None:
            try:
                snapshot.update(
                    {
                        "cuda_allocated_mb": round(float(torch.cuda.memory_allocated()) / _BYTES_PER_MIB, 4),
                        "cuda_reserved_mb": round(float(torch.cuda.memory_reserved()) / _BYTES_PER_MIB, 4),
                        "cuda_peak_allocated_mb": round(
                            float(torch.cuda.max_memory_allocated()) / _BYTES_PER_MIB,
                            4,
                        ),
                    }
                )
            except (RuntimeError, AttributeError):
                pass
        return snapshot

    def _psutil_process(self) -> Any | None:
        if self._process is not None:
            return self._process
        try:
            import psutil
        except ModuleNotFoundError:
            return None
        self._process = psutil.Process(os.getpid())
        return self._process


def artifact_size_bytes(path: str | Path | None) -> int:
    """Return an artifact size without raising for absent optional files."""

    if path is None:
        return 0
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _difference(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return round(after - before, 4)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = ["StageContext", "StageProfiler", "artifact_size_bytes"]
