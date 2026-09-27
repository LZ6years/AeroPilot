"""Optional parent-side NVIDIA sampling; unsupported values stay unknown.

No Torch import, model allocation or GPU policy changes. A bounded subprocess
timeout keeps missing/broken drivers from blocking experiment supervision.
"""

import csv
import json
from pathlib import Path
import subprocess
import threading
import time


def sample_gpu(*, run=subprocess.run):
    query = "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory"
    try:
        result = run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=2, check=False)
        if result.returncode:
            return {"available": False, "error": result.stderr.strip()[:1000], "devices": []}
        devices = []
        for values in csv.reader(result.stdout.splitlines()):
            if len(values) != 8:
                raise ValueError("Unexpected nvidia-smi column count")
            def number(value):
                try:
                    return float(value.strip())
                except ValueError:
                    return None
            devices.append(dict(zip(("index", "uuid", "name", "total_mib", "used_mib", "free_mib", "gpu_percent", "memory_percent"),
                                    [number(values[0]), values[1].strip(), values[2].strip(), *map(number, values[3:])])))
        return {"available": bool(devices), "devices": devices}
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"available": False, "error_type": type(exc).__name__, "error": str(exc)[:1000], "devices": []}


class GpuResourceMonitor:
    """One background sampler in the surviving parent, independent of child exit."""

    def __init__(self, path, *, interval=1., sampler=sample_gpu):
        if interval <= 0:
            raise ValueError("GPU sampling interval must be positive")
        self.path, self.interval, self.sampler = Path(path), interval, sampler
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="flyagent-gpu-monitor", daemon=True)

    def _run(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            while not self.stop.is_set():
                start = time.perf_counter()
                try:
                    sample = self.sampler()
                except Exception as exc:
                    sample = {"available": False, "devices": [], "error_type": type(exc).__name__, "error": str(exc)[:1000]}
                row = {"timestamp": time.time(), "monotonic_sec": start, **sample}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                self.stop.wait(max(0., self.interval - (time.perf_counter() - start)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()  # sampler subprocess has its own bounded timeout


def summarize_resources(directory):
    """Device totals and child RSS remain distinct; absent readings stay null."""
    directory = Path(directory)
    devices, errors, samples = {}, 0, 0
    path = directory / "gpu_resources.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue
        samples += 1
        errors += not row.get("available", False)
        for device in row.get("devices", []):
            summary = devices.setdefault(device["uuid"], {"name": device["name"], "peak_used_mib": None, "peak_gpu_percent": None})
            for source, target in (("used_mib", "peak_used_mib"), ("gpu_percent", "peak_gpu_percent")):
                value = device.get(source)
                if value is not None:
                    summary[target] = max(value, summary[target]) if summary[target] is not None else value
    rss, available = [], []
    path = directory / "resources.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("rss") is not None:
            rss.append(row["rss"])
        if row.get("system_available") is not None:
            available.append(row["system_available"])
    return {"gpu_samples": samples, "gpu_query_errors": errors, "device_totals": devices,
            "peak_child_rss_bytes": max(rss) if rss else None,
            "min_system_available_bytes": min(available) if available else None}
