"""GPU monitoring must preserve unknowns and survive unavailable drivers."""

import json
from types import SimpleNamespace
import subprocess
import sys
import threading
from tests.manual.gpu_resource_monitor import sample_gpu, GpuResourceMonitor


def test_gpu_csv_unknown_values_are_not_zero():
    result = sample_gpu(run=lambda *a, **k: SimpleNamespace(returncode=0, stderr="", stdout="0, GPU-id, GPU name, 6144, 2000, 4144, N/A, [Not Supported]\n"))
    assert result["available"]
    assert result["devices"][0]["gpu_percent"] is None
    assert result["devices"][0]["used_mib"] == 2000


def test_missing_driver_and_timeout_are_records():
    for error in (FileNotFoundError("no driver"), subprocess.TimeoutExpired("nvidia-smi", 2)):
        def fail(*a, **k):
            raise error
        result = sample_gpu(run=fail)
        assert not result["available"] and result["error_type"] == type(error).__name__


def test_monitor_preserves_sample_and_stops(tmp_path):
    sampled = threading.Event()
    def sampler():
        sampled.set()
        return {"available": False, "devices": []}
    with GpuResourceMonitor(tmp_path / "gpu.jsonl", sampler=sampler) as monitor:
        assert sampled.wait(3)
    assert not monitor.thread.is_alive()
    assert json.loads((tmp_path / "gpu.jsonl").read_text().splitlines()[0])["available"] is False


def test_parent_retains_gpu_log_after_native_child_exit(tmp_path, monkeypatch):
    import tests.manual.gpu_resource_monitor as gpu
    from tests.manual.process_diagnostics import supervise
    original = gpu.GpuResourceMonitor
    monkeypatch.setattr(gpu, "GpuResourceMonitor", lambda path: original(path, sampler=lambda: {"available": False}))
    status = supervise([sys.executable, "-c", "import os; os._exit(9)"], tmp_path / "native", monitor_gpu=True)
    assert status["exit_code"] == 9
    assert (tmp_path / "native/gpu_resources.jsonl").is_file()
