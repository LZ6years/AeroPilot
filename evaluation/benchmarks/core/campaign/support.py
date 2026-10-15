"""Frozen campaign resources, locking and dataset-independent utilities."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import psutil
from src.agent.config import load_model_config
from src.llm import DefaultModelClient, ModelRequest, ProviderRequestError
from evaluation.benchmarks.core.report_io import write_json
from evaluation.benchmarks.core.campaign.contracts import TaskIdentity
from evaluation.benchmarks.core.campaign.recovery import RecoveryQueue
ROOT = Path(__file__).resolve().parents[4]

def environment_diagnostics():
    """Verify the user-required interpreter and actual CUDA environment."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        torch = None
    dependencies = {}
    for name in ("openai", "httpx2", "transformers", "numpy", "pillow", "psutil"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    info = {"sys_executable": sys.executable, "torch_version": torch.__version__ if torch else None,
            "torch_cuda": torch.version.cuda if torch else None,
            "cuda_available": bool(torch and torch.cuda.is_available()), "dependencies": dependencies}
    if os.name == "nt" and Path(sys.executable).resolve() != Path("F:/conda_env/flyagent/python.exe").resolve():
        raise RuntimeError("Campaign requires the explicit flyagent interpreter")
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_tensor_probe"] = float(torch.ones(1, device="cuda").item())
    return info

def provider_probe(env_path, output, *, injected_tls=0, service_check=False):
    """Small real request; optional deterministic TLS faults before real recovery."""
    settings = load_model_config(env_path).to_provider_settings()
    if service_check:
        settings = replace(settings, timeout_sec=30, retry_deadline_sec=30,
                           provider_connection_retries=0, provider_timeout_retries=0)
    client = DefaultModelClient(settings)
    trace = {}
    if injected_tls:
        import ssl
        original = client._request_chat_stream if settings.stream else client._request_chat
        calls = 0

        def fault_then_request(*args):
            nonlocal calls
            calls += 1
            if calls <= injected_tls:
                raise ssl.SSLEOFError("local recovery-validation TLS fault")
            return original(*args)

        if settings.stream:
            client._request_chat_stream = fault_then_request
        else:
            client._request_chat = fault_then_request
    try:
        result = client.request(ModelRequest([{"role": "user", "content": "Reply with exactly OK."}]), transport_trace=trace)
        record = {"success": bool(result.text and result.text.strip()), "transport": trace,
                  "injected_tls_failures": injected_tls, "model": settings.model_name}
    except ProviderRequestError as exc:
        record = {"success": False, "error": exc.to_dict(), "transport": trace}
    finally:
        client._reset_connection()
    write_json(output, record)
    return record["success"]

@contextmanager
def campaign_lock(output):
    """OS releases the lock on crash; only one host dispatcher can own the queue."""
    path = Path(output) / "campaign.lock"
    with path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)

def storage_check(output, queue, tasks):
    """Budget retained bytes twice plus one observed temporary peak; keep 10 GiB."""
    sizes = []
    peaks = []
    for item in queue.state["tasks"].values():
        for attempt in item["attempts"]:
            if attempt.get("retained_bytes"):
                sizes.append(attempt["retained_bytes"])
                peaks.append(attempt.get("temporary_depth_bytes", 0))
    remaining = sum(queue.state["tasks"][RecoveryQueue.key(task)]["status"] != "scored" for task in tasks)
    free = shutil.disk_usage(output).free
    # Before the pilot there is no empirical estimate; require the minimum.
    per_case = max(sizes) if sizes else 0
    required = max(10 * 1024**3, remaining * per_case * 2 + max(peaks, default=0))
    record = {"free_bytes": free, "required_bytes": required, "remaining_cases": remaining,
              "observed_max_retained_bytes": per_case, "observed_cases": len(sizes), "passed": free >= required}
    write_json(Path(output) / "storage_check.json", record)
    return record["passed"]

def _attempt_dir(output, task, number):
    return TaskIdentity.from_task(task).attempt_dir(Path(output), number)


def freeze_source(output, *, root=ROOT, plugin_config=None):
    """Copy the actual worktree, excluding outputs, models and credentials."""
    source = Path(output) / "source"
    for folder in ("src", "plugins", "configs", "evaluation", "memory", "tests/manual"):
        for directory, folders, files in os.walk(Path(root) / folder):
            folders[:] = [f for f in folders if f not in {"outputs", "__pycache__", ".git", "node_modules"}]
            for name in files:
                path = Path(directory) / name
                if path.suffix not in {".py", ".json", ".md", ".txt", ".ps1"}:
                    continue
                target = source / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in ("AGENT.md", "AGENTS.md", "pyproject.toml", "requirements.txt", "tests/__init__.py"):
        path = Path(root) / name
        if path.is_file():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if plugin_config is not None:
        shutil.copy2(plugin_config, source / "configs/plugins.json")
    return source
