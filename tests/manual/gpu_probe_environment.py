"""Fail-closed environment and cached-weight checks for offline GPU probes.

Diagnostic only: no package installation, downloads, model loading, precision
changes or CPU fallback. CUDA inspection runs in experiment children so the
supervising parent does not retain a CUDA context during measurements.
"""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys

FLYAGENT_PYTHON = Path("F:/conda_env/flyagent/python.exe")
PACKAGES = ("torch", "transformers", "numpy", "Pillow", "psutil", "huggingface-hub")


def digest(path):
    digest_value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def check_interpreter(expected=FLYAGENT_PYTHON, actual=None):
    actual = Path(actual or sys.executable).resolve()
    if actual != Path(expected).resolve():
        raise ValueError(f"Wrong interpreter: {actual}; expected {expected}")
    return str(actual)


def cache_sources(settings):
    """Freeze complete local snapshots, including tokenizer/processor assets."""
    from huggingface_hub import snapshot_download
    records = {}
    for key in ("grounding_model", "segmentation_model"):
        model = settings[key]
        path = Path(model)
        if not path.is_dir():
            path = Path(snapshot_download(model, local_files_only=True))
        files = {str(p.resolve()): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}
        if not files or not any(p.suffix in {".bin", ".safetensors"} for p in path.rglob("*")):
            raise ValueError(f"No cached weights: {model}")
        records[key] = {"model": model, "snapshot": str(path.resolve()), "files": files}
    return records


def verify_weights(records):
    for record in records.values():
        if any(not Path(p).is_file() or digest(p) != value for p, value in record["files"].items()):
            raise ValueError("Frozen cached model assets changed")


def environment(directory, *, require_cuda=False, expected=None):
    """Caller persists the returned record even if validate_environment rejects."""
    import torch
    available = torch.cuda.is_available()
    return {"executable": str(Path(sys.executable).resolve()), "python": sys.version,
            "dependencies": {name: importlib.metadata.version(name) for name in PACKAGES},
            "cuda_build": torch.version.cuda, "cuda_available": available,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else [],
            "disk_free_bytes": shutil.disk_usage(directory).free,
            "require_cuda": require_cuda}


def validate_environment(record, *, require_cuda=False, expected=None):
    if require_cuda:
        check_interpreter(actual=record["executable"])
        if not record["cuda_build"] or not record["cuda_available"]:
            raise ValueError("CUDA required but unavailable; no CPU fallback")
    if expected:
        check_interpreter(expected["executable"], record["executable"])
        for key in ("dependencies", "cuda_build", "cuda_available", "gpu_names"):
            if record[key] != expected[key]:
                raise ValueError(f"Child environment changed: {key}")
    if record["disk_free_bytes"] < 5 * 1024 ** 3:
        raise ValueError("Less than 5 GiB free for offline artifacts")


def other_project_jobs(workspace):
    """Read process identity only; never stop another experiment's processes."""
    import psutil
    excluded = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            args = process.info["cmdline"] or []
            if process.pid in excluded or not str(process.info["name"]).lower().startswith("python"):
                continue
            command = " ".join(args).replace("\\", "/").lower()
            cwd = Path(process.cwd()).resolve()
            if (cwd.is_relative_to(workspace) or str(workspace).replace("\\", "/").lower() in command) and any(
                    token in command for token in ("tests/manual/", "pytest", "evaluation/benchmarks")):
                found.append({"pid": process.pid, "created": process.create_time(), "executable": process.exe()})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found
