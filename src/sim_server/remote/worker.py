"""One-shot Linux worker for safely starting and stopping FlyAgent-owned UE.

The script is uploaded and invoked over SSH; it is neither installed nor kept
running as a service. Ownership is established with a registry containing PID,
process-group ID, executable, and Linux process start ticks.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _process_start_ticks(pid: int) -> int | None:
    try:
        # /proc/<pid>/stat field 22 follows a parenthesized command that may
        # contain spaces, so split only after the final closing parenthesis.
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields_after_comm = stat.rsplit(")", 1)[1].strip().split()
        return int(fields_after_comm[19])
    except (FileNotFoundError, IndexError, ValueError, PermissionError):
        return None


def _cmdline(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", errors="replace")
        )
    except (FileNotFoundError, PermissionError):
        return ""


def _identity_matches(registry: dict[str, Any]) -> bool:
    pid = int(registry["pid"])
    if _process_start_ticks(pid) != int(registry["process_start_ticks"]):
        return False
    return str(registry["executable"]) in _cmdline(pid)


def start_scene(spec_path: str) -> dict[str, Any]:
    spec = _read_json(spec_path)
    executable = Path(str(spec["executable"])).expanduser().resolve()
    settings_value = spec.get("settings_path")
    settings_path = (
        Path(str(settings_value)).expanduser().resolve()
        if settings_value
        else None
    )
    registry_path = Path(str(spec["registry_path"])).expanduser()
    log_path = Path(str(spec["log_path"])).expanduser()
    if not executable.is_file():
        raise FileNotFoundError(f"UE executable not found: {executable}")
    if settings_path is not None and not settings_path.is_file():
        raise FileNotFoundError(f"AirSim settings not found: {settings_path}")

    if registry_path.exists():
        registry = _read_json(registry_path)
        if registry.get("status") == "running" and _identity_matches(registry):
            return {"ok": True, **registry, "reused": True}

    args = [str(value) for value in spec.get("args", [])]
    style = str(spec.get("settings_arg_style", "double_dash"))
    if style == "double_dash":
        if settings_path is None:
            raise ValueError(
                "settings_arg_style=double_dash requires settings_path"
            )
        args.extend(["--settings", str(settings_path)])
    elif style == "single_dash_equals":
        if settings_path is None:
            raise ValueError(
                "settings_arg_style=single_dash_equals requires settings_path"
            )
        args.append(f"-settings={settings_path}")
    elif style == "none":
        pass
    else:
        raise ValueError(f"Unsupported settings_arg_style: {style}")

    method = str(spec.get("launch_method", "linux_bash"))
    command = (
        ["bash", str(executable), *args]
        if method == "linux_bash"
        else [str(executable), *args]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.update(
        {str(key): str(value) for key, value in spec.get("env", {}).items()}
    )
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(executable.parent),
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    start_ticks = _process_start_ticks(process.pid)
    if start_ticks is None:
        raise RuntimeError(f"Could not read identity for started process {process.pid}")
    registry = {
        "scene_instance_id": str(spec["scene_instance_id"]),
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "process_start_ticks": start_ticks,
        "executable": str(executable),
        "settings_path": str(settings_path) if settings_path is not None else None,
        "log_path": str(log_path),
        "status": "running",
    }
    _write_json(registry_path, registry)
    return {"ok": True, **registry, "reused": False}


def scene_status(registry_path: str) -> dict[str, Any]:
    path = Path(registry_path).expanduser()
    if not path.exists():
        return {"ok": True, "status": "not_found"}
    registry = _read_json(path)
    running = registry.get("status") == "running" and _identity_matches(registry)
    return {"ok": True, **registry, "status": "running" if running else "stopped"}


def stop_scene(registry_path: str, timeout_sec: float = 10.0) -> dict[str, Any]:
    path = Path(registry_path).expanduser()
    if not path.exists():
        return {"ok": True, "status": "not_found"}
    registry = _read_json(path)
    if registry.get("status") != "running":
        return {"ok": True, **registry, "status": "stopped"}
    actual_start_ticks = _process_start_ticks(int(registry["pid"]))
    if actual_start_ticks is None:
        registry["status"] = "stopped"
        _write_json(path, registry)
        return {"ok": True, **registry}
    if not _identity_matches(registry):
        raise RuntimeError(
            "Refusing to stop process because PID identity no longer matches registry"
        )

    pgid = int(registry["pgid"])
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and _identity_matches(registry):
        time.sleep(0.1)
    if _identity_matches(registry):
        os.killpg(pgid, signal.SIGKILL)
    registry["status"] = "stopped"
    _write_json(path, registry)
    return {"ok": True, **registry}


def probe_port(host: str, port: int, timeout_sec: float = 2.0) -> dict[str, Any]:
    """Check endpoint reachability from the Linux host."""

    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return {
                "ok": True,
                "status": "reachable",
                "host": host,
                "port": port,
            }
    except OSError as exc:
        return {
            "ok": True,
            "status": "unreachable",
            "host": host,
            "port": port,
            "error": str(exc),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlyAgent one-shot Linux UE worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--spec", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--registry", required=True)
    stop = subparsers.add_parser("stop")
    stop.add_argument("--registry", required=True)
    stop.add_argument("--timeout-sec", type=float, default=10.0)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--host", required=True)
    probe.add_argument("--port", required=True, type=int)
    probe.add_argument("--timeout-sec", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "start":
            result = start_scene(args.spec)
        elif args.command == "status":
            result = scene_status(args.registry)
        elif args.command == "stop":
            result = stop_scene(args.registry, args.timeout_sec)
        else:
            result = probe_port(args.host, args.port, args.timeout_sec)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
