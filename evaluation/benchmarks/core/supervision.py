"""Isolated serial child supervision and verified cleanup of owned scenes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from src.runtime.configs.map_config import load_map_config
from src.sim_server.remote.control import RemoteSceneControl
from evaluation.benchmarks.core.report_io import write_json


def supervise(command, directory, *, timeout=7200, no_progress=900, stop=lambda: False):
    """Survive native exits; interruption kills only this child, then caller cleans."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    descendants = {}
    started = progress_at = time.monotonic()
    previous_activity = None
    reason = None
    with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, \
            (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"},
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            created = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess:
            created = None
        owner = {"pid": process.pid, "created": created, "command": command}
        write_json(directory / "process.json", {**owner, "descendants": [], "status": "running"})
        try:
            while process.poll() is None:
                now = time.monotonic()
                try:
                    root = psutil.Process(process.pid)
                    for child in root.children(recursive=True):
                        descendants[child.pid] = {"pid": child.pid, "created": child.create_time(), "command": child.cmdline()}
                except psutil.Error:
                    pass
                activity = max((path.stat().st_mtime_ns for path in (directory / "runs").rglob("*")
                                if path.is_file() and path.suffix in {".jsonl", ".png", ".json"}), default=0)
                if activity != previous_activity:
                    progress_at = now
                    previous_activity = activity
                if stop():
                    reason = "manual_interrupt"
                elif now - started > timeout:
                    reason = "episode_deadline"
                elif now - progress_at > no_progress:
                    reason = "no_progress"
                if reason:
                    process.kill()
                    break
                write_json(directory / "process.json", {**owner, "descendants": list(descendants.values()),
                                                        "status": "running", "elapsed_seconds": now - started})
                time.sleep(2)
        except KeyboardInterrupt:
            reason = "manual_interrupt"
            process.kill()
        code = process.wait()
    result = {**owner, "exit_code": code, "reason": reason,
              "elapsed_seconds": time.monotonic() - started, "descendants": list(descendants.values())}
    write_json(directory / "process.json", result)
    return result


def cleanup_owned(run_root, map_config, descendants, *, control=None):
    """Delegate lifecycle to scene control after matching persisted ownership."""
    packages, _ = load_map_config(map_config)
    if packages and all(p.connection.get("mode") == "local" for p in packages):
        from evaluation.benchmarks.core.campaign.local_execution import cleanup_local
        return cleanup_local(run_root)
    control = control or RemoteSceneControl()
    packages = {package.map_id: package for package in packages}
    receipts = []
    for path in Path(run_root).glob("*/state/scene_instance.json"):
        scene = json.loads(path.read_text(encoding="utf-8"))
        if scene.get("process_ownership") != "remote_managed":
            raise ValueError("Campaign requires managed scene ownership")
        package = packages[scene["map_id"]]
        scene_id = scene["scene_instance_id"]
        status = control.status(package, scene_id)
        if status.get("pid") != scene.get("remote_process_id") or status.get("scene_instance_id") != scene_id:
            raise ValueError("Remote ownership mismatch; cleanup cannot be verified")
        stopped = control.stop(package, scene_id)
        settings = json.loads((path.parents[1] / "scenes" / scene_id / "settings.json").read_text(encoding="utf-8"))
        endpoint = control.probe_prepared_port(package, scene_id, "127.0.0.1", settings["ApiServerPort"])
        tunnels = []
        obsolete_tunnels = []
        for child in descendants:
            command = child["command"]
            if not command or Path(command[0]).stem.lower() != "ssh" or "-L" not in command:
                continue
            if not any(arg.startswith(f'127.0.0.1:{scene["control_port"]}:') for arg in command):
                continue
            try:
                process = psutil.Process(child["pid"])
                if process.create_time() != child["created"] or process.cmdline() != command:
                    # Reused PIDs do not belong to this scene. Never signal
                    # them; require independent endpoint absence below.
                    obsolete_tunnels.append(child["pid"])
                    continue
                process.terminate()
                process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
            tunnels.append(child["pid"])
        from src.sim_server.remote.recovery import _closed_local_port
        receipts.append({"scene": scene_id, "remote_stop": stopped, "endpoint": endpoint, "tunnels": tunnels,
                         "obsolete_tunnel_pids": obsolete_tunnels,
                         "local_endpoint_closed": _closed_local_port("127.0.0.1", scene["control_port"])})
    return receipts


def cleanup_verified(receipts):
    """Empty cleanup is not proof that a launched scene was stopped."""
    return bool(receipts) and all(item["remote_stop"].get("cleanup_verified") is True
                                  and item["endpoint"].get("status") == "unreachable"
                                  and item.get("local_endpoint_closed", True) for item in receipts)
