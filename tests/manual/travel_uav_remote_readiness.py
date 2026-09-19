"""Read-only TravelUAV remote readiness audit.

The probe validates the compact-index selection and selected MapPackages, then
runs one read-only Python snapshot per remote endpoint.  It does not start or
stop Unreal, create a tunnel, load a model, or expose the private-key path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner
from src.env_client import SceneInstance
from src.env_client.schemas import MapPackage
from src.sim_server.remote.control import RemoteSceneControl
from src.sim_server.remote.ssh import SshCommandRunner, SshTarget


_REMOTE_SNAPSHOT = r"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

packages = json.loads(sys.argv[1])
settings_path = Path(sys.argv[2])
remote_host = sys.argv[3]
remote_port = int(sys.argv[4])
settings = {}
settings_error = None
try:
    settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
except Exception as exc:
    settings_error = f"{type(exc).__name__}: {exc}"
vehicle_name = packages[0]["vehicle_name"] if packages else ""
camera_name = packages[0]["camera_name"] if packages else ""
vehicle = settings.get("Vehicles", {}).get(vehicle_name, {})
camera = vehicle.get("Cameras", {}).get(camera_name, {})
captures = camera.get("CaptureSettings", [])
capture_sizes = {
    str(item.get("ImageType")): [item.get("Width"), item.get("Height")]
    for item in captures if isinstance(item, dict)
}
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    listening = sock.connect_ex((remote_host, remote_port)) == 0
finally:
    sock.close()
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    capture_output=True, text=True, check=False,
)
process_rows = []
scene_tokens = (
    "ue4",
    "carlaue4",
    "nycenvironment",
    "modular",
    "tropical",
    "moderncity",
)
for process_dir in Path("/proc").iterdir():
    if not process_dir.name.isdigit():
        continue
    try:
        executable = os.path.basename(os.readlink(process_dir / "exe"))
    except (FileNotFoundError, PermissionError, OSError):
        continue
    if not any(token in executable.lower() for token in scene_tokens):
        continue
    try:
        command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        ).strip()
    except (FileNotFoundError, PermissionError, OSError):
        command = executable
    process_rows.append(f"{process_dir.name} {executable} {command}")
print(json.dumps({
    "python": sys.version.split()[0],
    "scenes": [{
        "scene_id": item["scene_id"],
        "map_id": item["map_id"],
        "executable": item["executable"],
        "is_file": Path(item["executable"]).is_file(),
        "is_executable": os.access(item["executable"], os.X_OK),
    } for item in packages],
    "settings": {
        "readable": os.access(settings_path, os.R_OK),
        "sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest() if settings_path.is_file() else None,
        "error": settings_error,
        "sim_mode": settings.get("SimMode"),
        "api_server_port": settings.get("ApiServerPort"),
        "vehicle_present": bool(vehicle),
        "camera_present": bool(camera),
        "capture_sizes": capture_sizes,
    },
    "remote_endpoint": {"host": remote_host, "port": remote_port, "listening": listening},
    "gpu": {"ok": gpu.returncode == 0, "rows": [line for line in gpu.stdout.splitlines() if line.strip()]},
    "airsim_processes": sorted(process_rows),
}))
""".strip()


def inspect_remote_readiness(
    config: TravelUavEvalConfig,
    *,
    ssh_runner: SshCommandRunner | None = None,
) -> dict[str, Any]:
    benchmark = TravelUavBenchmarkRunner(config)
    prepared = benchmark.prepare_evaluation(require_execution_ready=True)
    packages = [
        benchmark.map_packages_by_id[config.scene_map[map_name]]
        for map_name in prepared.selection.selected_map_names
    ]
    runner = ssh_runner or SshCommandRunner()
    snapshots = []
    errors = []
    for group in _package_groups(packages):
        target = RemoteSceneControl.target(group[0])
        try:
            _require_private_key(target)
            output = runner.run(
                target,
                [
                    "python3",
                    "-c",
                    _REMOTE_SNAPSHOT,
                    json.dumps(
                        [
                            {
                                "scene_id": package.scene_id,
                                "map_id": package.map_id,
                                "executable": package.launch["linux_executable"],
                                "vehicle_name": package.airsim["vehicle_name"],
                                "camera_name": package.airsim["camera_name"],
                            }
                            for package in group
                        ],
                        separators=(",", ":"),
                    ),
                    str(group[0].airsim["settings_template"]),
                    str(group[0].airsim.get("remote_host", "127.0.0.1")),
                    str(group[0].airsim["port"]),
                ],
                timeout_sec=max(float(package.ssh.get("connect_timeout_sec", 10)) for package in group),
            )
            snapshot = json.loads(output.strip().splitlines()[-1])
            snapshots.append(snapshot)
            errors.extend(_snapshot_errors(snapshot))
        except Exception as exc:
            errors.append(
                {
                    "stage": "ssh_readiness",
                    "host": target.host,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "ready": not errors,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "dataset": prepared.provenance.to_dict(),
        "case_selection": prepared.selection.to_dict(),
        "execution_readiness": prepared.execution_readiness,
        "remote_snapshots": snapshots,
        "errors": errors,
    }


def _package_groups(packages: list[MapPackage]) -> list[list[MapPackage]]:
    groups: dict[tuple[Any, ...], list[MapPackage]] = {}
    for package in packages:
        key = (
            package.ssh["host"],
            int(package.ssh.get("port", 22)),
            package.ssh["user"],
            package.airsim.get("remote_host", "127.0.0.1"),
            int(package.airsim["port"]),
            package.airsim["settings_template"],
        )
        groups.setdefault(key, []).append(package)
    return [groups[key] for key in sorted(groups)]


def _require_private_key(target: SshTarget) -> None:
    path = Path(os.path.expandvars(target.key_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError("SSH private key is not configured or missing")


def _snapshot_errors(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    for scene in snapshot.get("scenes", []):
        if not scene.get("is_file") or not scene.get("is_executable"):
            errors.append({"stage": "scene_executable", **scene})
    settings = snapshot.get("settings", {})
    if not settings.get("readable") or settings.get("error"):
        errors.append({"stage": "settings", **settings})
    elif (
        settings.get("sim_mode") != "Multirotor"
        or not settings.get("vehicle_present")
        or not settings.get("camera_present")
    ):
        errors.append({"stage": "settings_contract", **settings})
    endpoint = snapshot.get("remote_endpoint", {})
    captures = settings.get("capture_sizes", {})
    if (
        captures.get("0") != [256, 256]
        or captures.get("2") != [256, 256]
        or settings.get("api_server_port") != endpoint.get("port")
    ):
        errors.append({"stage": "capture_contract", **settings})
    if endpoint.get("listening") or snapshot.get("airsim_processes"):
        errors.append({"stage": "existing_process", **endpoint})
    gpu = snapshot.get("gpu", {})
    if not gpu.get("ok") or not any(
        str(row).split(",", 1)[0].strip() == "0" for row in gpu.get("rows", [])
    ):
        errors.append({"stage": "gpu0", **gpu})
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-id")
    parser.add_argument("--output-dir", default="tests/manual_outputs/travel_uav_remote_readiness")
    args = parser.parse_args()
    result = inspect_remote_readiness(TravelUavEvalConfig.from_file(args.config))
    run_id = args.eval_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "remote_readiness.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, indent=2, ensure_ascii=False))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
