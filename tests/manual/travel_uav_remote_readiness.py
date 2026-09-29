"""Read-only TravelUAV remote readiness audit.

The probe validates the compact-index selection and selected MapPackages, then
runs one read-only Python snapshot per remote endpoint.  It does not start or
stop Unreal, create a tunnel, load a model, or expose the private-key path.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
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
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

packages = json.loads(sys.argv[1])
settings_source = sys.argv[2]
settings_value = sys.argv[3]
remote_host = sys.argv[4]
remote_port = int(sys.argv[5])
settings = {}
settings_error = None
settings_bytes = b""
settings_path = None
try:
    if settings_source == "local":
        settings_bytes = base64.b64decode(settings_value.encode("ascii"))
    elif settings_source == "remote":
        settings_path = Path(settings_value)
        settings_bytes = settings_path.read_bytes()
    else:
        raise ValueError(f"unsupported settings source: {settings_source}")
    settings = json.loads(settings_bytes.decode("utf-8-sig"))
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
        "launch_method": item["launch_method"],
        "graphics_adapter": item.get("graphics_adapter"),
        "is_file": Path(item["executable"]).is_file(),
        "is_readable": os.access(item["executable"], os.R_OK),
        "is_executable": os.access(item["executable"], os.X_OK),
        "bash_syntax_ok": (
            subprocess.run(
                ["bash", "-n", item["executable"]],
                capture_output=True,
                text=True,
                check=False,
            ).returncode == 0
            if item["launch_method"] == "linux_bash" and Path(item["executable"]).is_file()
            else None
        ),
        "launchable": (
            Path(item["executable"]).is_file()
            and (
                os.access(item["executable"], os.R_OK)
                if item["launch_method"] == "linux_bash"
                else os.access(item["executable"], os.X_OK)
            )
        ),
    } for item in packages],
    "settings": {
        "source": settings_source,
        "template": settings_value if settings_source == "remote" else "local payload",
        "readable": bool(settings_bytes),
        "sha256": hashlib.sha256(settings_bytes).hexdigest() if settings_bytes else None,
        "error": settings_error,
        "sim_mode": settings.get("SimMode"),
        "api_server_port": settings.get("ApiServerPort"),
        "rendered_api_server_port": remote_port,
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
                                "launch_method": package.launch.get("method", "linux_bash"),
                                "graphics_adapter": _graphics_adapter(package),
                                "vehicle_name": package.airsim["vehicle_name"],
                                "camera_name": package.airsim["camera_name"],
                            }
                            for package in group
                        ],
                        separators=(",", ":"),
                    ),
                    *_settings_arguments(group[0]),
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
            package.airsim.get("settings_template_source", "local"),
            package.airsim["settings_template"],
        )
        groups.setdefault(key, []).append(package)
    return [groups[key] for key in sorted(groups)]


def _require_private_key(target: SshTarget) -> None:
    path = Path(os.path.expandvars(target.key_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError("SSH private key is not configured or missing")


def _settings_arguments(package: MapPackage) -> list[str]:
    """Return a read-only settings payload for the remote snapshot.

    A local template is validated from the repository and transported only as
    bytes for inspection.  The readiness probe never uploads it or creates a
    remote scene.  Remote templates retain the existing path-based behavior.
    """

    source = str(package.airsim.get("settings_template_source", "local"))
    template = str(package.airsim["settings_template"])
    if source == "remote":
        return [source, template]
    if source != "local":
        raise ValueError(f"Unsupported settings_template_source: {source}")
    path = Path(os.path.expandvars(template)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    payload = path.read_bytes()
    return [source, base64.b64encode(payload).decode("ascii")]


def _graphics_adapter(package: MapPackage) -> int | None:
    for argument in package.launch.get("args", []):
        match = re.fullmatch(r"-GraphicsAdapter=(\d+)", str(argument), re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _snapshot_errors(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    for scene in snapshot.get("scenes", []):
        if not scene.get("launchable"):
            errors.append({"stage": "scene_executable", **scene})
        elif scene.get("launch_method") == "linux_bash" and not scene.get(
            "bash_syntax_ok"
        ):
            errors.append({"stage": "scene_script_syntax", **scene})
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
        or (
            settings.get("source") == "remote"
            and settings.get("api_server_port") != endpoint.get("port")
        )
        or settings.get("rendered_api_server_port") != endpoint.get("port")
    ):
        errors.append({"stage": "capture_contract", **settings})
    if endpoint.get("listening") or snapshot.get("airsim_processes"):
        errors.append({"stage": "existing_process", **endpoint})
    gpu = snapshot.get("gpu", {})
    expected_gpu_indices = sorted(
        {
            int(scene["graphics_adapter"])
            for scene in snapshot.get("scenes", [])
            if scene.get("graphics_adapter") is not None
        }
    )
    available_gpu_indices = {
        int(str(row).split(",", 1)[0].strip())
        for row in gpu.get("rows", [])
        if str(row).split(",", 1)[0].strip().isdigit()
    }
    if not gpu.get("ok") or not set(expected_gpu_indices).issubset(
        available_gpu_indices
    ):
        errors.append(
            {
                "stage": "gpu",
                "expected_indices": expected_gpu_indices,
                "available_indices": sorted(available_gpu_indices),
                **gpu,
            }
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-id")
    parser.add_argument(
        "--split",
        choices=("test_seen", "test_unseen_map", "test_unseen_object"),
    )
    parser.add_argument("--difficulties", help="Comma-separated easy,hard")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--limit-per-difficulty", type=int)
    limits.add_argument("--all-per-difficulty", action="store_true")
    parser.add_argument("--case-ids", help="Comma-separated compact-index IDs")
    parser.add_argument("--output-dir", default="runs/experiments/travel_uav_remote_readiness")
    args = parser.parse_args()
    config_data = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.split:
        config_data["split"] = args.split
    if args.difficulties:
        config_data["difficulties"] = _csv(args.difficulties)
    if args.limit_per_difficulty is not None:
        config_data["limit_per_difficulty"] = args.limit_per_difficulty
    if args.all_per_difficulty:
        config_data["limit_per_difficulty"] = None
    if args.case_ids:
        config_data["case_ids"] = _csv(args.case_ids)
    result = inspect_remote_readiness(TravelUavEvalConfig.from_dict(config_data))
    run_id = args.eval_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "remote_readiness.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, indent=2, ensure_ascii=False))
    return 0 if result["ready"] else 2


def _csv(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Comma-separated option cannot be empty")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
