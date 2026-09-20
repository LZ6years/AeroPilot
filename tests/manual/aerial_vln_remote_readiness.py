"""Read-only readiness audit for selected AerialVLN-S remote scenes.

The probe reuses production dataset classification and MapPackage routing, but
it never creates a model client, Runtime, SimulationServer, SSH tunnel, worker,
or simulator process.  Its output is safe to retain beside manual test runs and
does not include the resolved SSH private-key path.
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

from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
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
    for item in captures
    if isinstance(item, dict)
}

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    port_listening = sock.connect_ex((remote_host, remote_port)) == 0
finally:
    sock.close()

gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
    check=False,
)
processes = subprocess.run(
    # The bracketed first character is the conventional pgrep self-match
    # guard: the regex still matches the Unreal process name, while the
    # readiness Python command line contains the literal ``[A]`` form and is
    # therefore excluded from its own result.
    ["pgrep", "-af", "[A]irVLN-Linux-Shipping"],
    capture_output=True,
    text=True,
    check=False,
)
registry_root = Path.home() / ".flyagent" / "scenes"

payload = {
    "python": sys.version.split()[0],
    "scenes": [
        {
            "scene_id": item["scene_id"],
            "map_id": item["map_id"],
            "executable": item["executable"],
            "launch_method": item["launch_method"],
            "graphics_adapter": item.get("graphics_adapter"),
            "expected_mode": item["expected_mode"],
            "control_strategy": item["control_strategy"],
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
                if item["launch_method"] == "linux_bash"
                and Path(item["executable"]).is_file()
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
        }
        for item in packages
    ],
    "settings": {
        "source": settings_source,
        "template": settings_value if settings_source == "remote" else "local payload",
        "readable": bool(settings_bytes),
        "sha256": (
            hashlib.sha256(settings_bytes).hexdigest()
            if settings_bytes
            else None
        ),
        "error": settings_error,
        "sim_mode": settings.get("SimMode"),
        "api_server_port": settings.get("ApiServerPort"),
        "rendered_api_server_port": remote_port,
        "vehicle_present": bool(vehicle),
        "camera_present": bool(camera),
        "capture_sizes": capture_sizes,
    },
    "remote_endpoint": {
        "host": remote_host,
        "port": remote_port,
        "listening": port_listening,
    },
    "gpu": {
        "ok": gpu.returncode == 0,
        "rows": [line for line in gpu.stdout.splitlines() if line.strip()],
        "error": gpu.stderr.strip() or None,
    },
    "airsim_processes": [
        line for line in processes.stdout.splitlines() if line.strip()
    ],
    "historical_registry_count": (
        len(list(registry_root.glob("*/scene-registry.json")))
        if registry_root.is_dir()
        else 0
    ),
}
print(json.dumps(payload))
""".strip()


def inspect_remote_readiness(
    config: AerialVlnEvalConfig,
    *,
    ssh_runner: SshCommandRunner | None = None,
) -> dict[str, Any]:
    """Return dataset-bound, read-only remote execution readiness."""

    benchmark = AerialVlnBenchmarkRunner(config)
    prepared = benchmark.prepare_evaluation(require_execution_ready=True)
    packages = [
        benchmark.map_packages_by_id[config.scene_map[scene_id]]
        for scene_id in prepared.selection.selected_scene_ids
    ]
    groups = _package_groups(packages)
    runner = ssh_runner or SshCommandRunner()
    snapshots = []
    errors = []
    for group in groups:
        target = RemoteSceneControl.target(group[0])
        try:
            _require_private_key(target)
            package_payload = [
                {
                    "scene_id": package.scene_id,
                    "map_id": package.map_id,
                    "executable": str(package.launch["linux_executable"]),
                    "launch_method": str(
                        package.launch.get("method", "linux_bash")
                    ),
                    "graphics_adapter": _graphics_adapter(package),
                    "expected_mode": str(package.airsim["mode"]),
                    "control_strategy": str(
                        package.airsim["control_strategy"]
                    ),
                    "vehicle_name": str(package.airsim["vehicle_name"]),
                    "camera_name": str(package.airsim["camera_name"]),
                }
                for package in group
            ]
            output = runner.run(
                target,
                [
                    "python3",
                    "-c",
                    _REMOTE_SNAPSHOT,
                    json.dumps(package_payload, separators=(",", ":")),
                    *_settings_arguments(group[0]),
                    str(group[0].airsim.get("remote_host", "127.0.0.1")),
                    str(group[0].airsim["port"]),
                ],
                timeout_sec=max(
                    float(package.ssh.get("connect_timeout_sec", 10))
                    for package in group
                ),
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


def _package_groups(
    packages: list[MapPackage],
) -> list[list[MapPackage]]:
    grouped: dict[
        tuple[str, int, str, str, int, str, str, str, str], list[MapPackage]
    ] = {}
    for package in packages:
        key = (
            str(package.ssh["host"]),
            int(package.ssh.get("port", 22)),
            str(package.ssh["user"]),
            str(package.airsim.get("remote_host", "127.0.0.1")),
            int(package.airsim["port"]),
            str(package.airsim.get("settings_template_source", "local")),
            str(package.airsim["settings_template"]),
            str(package.airsim["mode"]),
            str(package.airsim["control_strategy"]),
        )
        grouped.setdefault(key, []).append(package)
    return [grouped[key] for key in sorted(grouped)]


def _require_private_key(target: SshTarget) -> None:
    path = Path(os.path.expandvars(target.key_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError("SSH private key is not configured or missing")


def _settings_arguments(package: MapPackage) -> list[str]:
    """Return settings to the read-only remote snapshot without uploading.

    Local templates are inspected from the repository and transported as an
    in-memory base64 payload. Remote templates retain their path contract.
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
    """Read the configured GPU index from the Unreal launch arguments."""

    for argument in package.launch.get("args", []):
        match = re.fullmatch(
            r"-GraphicsAdapter=(\d+)", str(argument), re.IGNORECASE
        )
        if match:
            return int(match.group(1))
    return None


def _snapshot_errors(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
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
        settings.get("sim_mode")
        not in {
            scene.get("expected_mode")
            for scene in snapshot.get("scenes", [])
        }
        or not settings.get("vehicle_present")
        or not settings.get("camera_present")
    ):
        errors.append({"stage": "settings_contract", **settings})
    endpoint = snapshot.get("remote_endpoint", {})
    capture_sizes = settings.get("capture_sizes", {})
    rgb_size = capture_sizes.get("0", [0, 0])
    depth_size = capture_sizes.get("2", [0, 0])
    if (
        len(rgb_size) != 2
        or any(int(value or 0) <= 0 for value in rgb_size)
        or len(depth_size) != 2
        or any(int(value or 0) <= 0 for value in depth_size)
        or (
            settings.get("source") == "remote"
            and settings.get("api_server_port") != endpoint.get("port")
        )
        or settings.get("rendered_api_server_port") != endpoint.get("port")
    ):
        errors.append({"stage": "capture_contract", **settings})
    if endpoint.get("listening"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-id", default=None)
    parser.add_argument(
        "--output-dir",
        default="tests/manual_outputs/aerial_vln_remote_readiness",
    )
    args = parser.parse_args()
    config = AerialVlnEvalConfig.from_file(args.config)
    result = inspect_remote_readiness(config)
    run_id = args.eval_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "remote_readiness.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output_path), **result},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
