"""Read-only resource and remote executable checks before the mixed campaign.

No scene is started/stopped here. On success, freeze lane transport/GPU bindings
using the existing controller utility; actual navigation is watchdog-owned.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket


def preflight(output):
    from evaluation.benchmarks.aerial_vln.parallel_campaign import (
        REMOTE_GPU_INSPECTION, discover_gpu_bindings, prepare_extension, read_json,
        verify_frozen, write_json,
    )
    from evaluation.benchmarks.aerial_vln.campaign import environment_diagnostics, provider_probe
    from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
    from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
    from tests.manual.aerial_vln_remote_readiness import inspect_remote_readiness as aerial_readiness
    from tests.manual.travel_uav_remote_readiness import inspect_remote_readiness as travel_readiness
    from src.runtime.configs.map_config import load_map_config
    from src.sim_server.remote.control import RemoteSceneControl

    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    if manifest.get("experiment") != "image_budget" or manifest.get("strict_workers") != 2:
        raise ValueError("A frozen two-lane image-budget experiment is required")
    for split, data in manifest["configs"].items():
        if not data.get("save_artifacts"):
            raise ValueError("Original image artifacts are required for lightweight request-media auditing")
    verify_frozen(output, manifest)
    environment = environment_diagnostics()
    busy, validations = [], {}
    for port in (35001, 35002):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                busy.append(port)
    for split, data in manifest["configs"].items():
        check = (travel_readiness(TravelUavEvalConfig.from_dict(data)) if split.startswith("test_")
                 else aerial_readiness(AerialVlnEvalConfig.from_dict(data)))
        validations[split] = check
        write_json(output / "preflight" / (split + ".json"), check)
    map_path = manifest["configs"]["val_seen"]["map_config_path"]
    packages, _ = load_map_config(map_path)
    control = RemoteSceneControl()
    target = control.target(packages[0])
    gpu = json.loads(control.runner.run(target, ["python3", "-c", REMOTE_GPU_INSPECTION, "[]"], timeout_sec=30))
    remote = control.runner.run(target, ["ss", "-ltnp"], timeout_sec=15)
    conflicts = [line for line in remote.splitlines() if any(
        address.rsplit(":", 1)[-1] in {"30001", "30002"} for address in line.split()[3:4])]
    compute = [p for card in gpu["gpus"] if card["index"] in {0, 2} for p in card["processes"] if "C" in p["type"]]
    record = {"environment": environment, "local_busy_ports": busy, "remote_conflicts": conflicts,
              "remote_compute_jobs": compute, "gpu": gpu, "remote_ready": {k: v["ready"] for k, v in validations.items()},
              "passed": bool(environment["cuda_available"] and not busy and not conflicts and not compute and all(v["ready"] for v in validations.values()))}
    write_json(output / "preflight.json", record)
    if not record["passed"]:
        raise RuntimeError("Resource/remote preflight failed; existing processes left intact")
    # Small provider probe is a separate diagnostic, outside the 600 metrics.
    if not provider_probe(manifest["configs"]["val_seen"]["env_path"], output / "provider_validation.json", service_check=True):
        raise RuntimeError("Decision provider unavailable; experiment not launched")
    overlay = prepare_extension(output, manifest)
    return {"passed": True, "output": str(output), "controller_script": overlay["controller_script"],
            "environment": environment, "workers": 2}
