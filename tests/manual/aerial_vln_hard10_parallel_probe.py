"""Fixed hard-ten evaluation using the existing supervised parallel controller.

prepare freezes the current optional-plan experiment; admit checks captured and
real API evidence; run delegates all leases, cleanup and two-wave gating to the
shared controller. report reads its native progress without a second writer.
All subprocesses inherit the explicit flyagent interpreter used to launch this.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from tests.manual import aerial_vln_plan_context_probe as probe
from evaluation.benchmarks.aerial_vln.parallel_recovery import ParallelRecoveryQueue

HARD_CASES = WORKSPACE / "tests/manual/aerial_vln_hard_instruction10.json"
PRIMARY = Path("H:/FlyAgent-Eval/avln_place_memory_tools_20_20261012_142206")
SECONDARY = Path("H:/FlyAgent-Eval/avln_pixel_place_plan_20_20261011_190946")


def configure_parallel(root, secondary):
    """Migrate only an untouched preparation to the native single-writer queue."""
    manifest = probe.read(root / "manifest.json")
    state = probe.read(root / "progress.json")
    if any(row["status"] != "pending" or row["attempts"] for row in state["cases"]):
        raise ValueError("Only an untouched preparation can become parallel")
    expected = probe.read(HARD_CASES)
    if [task["episode_id"] for task in manifest["tasks"]] != expected or len(expected) != 10:
        raise ValueError("Fixed ordered hard ten required")
    manifest.update(execution_mode="parallel", terminal_provider_errors=True, request_audit_gate=True,
        configs={"val_seen": manifest["config"]}, counts={"val_seen": 10, "val_unseen": 0},
        technical_trial=manifest["tasks"][:4], technical_gate_indices=list(range(4)),
        secondary_root=str(secondary))
    for cohort in manifest["cohorts"]:
        cohort["secondary_baseline"] = str(secondary)
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    probe.write(root / "manifest.json", manifest)
    # This is a one-time pre-execution schema conversion, not a writable report
    # projection. Preserve the original preparation as an immutable receipt.
    (root / "progress.json").rename(root / "preparation_receipt.json")
    ParallelRecoveryQueue(root / "progress.json", manifest["tasks"], manifest["fingerprint"],
                          terminal_provider_errors=True)
    return manifest


def prepare(root, primary=PRIMARY, secondary=SECONDARY):
    root = probe.prepare(root, probe.DEFAULT_BASELINE, HARD_CASES, previous_root=primary,
        plugins_config=WORKSPACE / "configs/plugins.pixel_map.json", hard_case_ids_file=HARD_CASES,
        hard_only=True, plan_mode="optional", reference_roots=(secondary,))
    configure_parallel(root, secondary)
    probe.report(root)
    return root


def resource_check(root):
    """Read occupancy without stopping any unrelated process or experiment."""
    from evaluation.benchmarks.aerial_vln.parallel_campaign import REMOTE_GPU_INSPECTION, discover_gpu_bindings
    from src.runtime.configs.map_config import load_map_config
    from src.sim_server.remote.control import RemoteSceneControl
    manifest = probe.guard(root)
    probe.load_credentials(manifest)
    busy = []
    for port in (35001, 35002):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                busy.append(port)
    config = manifest["config"]["map_config_path"]
    packages, _ = load_map_config(config)
    control = RemoteSceneControl()
    target = control.target(packages[0])
    gpu = json.loads(control.runner.run(target, ["python3", "-c", REMOTE_GPU_INSPECTION, "[]"], timeout_sec=30))
    remote = control.runner.run(target, ["ss", "-ltnp"], timeout_sec=15)
    conflicts = [line for line in remote.splitlines() if any(
        address.rsplit(":", 1)[-1] in {"30001", "30002"} for address in line.split()[3:4])]
    # Display graphics contexts are normal; compute jobs or existing simulator
    # listeners on the selected resources prevent this experiment from starting.
    compute = [process for card in gpu["gpus"] if card["index"] in {0, 2}
               for process in card["processes"] if "C" in process["type"]]
    record = {"local_busy_ports": busy, "remote_conflicts": conflicts, "gpu": gpu,
              "bindings": discover_gpu_bindings(config), "environment": probe.environment_diagnostics(),
              "passed": not busy and not conflicts and not compute}
    probe.write(root / "parallel_resource_check.json", record)
    if not record["passed"]:
        raise RuntimeError("Selected resources are occupied; no other experiment was stopped")
    return record


def run(root):
    manifest = probe.guard(root)
    if not probe.admission_passed(probe.read(root / "admission.json"), manifest["plan_mode"]):
        raise RuntimeError("Real interface admission must pass before navigation")
    for filename in ("historical_request_replay.json", "pixel_place_admission.json"):
        if not probe.read(root / filename).get("passed"):
            raise RuntimeError("Required admission failed: " + filename)
    if manifest.get("place_memory_admission_required") and probe.read(root / "place_memory_admission.json").get("status") != "passed":
        raise RuntimeError("Place memory admission must pass before navigation")
    resource_check(root)
    # A separately frozen controller repair may resume an interrupted campaign
    # without replacing navigation source or replaying terminal outcomes.
    from evaluation.benchmarks.aerial_vln.pixel_map_campaign import controller_entry
    controller = controller_entry(root) if (root / "parallel/execution_overlay.json").exists() else root / "source/evaluation/benchmarks/aerial_vln/parallel_campaign.py"
    subprocess.run([sys.executable, "-u", str(controller), "run", "--output", str(root), "--workers", "2"], check=True)
    probe.report(root)
    state = probe.read(root / "progress.json")
    if state["status"] != "seen_complete":
        raise RuntimeError(f'Parallel experiment incomplete: {state["status"]} / {state.get("reason")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "admit", "run", "report"))
    parser.add_argument("--root", type=Path, default=Path("H:/FlyAgent-Eval/avln_hard10_parallel_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        print(prepare(root))
    elif args.command == "admit":
        resource_check(root)
        probe.admit(root)
    elif args.command == "run":
        run(root)
    else:
        probe.report(root)


if __name__ == "__main__":
    main()
