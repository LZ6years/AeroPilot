"""Build/review an allowlisted recovery snapshot, then activate on a stopped campaign.

Usage (flyagent Python): install_recovery_revision.py --output ROOT [--activate].
Without --activate this only creates a candidate and a unified diff. Activation
requires both recorded execution processes to have exited; it preserves all
receipts, backs up the queue/overlay, audits scored errors and installs the
existing hidden Windows watchdog. It never changes the original source tree.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evaluation.benchmarks.core.campaign import parallel as control

RUNTIME_FILES = (
    "src/llm/recovery.py", "src/llm/client.py", "src/llm/providers/openai_compatible.py",
    "src/sim_server/remote/ssh.py", "src/sim_server/remote/worker.py",
    "evaluation/benchmarks/core/supervision.py", "evaluation/benchmarks/travel_uav/runner.py",
)
CONTROLLER_FILES = {
    "parallel_campaign.py": "evaluation/benchmarks/core/campaign/parallel.py",
    "campaign/parallel.py": "evaluation/benchmarks/core/campaign/parallel.py",
    "parallel_recovery.py": "evaluation/benchmarks/core/campaign/recovery.py",
    "campaign/recovery.py": "evaluation/benchmarks/core/campaign/recovery.py",
    "campaign/cleanup.py": "evaluation/benchmarks/core/campaign/cleanup.py",
    "compression_cost.py": "evaluation/experiments/compression_cost.py",
    "image_budget_report.py": "evaluation/experiments/image_budget_report.py",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(output):
    manifest = control.read_json(output / "manifest.json")
    control.verify_frozen(output, manifest)
    old = control.read_json(output / "parallel/execution_overlay.json")
    hashes = {name: sha((REPO/name).read_bytes()) for name in RUNTIME_FILES}
    changes = {name: sha((REPO/path).read_bytes()) for name, path in CONTROLLER_FILES.items()}
    revision = sha(json.dumps({"runtime": hashes, "controller": changes}, sort_keys=True).encode())
    directory = output / "parallel/recovery_revisions" / revision[:16]
    source = directory / "source"
    directory.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        shutil.copytree(output / "source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    diff = []
    for relative in RUNTIME_FILES:
        original = output / "source" / relative
        data = (REPO/relative).read_bytes()
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        diff.extend(difflib.unified_diff(original.read_text(encoding="utf-8").splitlines(True) if original.exists() else [],
                    data.decode("utf-8").splitlines(True), fromfile="original/"+relative, tofile="recovery/"+relative))
    (directory/"recovery.patch").write_text("".join(diff), encoding="utf-8")
    expected = {**manifest["source_hashes"], **hashes}
    for name, digest in expected.items():
        if sha((source/name).read_bytes()) != digest:
            raise ValueError("Unexpected source change: " + name)
    old_controller = Path(old["controller_script"]).parent
    controller_hashes = {**old["source_hashes"], **changes}
    controller_revision = sha(json.dumps(controller_hashes, sort_keys=True).encode())
    controller = output / "parallel/controller_source" / controller_revision[:16]
    for name, digest in controller_hashes.items():
        data = (REPO/CONTROLLER_FILES[name]).read_bytes() if name in CONTROLLER_FILES else (old_controller/name).read_bytes()
        if sha(data) != digest:
            raise ValueError("Controller input changed")
        target = controller/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Immutable controller already exists with different bytes")
        target.write_bytes(data)
    candidate = {**old, "controller_revision": controller_revision, "source_hashes": controller_hashes,
        "controller_script": str(controller/"parallel_campaign.py"), "execution_source": str(source),
        "execution_source_hashes": expected, "execution_source_sha256": sha(json.dumps(expected, sort_keys=True).encode()),
        "recovery_revision": revision, "recovery_policy": {"max_attempts": 6, "deadline_sec": 600},
        "recovery_validation_required": True}
    control.write_json(directory/"candidate_overlay.json", candidate)
    return directory, candidate, manifest


def powershell(*args):
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", *args],
                   check=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)


def activate(output, directory, candidate, manifest, *, allow_abandoned=False):
    control.verify_frozen(output, manifest)
    control.verify_overlay(candidate, manifest)
    for name, digest in candidate["execution_source_hashes"].items():
        if sha((Path(candidate["execution_source"])/name).read_bytes()) != digest:
            raise ValueError("Candidate source hash mismatch")
    for name in ("dispatcher.json", "parallel/host_process.json"):
        owner = control.read_json(output/name)
        if owner and control.matching_process(owner):
            raise RuntimeError("Drain existing flights before activation")
    task = "FlyAgent_ImageBudget_" + output.name
    powershell("-Command", "Disable-ScheduledTask -TaskName '"+task.replace("'", "''")+"' | Out-Null")
    owner = control.read_json(output/"parallel/watchdog/status.json").get("watchdog")
    process = control.matching_process(owner) if owner else None
    if process:
        if "watch" not in owner["command"] or str(output) not in owner["command"]:
            raise RuntimeError("Watchdog ownership mismatch")
        process.terminate()
        process.wait(15)
    with control.campaign_lock(output):
        if not allow_abandoned and any(v["status"] == "running" for v in control.read_json(output/"progress.json")["tasks"].values()):
            raise RuntimeError("Outstanding lease must finish before activation")
        for filename in ("progress.json", "parallel/execution_overlay.json"):
            backup = directory / (Path(filename).name + ".before")
            if not backup.exists():
                backup.write_bytes((output/filename).read_bytes())
        queue = control.ParallelRecoveryQueue(output/"progress.json", manifest["tasks"], manifest["fingerprint"])
        changes = queue.reconcile_scored_errors()
        resumed_reason = None
        if queue.state.get("reason") == "non_retryable_provider_error":
            cause = queue.state.get("pause_cause") or {}
            item = queue.state["tasks"].get(cause.get("task_key"))
            error = queue.provider_error(item["attempts"][-1].get("result")) if item and item.get("attempts") else {}
            message = str(error.get("error", ""))
            from src.llm.recovery import service_error
            if message.startswith("model provider SSE error:") and service_error(message.split(":", 1)[1]).retryable:
                resumed_reason = queue.state["reason"]
                queue.state.update(status="paused", reason="manual_interrupt")
                queue.state.pop("pause_cause", None)
                queue.save()
        control.write_json(directory/"score_audit.json", {"changes": changes, "audited_tasks": len(queue.state["tasks"]),
                           "reclassified_pause_reason": resumed_reason,
                           "raw_receipts_preserved": True, "at": control.utc_now()})
        control.write_json(output/"parallel/execution_overlay.json", candidate)
        control.atomic_write_text(output/"parallel/watchdog/RESUME", "Approved recovery revision " + candidate["recovery_revision"])
    powershell("-File", str(REPO/"tests/manual/install_campaign_watchdog.ps1"), "-OutputRoot", str(output), "-Workers", "2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path(r"F:\conda_env\flyagent\python.exe").resolve():
        raise RuntimeError("Use the flyagent interpreter")
    directory, candidate, manifest = build(args.output.resolve())
    if args.activate:
        activate(args.output.resolve(), directory, candidate, manifest)
    print(json.dumps({"directory": str(directory), "recovery_revision": candidate["recovery_revision"], "activated": args.activate}))
