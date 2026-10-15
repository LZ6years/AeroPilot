"""Drain live image-budget lanes, then install the sequential frozen controller.

Run with the flyagent interpreter and --output CAMPAIGN. This operator helper
owns a separate transition lock. It suspends only the recorded queue writer,
leaves its flight host running, and uses the host's existing STOP protocol to
finish current tasks. The old controller then commits their original receipts
and cleans its scenes before a new controller/watchdog is installed. No flight
source, manifest or sampled identities are rewritten. An audited queue migration
excludes historical technical errors that an older controller treated as scores.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.aerial_vln import parallel_campaign as control
from evaluation.experiments.image_budget import repair_reporting


def lease_requests(output, state):
    """Resolve each live lease to its exact host command, never by episode alone."""
    requests = []
    for item in state["tasks"].values():
        if item["status"] != "running":
            continue
        attempt = item["attempts"][-1]
        session = Path(attempt["session"]).resolve()
        if (output / "parallel/sessions").resolve() not in session.parents:
            raise ValueError("Live lease has an unrelated host session")
        request = control.read_json(session / "commands" / (attempt["lane"] + ".json"))
        if request.get("task") != item["task"] or request.get("attempt") != attempt:
            raise ValueError("Live host command does not match its recorded lease")
        requests.append((session, request))
    return requests


def completed_requests(requests):
    """Require terminal host receipts, including exact lease identity."""
    for session, request in requests:
        fact = control.read_json(session / "results" / (request["id"] + ".json"))
        if not fact:
            return False
        if fact.get("task") != request["task"] or fact.get("attempt") != request["attempt"] or fact.get("thread_finished") is not True:
            raise ValueError("Completed host receipt does not match the lease")
    return True


def powershell(arguments):
    subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", *arguments],
                   cwd=ROOT, check=True)


def reconcile_errors(output, manifest):
    """Back up the stopped queue before applying an idempotent accounting fix."""
    for filename in ("dispatcher.json", "parallel/host_process.json"):
        owner = control.read_json(output / filename)
        if owner and control.matching_process(owner) is not None:
            raise RuntimeError("Reconciliation requires a stopped controller and host")
    with control.campaign_lock(output):
        before = (output / "progress.json").read_bytes()
        audit = output / "parallel/reconciliations" / uuid.uuid4().hex
        audit.mkdir(parents=True)
        (audit / "progress.before.json").write_bytes(before)
        queue = control.ParallelRecoveryQueue(output / "progress.json", manifest["tasks"], manifest["fingerprint"])
        changes = queue.reconcile_scored_errors()
        receipt = {"changes": changes, "before_sha256": hashlib.sha256(before).hexdigest(),
                   "after_sha256": hashlib.sha256((output / "progress.json").read_bytes()).hexdigest(),
                   "created_at": control.utc_now(), "raw_results_preserved": True}
        control.write_json(audit / "receipt.json", receipt)
        return receipt


def drain_deadlines(output, requests):
    """Preserve each already-running task's elapsed budget during suspension."""
    limits = {}
    for _, request in requests:
        directory = control._attempt_dir(output, request["task"], request["attempt"]["number"])
        process = control.read_json(directory / "process.json")
        activity = control.activity_time(directory)
        limits[request["id"]] = {"directory": directory, "started": time.monotonic() - process.get("elapsed_seconds", 0),
                                 "activity": activity, "progress_at": time.monotonic() - max(0, time.time() - activity / 1e9) if activity else time.monotonic()}
    return limits


def handoff(output, dataset_order=None):
    output = output.resolve()
    if Path(sys.executable).resolve() != Path(r"F:\conda_env\flyagent\python.exe").resolve():
        raise RuntimeError("This transition requires the flyagent interpreter")
    manifest = control.read_json(output / "manifest.json")
    if manifest.get("experiment") != "image_budget":
        raise ValueError("Expected the prepared image-budget experiment")
    control.verify_frozen(output, manifest)
    marker = output / "PAUSE"
    if marker.exists():
        raise RuntimeError("Preserve the existing manual pause")
    directory = output / "parallel/scheduling_handoff"
    directory.mkdir(parents=True, exist_ok=True)
    with control.campaign_lock(directory):
        owner = control.read_json(output / "dispatcher.json")
        overlay = control.read_json(output / "parallel/execution_overlay.json")
        parent = control.matching_process(owner)
        if parent is None or overlay["controller_script"] not in owner["command"] or "run" not in owner["command"] or str(output) not in owner["command"]:
            raise ValueError("The recorded image-budget controller is not running")
        scheduled = control.read_json(output / "parallel/watchdog/scheduled_task.json")
        expected_name = "FlyAgent_ImageBudget_" + output.name
        if scheduled.get("task_name") != expected_name:
            raise ValueError("Scheduled watchdog identity changed")
        # The scheduled launcher must not restart the previous watchdog during
        # this short, explicit operator transition. Restore it after freezing.
        task_name = expected_name.replace("'", "''")
        powershell(["-Command", f"Disable-ScheduledTask -TaskName '{task_name}' | Out-Null"])
        watch = control.read_json(output / "parallel/watchdog/status.json").get("watchdog", {})
        watcher = control.matching_process(watch) if watch else None
        if watcher is not None:
            if overlay["controller_script"] not in watch["command"] or "watch" not in watch["command"]:
                raise ValueError("Watchdog identity changed")
            # Stop only the watchdog process; an adopted controller is independent.
            watcher.terminate()
            watcher.wait(timeout=15)
        token = "Budget order transition " + uuid.uuid4().hex
        record = {"status": "draining", "interpreter": sys.executable, "controller": owner,
                  "budget_order": manifest["image_budgets"], "requested_at": control.utc_now()}
        control.write_json(directory / "process.json", control.owner_of(control.psutil.Process()))
        suspended = False
        try:
            parent.suspend()
            suspended = True
            requests = lease_requests(output, control.read_json(output / "progress.json"))
            limits = drain_deadlines(output, requests)
            host = control.read_json(output / "parallel/host_process.json")
            session = Path(host["session"])
            record["preserved_running_tasks"] = [request["task"] for _, request in requests]
            # A command queued just before suspension must enter the host before
            # STOP is written, otherwise it would never produce a receipt.
            until = time.monotonic() + 120
            while any(not control.read_json(s / "results" / (r["id"] + ".json")) and
                      r["lane"] not in control.read_json(s / "host_status.json").get("active", []) for s, r in requests):
                if time.monotonic() > until:
                    raise TimeoutError("The host did not admit its last recorded lease")
                time.sleep(1)
            control.atomic_write_text(session / "STOP", token + "\n")
            control.write_json(directory / "status.json", record)
            # The new watchdog understands intentional draining. It supervises
            # this helper immediately and does not treat the suspended old
            # queue writer as a stale controller. Lane catalog bytes are equal;
            # the already running host retains its original loaded flight code.
            control.prepare_extension(output, manifest, dataset_order=dataset_order)
            powershell(["-File", str(ROOT / "tests/manual/install_aerial_vln_watchdog.ps1"), "-OutputRoot", str(output), "-Workers", "2"])
            while not completed_requests(requests):
                record["updated_at"] = control.utc_now()
                control.write_json(directory / "status.json", record)
                for saved_session, request in requests:
                    if completed_requests([(saved_session, request)]):
                        continue
                    limit = limits[request["id"]]
                    activity = control.activity_time(limit["directory"])
                    if activity != limit["activity"]:
                        limit.update(activity=activity, progress_at=time.monotonic())
                    reason = control.watchdog_reason(limit, time.monotonic(), manifest["policy"]["episode_deadline_sec"], manifest["policy"]["no_progress_sec"])
                    if reason:
                        raise TimeoutError("Drain reached existing task limit: " + reason)
                if control.matching_process(host) is None or marker.exists():
                    raise RuntimeError("Flight host stopped or transition was interrupted before terminal receipts")
                time.sleep(5)
            # PAUSE is owned by this transition. With no unfinished flights,
            # the previous controller safely commits results before it exits.
            if marker.exists():
                raise RuntimeError("A user pause appeared during the transition")
            control.atomic_write_text(marker, token + "\n")
            parent.resume()
            suspended = False
            parent.wait(timeout=300)
            state = control.read_json(output / "progress.json")
            if state.get("reason") not in {None, "manual_interrupt"} or any(
                    item["status"] == "running" for item in state["tasks"].values()):
                raise RuntimeError("The old controller did not finish verified cleanup")
            record["status"] = "installing"
            control.write_json(directory / "status.json", record)
            record["reconciliation"] = reconcile_errors(output, manifest)
            receipt = repair_reporting(output)
            overlay = control.read_json(output / "parallel/execution_overlay.json")
            expected_mode = "budget_dataset_sequential" if dataset_order else "budget_sequential"
            if overlay.get("scheduling_policy", {}).get("mode") != expected_mode:
                raise RuntimeError("The new controller is missing sequential budget scheduling")
            if marker.read_text(encoding="utf-8").strip() != token:
                raise RuntimeError("Preserve the changed manual pause marker")
            marker.unlink()
            control.atomic_write_text(output / "parallel/watchdog/RESUME", token + "\n")
            powershell(["-File", str(ROOT / "tests/manual/install_aerial_vln_watchdog.ps1"), "-OutputRoot", str(output), "-Workers", "2"])
            record.update(status="complete", completed_at=control.utc_now(), repair_receipt=receipt,
                          scheduling_policy=overlay["scheduling_policy"])
            control.write_json(directory / "status.json", record)
        except BaseException as exc:
            # Never leave a silently suspended writer if this helper fails.
            if suspended and control.matching_process(owner) is not None:
                if not marker.exists():
                    control.atomic_write_text(marker, token + "\n")
                parent.resume()
            record.update(status="needs_attention", error_type=type(exc).__name__, error=str(exc), updated_at=control.utc_now())
            control.write_json(directory / "status.json", record)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-order", nargs="+", choices=("aerial_vln", "travel_uav"))
    args = parser.parse_args()
    handoff(args.output, args.dataset_order)
