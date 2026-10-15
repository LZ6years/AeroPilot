"""Drain exactly the active two leases before installing a reviewed recovery candidate.

The recorded controller is suspended while its host completes already admitted
leases. No new lease can be issued. The helper then resumes the controller with
an owned PAUSE marker so receipts/cleanup are committed before activation.
Use the flyagent interpreter, --output ROOT --candidate candidate_overlay.json.
"""
import argparse
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.manual.install_recovery_revision import activate, powershell
from tests.manual.image_budget_order_handoff import lease_requests, completed_requests, drain_deadlines
from evaluation.benchmarks.core.campaign import parallel as control


def handoff(output, candidate_path):
    candidate = control.read_json(candidate_path)
    directory = candidate_path.parent
    manifest = control.read_json(output/"manifest.json")
    control.verify_frozen(output, manifest)
    control.verify_overlay(candidate, manifest)
    for name, digest in candidate["execution_source_hashes"].items():
        if control.hashlib.sha256((Path(candidate["execution_source"])/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Candidate execution source changed")
    marker = output/"PAUSE"
    if marker.exists(): raise RuntimeError("Preserve user pause")
    owner = control.read_json(output/"dispatcher.json")
    parent = control.matching_process(owner)
    if not parent or "run" not in owner["command"]: raise RuntimeError("Controller not running")
    scheduled = control.read_json(output/"parallel/watchdog/scheduled_task.json")
    task = scheduled["task_name"]
    if not (directory/"handoff_guard_enabled.json").exists():
        powershell("-Command", "Disable-ScheduledTask -TaskName '"+task.replace("'", "''")+"' | Out-Null")
    watch = control.read_json(output/"parallel/watchdog/status.json").get("watchdog", {})
    watcher = control.matching_process(watch) if watch else None
    if watcher:
        watcher.terminate()
        watcher.wait(15)
    token = "Recovery handoff " + candidate["recovery_revision"]
    suspended = False
    record = {"status": "draining", "recovery_revision": candidate["recovery_revision"],
              "process": control.owner_of(control.psutil.Process()), "controller": owner}
    try:
        # A restarted helper may inherit the same intentionally suspended
        # controller. Do not increment Windows' suspension count twice.
        if parent.status() != control.psutil.STATUS_STOPPED:
            parent.suspend()
        suspended = True
        requests = lease_requests(output, control.read_json(output/"progress.json"))
        limits = drain_deadlines(output, requests)
        for limit in limits.values():
            process_path = limit["directory"] / "process.json"
            if process_path.exists():
                limit["started"] -= max(0, time.time()-process_path.stat().st_mtime)
        record["tasks"] = [r["task"] for _, r in requests]
        # Do not STOP until the host has admitted every persisted command.
        until = time.monotonic()+120
        while any(not completed_requests([(s, r)]) and r["lane"] not in
                  control.read_json(s/"host_status.json").get("active", []) for s,r in requests):
            if time.monotonic() >= until: raise TimeoutError("Host did not admit lease")
            time.sleep(1)
        for session in {s for s, _ in requests}:
            control.atomic_write_text(session/"STOP", token)
        while not completed_requests(requests):
            record["updated_at"] = control.utc_now()
            control.write_json(directory/"handoff_status.json", record)
            if marker.exists(): raise RuntimeError("User stopped the handoff")
            for session, request in requests:
                if completed_requests([(session, request)]): continue
                limit = limits[request["id"]]
                activity = control.activity_time(limit["directory"])
                if activity != limit["activity"]:
                    limit.update(activity=activity, progress_at=time.monotonic())
                reason = control.watchdog_reason(limit, time.monotonic(), manifest["policy"]["episode_deadline_sec"], manifest["policy"]["no_progress_sec"])
                if reason: raise TimeoutError(reason)
            time.sleep(5)
        control.atomic_write_text(marker, token)
        parent.resume()
        suspended = False
        parent.wait(300)
        if marker.read_text(encoding="utf-8") != token:
            raise RuntimeError("User pause superseded handoff")
        marker.unlink()
        activate(output, directory, candidate, manifest)
        record.update(status="installed", installed_at=control.utc_now())
        control.write_json(directory/"handoff_status.json", record)
    except BaseException as exc:
        if not marker.exists(): control.atomic_write_text(marker, token)
        if suspended:
            parent.resume()
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc), updated_at=control.utc_now())
        control.write_json(directory/"handoff_status.json", record)
        powershell("-Command", "Enable-ScheduledTask -TaskName '"+task.replace("'", "''")+"' | Out-Null")
        raise


def ensure(output, candidate_path):
    """Five-minute OS guard: restart only a missing owned handoff helper.

After a machine restart, both execution owners are absent. Installing the
candidate then lets the ordinary controller recover persisted abandoned leases;
the guard itself never scores, claims or deletes an attempt.
"""
    directory = candidate_path.parent
    record = control.read_json(directory/"handoff_status.json")
    owner = record.get("process")
    if owner and control.matching_process(owner):
        return "helper_alive"
    if record.get("status") == "failed" or (output/"PAUSE").exists():
        return "needs_attention"
    parent_owner = control.read_json(output/"dispatcher.json")
    parent = control.matching_process(parent_owner) if parent_owner else None
    if parent is None:
        host = control.read_json(output/"parallel/host_process.json")
        if host and control.matching_process(host):
            return "waiting_for_host_exit"
        candidate = control.read_json(candidate_path)
        activate(output, directory, candidate, control.read_json(output/"manifest.json"), allow_abandoned=True)
        return "installed_after_owner_exit"
    command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
               "--output", str(output), "--candidate", str(candidate_path)]
    guard_path = directory/"handoff_guard.json"
    guard = control.read_json(guard_path)
    if guard.get("restarts", 0) >= 3:
        return "restart_limit_requires_attention"
    control.write_json(guard_path, {"restarts": guard.get("restarts", 0)+1, "at": control.utc_now()})
    with (directory/"handoff_guard_stdout.log").open("a", encoding="utf-8") as out, \
         (directory/"handoff_guard_stderr.log").open("a", encoding="utf-8") as err:
        subprocess.Popen(command, stdout=out, stderr=err,
                         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    return "helper_restarted"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ensure", action="store_true")
    args = parser.parse_args()
    if args.ensure:
        print(ensure(args.output.resolve(), args.candidate.resolve()))
    else:
        handoff(args.output.resolve(), args.candidate.resolve())
