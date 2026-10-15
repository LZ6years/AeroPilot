"""Verify a recorded pre-navigation import failure without touching simulator processes.

This repair is restricted to the recovery rollout's missing ssh_observer import.
It preserves every failed receipt and attempt budget and requires stopped owners,
completed host facts, no run directories, and fresh closed local/remote ports.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.benchmarks.core.campaign import parallel as control
from src.sim_server.remote.control import RemoteSceneControl
from src.sim_server.remote.recovery import _closed_local_port


def audit(output, session):
    control.verify_frozen(output, control.read_json(output/"manifest.json"))
    for name in ("dispatcher.json", "parallel/host_process.json"):
        owner = control.read_json(output/name)
        if owner and control.matching_process(owner):
            raise RuntimeError("Execution must be stopped")
    with control.campaign_lock(output):
        queue = control.ParallelRecoveryQueue(output/"progress.json", control.read_json(output/"manifest.json")["tasks"],
                                            control.read_json(output/"manifest.json")["fingerprint"])
        backup = session/"progress.before_bootstrap_audit.json"
        if not backup.exists(): backup.write_bytes((output/"progress.json").read_bytes())
        changes = []
        for item in queue.state["tasks"].values():
            if not item["attempts"]: continue
            attempt = item["attempts"][-1]
            if attempt.get("session") != str(session) or attempt.get("cleanup_verified"): continue
            command = control.read_json(session/"commands"/(attempt["lane"]+".json"))
            fact = control.read_json(session/"results"/(command["id"]+".json"))
            error = fact.get("exception") or {}
            if (fact.get("thread_finished") is not True or fact["task"] != item["task"] or
                fact["attempt"]["number"] != attempt["number"] or error.get("error_type") != "ImportError" or
                "cannot import name 'ssh_observer'" not in error.get("error", "")):
                raise ValueError("Not the verified pre-navigation import failure")
            directory = Path(attempt["directory"])
            if list((directory/"runs").glob("*")) or (directory/"execution_phase.json").exists():
                raise ValueError("Unexpected navigation evidence")
            packages, _ = control.load_map_config(attempt["map_config_path"])
            package = packages[0]
            remote = RemoteSceneControl()
            code = "import socket,errno,json,sys; s=socket.socket(); s.settimeout(3); c=s.connect_ex(('127.0.0.1',int(sys.argv[1]))); print(json.dumps({'closed':c==errno.ECONNREFUSED,'code':c}))"
            remote_proof = json.loads(remote.runner.run(remote.target(package), ["python3", "-c", code, str(attempt["remote_port"])], timeout_sec=15))
            proof = {"local_closed": _closed_local_port("127.0.0.1", attempt["local_port"]),
                     "remote": remote_proof, "host_receipt": str(session/"results"/(command["id"]+".json")),
                     "no_navigation_started": True, "checked_at": control.utc_now()}
            if not proof["local_closed"] or not remote_proof["closed"]:
                raise RuntimeError("Endpoint absence was not verified")
            control.write_json(directory/"bootstrap_cleanup_proof.json", proof)
            attempt.update(cleanup_verified=True, valid_score=False, status="infrastructure_failed",
                           recovery_revision="rollout_bootstrap_failure", bootstrap_cleanup_proof=proof)
            item["status"] = "retry_pending" if len(item["attempts"]) < queue.attempt_limit(item) else "retry_exhausted"
            changes.append(queue.key(item["task"]))
        if changes:
            queue.state.update(status="paused", reason="manual_interrupt")
            queue.save()
        control.write_json(session/"bootstrap_audit.json", {"changes": changes, "at": control.utc_now()})
        print(json.dumps({"verified_pre_navigation_failures": changes}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output.resolve(), args.session.resolve())
