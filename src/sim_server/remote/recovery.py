"""Read-only cleanup proof for failed remote scene startup.

Startup can fail before a ready SceneInstance is persisted. An absent registry
alone is insufficient: require rollback records, no launch spec, no matching
remote process, a closed RPC endpoint and no surviving owned local transport.
When a spec was written, require the completed rollback and a fresh stopped
owned-group registry instead; a spec alone never proves cleanup.
This module never stops processes, launches a scene or changes evaluation state.
"""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import socket

import psutil

from .control import RemoteSceneControl


# Run directly over SSH because a failure may precede worker upload. Inspect
# only the remote user's processes; exclude this probe and its SSH ancestors.
_REMOTE_INSPECTION = """import errno,json,os,socket,sys
from pathlib import Path
scene,directory,port=sys.argv[1],Path(sys.argv[2]),int(sys.argv[3])
ancestors=set(); pid=os.getpid()
while pid>1 and pid not in ancestors:
 ancestors.add(pid)
 pid=int(Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[1])
matches=[]; unreadable=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name) in ancestors: continue
 try:
  if proc.stat().st_uid!=os.getuid(): continue
  args=(proc/'cmdline').read_bytes().split(bytes([0]))
  if any(scene.encode() in arg for arg in args): matches.append(int(proc.name))
 except FileNotFoundError: pass
 except OSError: unreadable.append(int(proc.name))
with socket.socket() as connection:
 connection.settimeout(3)
 endpoint_code=connection.connect_ex(('127.0.0.1',port))
files={name:(directory/name).exists() for name
 in ('scene-registry.json','scene-spec.json','settings.json')}
documents={name:json.loads((directory/name).read_text()) for name
 in ('scene-registry.json','scene-spec.json') if files[name]}
print(json.dumps({'scene':scene,'files':{name:(directory/name).exists() for name
 in ('scene-registry.json','scene-spec.json','settings.json')},
 'documents':documents,'resolved_paths':{'settings':str((directory/'settings.json').resolve()),
 'executable':str(Path(documents.get('scene-spec.json',{}).get('executable','.')).expanduser().resolve())},
 'matching_processes':matches,'unreadable_processes':unreadable,
 'endpoint_connect_ex':endpoint_code,'endpoint_closed':endpoint_code==errno.ECONNREFUSED}))
"""


def _closed_local_port(host, port):
    # Windows loopback connect_ex may time out even after a tunnel exits. Its
    # authoritative TCP table proves absence without interpreting that timeout
    # as refusal. An unreadable table or any surviving endpoint is not proof.
    if os.name == "nt":
        if host not in {"127.0.0.1", "::1", "localhost"}:
            return False
        try:
            return not any(connection.laddr and connection.laddr.port == port
                           for connection in psutil.net_connections(kind="tcp"))
        except (psutil.Error, OSError):
            return False
    with socket.socket() as connection:
        connection.settimeout(3)
        code = connection.connect_ex((host, port))
    return code in {errno.ECONNREFUSED, 10061}


def verify_unlaunched_scene(package, scene_dir, processes, *, control=None,
                            completed_attempt=False, local_endpoint=None,
                            recheck_cleanup=False):
    """Return fresh proof, or a refusal, without treating a dead parent as proof.

    ``processes`` contains the supervisor's PID/create-time/command receipts.
    An unknown or partially launched scene must use normal owned-group cleanup;
    this narrow proof only applies before any launch spec was written.

    A parallel controller may supply completed_attempt only after its future
    has returned and the lane cannot be leased again. In that mode the shared
    host is not owned by this attempt: inspect only its own transports and the
    explicitly reserved local endpoint, including failure before tunnel open.

    ``recheck_cleanup`` permits a completed lane to replace an old transport
    failure of the stop command with fresh absence evidence. This applies only
    before any launch spec/process exists, and never edits historical rollback
    errors. All other rollback, ownership and endpoint checks still apply.
    """
    scene_dir = Path(scene_dir)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scene": scene_dir.name, "kind": "startup_not_launched_verified", "verified": False}

    def refuse(reason):
        return {**record, "reason": reason}

    events = [json.loads(line) for line in (scene_dir / "lifecycle.jsonl").read_text(encoding="utf-8").splitlines()]
    record["lifecycle_stages"] = [{"stage": event["stage"], "status": event["status"]} for event in events]
    if any(event["stage"] in {"process_started", "scene_ready"} for event in events):
        return refuse("launch_may_have_started")
    if (scene_dir / "remote-scene-spec.json").exists():
        return refuse("launch_spec_exists")
    failure = next((event for event in reversed(events) if event["stage"] == "startup_failed"), None)
    if failure is None:
        return refuse("startup_rollback_not_complete")
    cleanup_errors = failure.get("cleanup_errors")
    if cleanup_errors is None:
        cleanup_errors = []
    if not isinstance(cleanup_errors, list):
        return refuse("startup_rollback_not_complete")
    if cleanup_errors:
        transient = {"TimeoutError", "TimeoutExpired", "ConnectionError", "ConnectionResetError",
                     "ConnectionAbortedError", "ConnectionRefusedError", "BrokenPipeError", "SSLEOFError"}
        if not (recheck_cleanup and completed_attempt and local_endpoint
                and isinstance(cleanup_errors, list)
                and all(isinstance(error, dict) and error.get("stage") == "owned_process_stopped"
                        and error.get("error_type") in transient and error.get("retryable") is not False
                        for error in cleanup_errors)
                and any(event["stage"] == "backend_closed" and event["status"] == "ok" for event in events)):
            return refuse("startup_rollback_not_complete")
        record.update(kind="startup_not_launched_rechecked", cleanup_recheck_errors=cleanup_errors)
    stages = ("start_requested", "port_reservation_released") if completed_attempt else (
        "start_requested", "tunnel_closed", "port_reservation_released")
    for stage in stages:
        if not any(event["stage"] == stage and event["status"] == "ok" for event in events):
            return refuse("startup_rollback_not_complete")
    if not processes and not (completed_attempt and local_endpoint):
        return refuse("local_ownership_receipt_missing")
    local = []
    tunnels = []
    settings_path = scene_dir / "settings.json"
    if settings_path.exists():
        remote_port = int(json.loads(settings_path.read_text(encoding="utf-8"))["ApiServerPort"])
    elif completed_attempt and local_endpoint:
        remote_port = int(package.airsim["port"])
    else:
        return refuse("settings_endpoint_missing")
    for receipt in processes:
        process = None
        try:
            process = psutil.Process(receipt["pid"])
            alive = process.create_time() == receipt["created"] and process.cmdline() == receipt["command"]
        except psutil.NoSuchProcess:
            alive = False
        local.append({"pid": receipt["pid"], "original_process_alive": alive})
        if alive:
            record["local_processes"] = local
            return refuse("owned_local_process_alive")
        command = receipt["command"]
        if command and Path(command[0]).stem.lower() == "ssh" and "-L" in command:
            forward = command[command.index("-L") + 1].split(":")
            if len(forward) != 4 or forward[0] != "127.0.0.1" or forward[2] != "127.0.0.1" or int(forward[3]) != remote_port:
                return refuse("tunnel_endpoint_unknown")
            tunnels.append({"host": forward[0], "port": int(forward[1]),
                            "closed": _closed_local_port(forward[0], int(forward[1]))})
    if completed_attempt and local_endpoint:
        host, port = local_endpoint
        if host != "127.0.0.1":
            return refuse("local_endpoint_unknown")
        tunnels.append({"host": host, "port": port, "closed": _closed_local_port(host, port),
                        "kind": "reserved_attempt_endpoint"})
    record.update(local_processes=local, local_tunnels=tunnels)
    if not tunnels or not all(tunnel["closed"] for tunnel in tunnels):
        return refuse("local_tunnel_not_verified_closed")
    control = control or RemoteSceneControl()
    paths = control._paths(package, scene_dir.name)
    output = control.runner.run(control.target(package),
        ["python3", "-c", _REMOTE_INSPECTION, scene_dir.name, paths.directory, str(remote_port)], timeout_sec=15)
    remote = json.loads(output)
    record["remote"] = remote
    if remote["scene"] != scene_dir.name or any(remote["files"].values()):
        return refuse("remote_launch_state_exists")
    if remote["matching_processes"] or remote["unreadable_processes"]:
        return refuse("remote_process_absence_unverified")
    if not remote["endpoint_closed"]:
        return refuse("remote_endpoint_not_verified_closed")
    return {**record, "verified": True, "reason": None}


def verify_failed_launch(package, scene_dir, processes, *, control=None,
                         completed_attempt=False, local_endpoint=None):
    """Prove an already rolled-back launch without signalling any process.

    A timed-out SSH start can still have launched a simulator and written its
    remote registry before the local SceneInstance exists. The scene owner
    performs rollback. Its completed records, the immutable launch spec and a
    fresh worker-owned group inspection must all agree before a controller may
    reuse the lane. Incomplete rollback or missing ownership remains a pause.
    """
    scene_dir = Path(scene_dir)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scene": scene_dir.name, "kind": "startup_launched_rollback_verified", "verified": False}

    def refuse(reason):
        return {**record, "reason": reason}

    if not completed_attempt or not local_endpoint or local_endpoint[0] != "127.0.0.1":
        return refuse("completed_lane_ownership_missing")
    events = [json.loads(line) for line in (scene_dir / "lifecycle.jsonl").read_text(encoding="utf-8").splitlines()]
    record["lifecycle_stages"] = [{"stage": event["stage"], "status": event["status"]} for event in events]
    failure = next((event for event in reversed(events) if event["stage"] == "startup_failed"), None)
    rollback = ("start_requested", "backend_closed", "owned_process_stopped", "tunnel_closed",
                "port_reservation_released")
    if failure is None or failure.get("cleanup_errors") or any(
            not any(event["stage"] == stage and event["status"] == "ok" for event in events)
            for stage in rollback):
        return refuse("startup_rollback_not_complete")
    if any(event["stage"] == "scene_ready" for event in events):
        return refuse("scene_was_ready_use_normal_cleanup")
    spec = json.loads((scene_dir / "remote-scene-spec.json").read_text(encoding="utf-8"))
    control = control or RemoteSceneControl()
    paths = control._paths(package, scene_dir.name)
    expected = {"scene_instance_id": scene_dir.name, "executable": str(package.launch["linux_executable"]),
                "registry_path": paths.registry, "settings_path": paths.settings}
    if any(spec.get(key) != value for key, value in expected.items()):
        return refuse("local_launch_identity_mismatch")
    port = int(json.loads((scene_dir / "settings.json").read_text(encoding="utf-8"))["ApiServerPort"])
    record["local_processes"] = []
    for receipt in processes:
        try:
            process = psutil.Process(receipt["pid"])
            alive = process.create_time() == receipt["created"] and process.cmdline() == receipt["command"]
        except psutil.NoSuchProcess:
            alive = False
        record["local_processes"].append({"pid": receipt["pid"], "original_process_alive": alive})
        if alive:
            return refuse("owned_local_process_alive")
    record["local_endpoint"] = {"host": local_endpoint[0], "port": local_endpoint[1],
                                "closed": _closed_local_port(*local_endpoint)}
    if not record["local_endpoint"]["closed"]:
        return refuse("local_tunnel_not_verified_closed")
    # Worker status checks live group membership and PID/start-tick identity.
    # Never substitute an old stopped receipt or an absent script parent.
    status = control.status(package, scene_dir.name)
    record["remote_status"] = status
    if (status.get("ok") is not True or status.get("status") != "stopped"
            or status.get("cleanup_verified") is not True or status.get("live_members") != []):
        return refuse("owned_remote_group_not_verified_stopped")
    identity = ("scene_instance_id", "pid", "pgid", "process_start_ticks", "executable", "settings_path")
    if status.get("scene_instance_id") != scene_dir.name or any(
            not isinstance(status.get(key), int) or status[key] <= 0
            for key in ("pid", "pgid", "process_start_ticks")):
        return refuse("remote_registry_identity_missing")
    output = control.runner.run(control.target(package),
        ["python3", "-c", _REMOTE_INSPECTION, scene_dir.name, paths.directory, str(port)], timeout_sec=15)
    remote = json.loads(output)
    record["remote"] = remote
    documents = remote.get("documents", {})
    registry = documents.get("scene-registry.json", {})
    if (remote.get("scene") != scene_dir.name or documents.get("scene-spec.json") != spec
            or any(registry.get(key) != status.get(key) for key in identity)
            or registry.get("status") != "stopped" or registry.get("cleanup_verified") is not True
            or registry.get("live_members") != []):
        return refuse("remote_launch_identity_mismatch")
    resolved = remote.get("resolved_paths", {})
    if status["executable"] != resolved.get("executable") or status["settings_path"] != resolved.get("settings"):
        return refuse("remote_launch_identity_mismatch")
    if remote.get("matching_processes") or remote.get("unreadable_processes"):
        return refuse("remote_process_absence_unverified")
    if remote.get("endpoint_closed") is not True:
        return refuse("remote_endpoint_not_verified_closed")
    return {**record, "verified": True, "reason": None}
