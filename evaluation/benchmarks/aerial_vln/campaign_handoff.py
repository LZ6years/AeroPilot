"""Legacy serial-to-parallel operator handoff for AerialVLN campaigns."""
from __future__ import annotations
import psutil, time, uuid
from pathlib import Path
from evaluation.benchmarks.core.campaign.parallel import (read_json, owner_of, matching_process, completed_boundary, atomic_write_text, write_json, utc_now, campaign_lock)

def handoff(args):
    """Briefly suspend only our dispatcher to prove a safe no-flight boundary."""
    output = args.output.resolve()
    if (output / "PAUSE").exists():
        raise RuntimeError("An existing manual pause must be preserved")
    dispatcher = read_json(output / "dispatcher.json")
    parent = psutil.Process(dispatcher["pid"])
    owner = owner_of(parent)
    if "evaluation.benchmarks.aerial_vln.campaign" not in owner["command"] or "run" not in owner["command"] \
            or str(output) not in owner["command"]:
        raise ValueError("Only this campaign's serial dispatcher may be handed off")
    observer_info = read_json(output / "handoff_control.json")
    observer = psutil.Process(observer_info["pid"])
    observer_owner = owner_of(observer)
    if str(output) not in observer_owner["command"] or not any(
            Path(arg).name == "aerial_vln_pilot_handoff.py" for arg in observer_owner["command"]):
        raise ValueError("Observer ownership mismatch")
    observer.terminate()
    observer.wait(timeout=15)
    token = "Parallel handoff " + uuid.uuid4().hex
    print(json.dumps({"handoff_waiting": True, "dispatcher_pid": parent.pid}), flush=True)
    while matching_process(owner) is not None:
        state = read_json(output / "progress.json")
        item = completed_boundary(output, state)
        if item is None:
            if state.get("status") == "paused":
                raise RuntimeError("Serial campaign paused before a safe handoff")
            time.sleep(0.02)
            continue
        parent = matching_process(owner)
        if parent is None:
            break
        parent.suspend()
        try:
            # Suspending this parent cannot suspend the independent child.
            # If a new flight was already spawned, resume and wait for it.
            live_flights = []
            for child in parent.children(recursive=True):
                try:
                    command = child.cmdline()
                    if "evaluation.benchmarks.aerial_vln.campaign" in command and "child" in command:
                        live_flights.append(child.pid)
                except psutil.NoSuchProcess:
                    pass
            fresh = completed_boundary(output, read_json(output / "progress.json"))
            if live_flights or fresh is None or fresh["task"] != item["task"]:
                continue
            atomic_write_text(output / "PAUSE", token + "\n")
            write_json(output / "parallel/handoff.json", {"token": token, "dispatcher": owner,
                       "observer": observer_owner, "boundary_episode": item["task"], "requested_at": utc_now()})
        finally:
            parent.resume()
        break
    while matching_process(owner) is not None:
        time.sleep(0.2)
    with campaign_lock(output):
        state = read_json(output / "progress.json")
        if any(item["status"] == "running" for item in state["tasks"].values()):
            raise RuntimeError("Serial task remained running at handoff")
        if state.get("reason") not in {None, "manual_interrupt"}:
            raise RuntimeError("Serial cleanup did not finish successfully")
        marker = output / "PAUSE"
        if not marker.exists() or marker.read_text(encoding="utf-8").strip() != token:
            raise RuntimeError("Handoff pause marker changed; preserve it")
        marker.unlink()
        write_json(output / "parallel/handoff_ready.json", {"verified": True, "completed_at": utc_now(),
                   "seen_scored": sum(item["status"] == "scored" and item["task"]["split"] == "val_seen"
                                      for item in state["tasks"].values())})
    print(json.dumps({"handoff_ready": True}), flush=True)
