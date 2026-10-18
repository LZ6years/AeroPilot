"""Prepare TravelUAV deployment and switch only at the completed Aerial boundary.

Operator-only helper: --prepare creates immutable assets; --tick is safe to run
from a hidden recurring Windows task. It never stops active flights, rewrites
scores, or grants retries. The normal two-lane watchdog owns all evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def controller(overlay):
    return runpy.run_path(overlay["controller_script"], run_name="handoff_controller")


def aerial_finished(state):
    tasks = [v for v in state["tasks"].values() if v["task"]["dataset"] == "aerial_vln"]
    return len(tasks) == 300 and all(v["status"] == "scored" for v in tasks)


def prepare(root, directory):
    old = read(root / "parallel/execution_overlay.json")
    c = controller(old)
    deployment = read(Path(old["controller_script"]).parent / "deployment.json")
    catalog = read(Path(deployment["catalog_path"]))
    template = catalog["maps"][0]
    travel = [p for p in read(root / "parallel/maps_A.json")["maps"]
              if p["map_id"].startswith("travel_uav_")]
    assert travel, "Missing original Travel catalog"
    for package in travel:
        package["ssh"] = copy.deepcopy(template["ssh"])
        executable = package["launch"]["linux_executable"]
        for family in ("closeloop_envs", "carla_town_envs"):
            executable = executable.replace(
                "/home/Newdisk/wzq/data/traveluav/ENVs/" + family + "/",
                "/home/wangzequan/TravelUAV/ENVs/" + family + "/" + family + "/")
        assert executable.startswith("/home/wangzequan/TravelUAV/")
        package["launch"].update(linux_executable=executable,
                                args=template["launch"]["args"], env=template["launch"]["env"])
        package["airsim"]["settings_template"] = str(
            Path(old["execution_source"]) / package["airsim"]["settings_template"])
    # UE packages use bare executable names, not necessarily Linux-Shipping.
    probe = """import json,sys,subprocess
from pathlib import Path
out=[]
for p in json.loads(sys.argv[1]):
 s=Path(p); script=s.read_text() if s.is_file() else ''; binaries=[b for b in s.parent.glob('*/Binaries/Linux/*') if b.is_file() and b.relative_to(s.parent).as_posix() in script and b.open('rb').read(4)==b'\\x7fELF' and b.suffix not in ('.debug','.sym')]
 missing=[]
 for b in binaries:
  r=subprocess.run(['ldd',str(b)],capture_output=True,text=True)
  missing.extend(x.strip() for x in r.stdout.splitlines() if 'not found' in x)
 out.append(dict(path=p,script_exists=s.is_file(),binaries=[str(b) for b in binaries],missing_libraries=missing))
print(json.dumps(out))
"""
    ssh = template["ssh"]
    command = ["ssh", "-p", str(ssh["port"]), "-i", ssh["key_path"], "-o", "BatchMode=yes",
               "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
               ssh["user"] + "@" + ssh["host"], "python3 -c " + shlex.quote(probe) + " " +
               shlex.quote(json.dumps([p["launch"]["linux_executable"] for p in travel]))]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90, check=True)
    readiness = json.loads(result.stdout)
    write(directory / "remote_readiness.json", readiness)
    assert all(p["script_exists"] and p["binaries"] and not p["missing_libraries"] for p in readiness), readiness
    catalog["maps"].extend(travel)
    assets = dict(deployment["asset_hashes"])

    def asset(path, data):
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Refuse to overwrite immutable asset: " + str(path))
        path.write_bytes(payload)
        assets[str(path)] = digest(payload)

    asset(directory / "catalog.json", catalog)
    lanes = {}
    for lane in ("A", "B"):
        path = directory / ("maps_" + lane + ".json")
        asset(path, c["lane_catalog"](catalog, lane, old["remote_gpu_bindings"][lane]))
        lanes[lane] = {**old["lanes"][lane], "map_config_path": str(path), "sha256": assets[str(path)]}
    for package in travel:
        path = Path(package["airsim"]["settings_template"])
        assets[str(path)] = digest(path.read_bytes())
    deployment.update(catalog_path=str(directory / "catalog.json"),
                      enabled_datasets=["aerial_vln", "travel_uav"], asset_hashes=assets)
    previous = Path(old["controller_script"]).parent
    data = {name: (previous / name).read_bytes() for name in old["source_hashes"]}
    data["deployment.json"] = json.dumps(deployment, sort_keys=True, indent=2).encode()
    hashes = {name: digest(content) for name, content in data.items()}
    revision = digest(json.dumps(hashes, sort_keys=True).encode())
    snapshot = root / "parallel/controller_source" / revision[:16]
    for name, content in data.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Immutable snapshot collision")
        path.write_bytes(content)
    candidate = {**old, "controller_script": str(snapshot / "parallel_campaign.py"),
                 "controller_revision": revision, "source_hashes": hashes, "lanes": lanes,
                 "recovery_revision": "travel10245_" + revision[:16], "recovery_validation_required": True}
    # Preserve deployment_revision: it scopes retirement of old-server cleanup.
    controller(candidate)["verify_overlay"](candidate, read(root / "manifest.json"))
    write(directory / "previous_overlay.json", old)
    write(directory / "candidate_overlay.json", candidate)
    print(json.dumps({"candidate": str(directory / "candidate_overlay.json"), "scenes": len(travel)}))


def tick(root, directory):
    from tests.manual.install_recovery_revision import powershell
    candidate = read(directory / "candidate_overlay.json")
    c = controller(candidate)
    record = read(directory / "handoff_status.json")
    if record.get("status") == "installed":
        return "installed"
    state = read(root / "progress.json")
    if not aerial_finished(state):
        write(directory / "handoff_status.json", {"status": "waiting_for_aerial", "updated_at": c["utc_now"]()})
        return "waiting_for_aerial"
    for filename in ("dispatcher.json", "parallel/host_process.json"):
        owner = read(root / filename)
        if owner and c["matching_process"](owner):
            return "waiting_for_owner_exit"
    marker = root / "PAUSE"
    expected = "Waiting for dataset readiness: (20, 'travel_uav')"
    committed = record.get("status") == "committed"
    if not committed and (not marker.exists() or marker.read_text(encoding="utf-8") != expected):
        return "preserving_other_pause"
    manifest = read(root / "manifest.json")
    c["verify_frozen"](root, manifest)
    c["verify_overlay"](candidate, manifest)
    for name, sha in candidate["execution_source_hashes"].items():
        assert digest((Path(candidate["execution_source"]) / name).read_bytes()) == sha, name
    task = "FlyAgent_ImageBudget_" + root.name
    powershell("-Command", "Disable-ScheduledTask -TaskName '" + task + "' | Out-Null")
    try:
        owner = read(root / "parallel/watchdog/status.json").get("watchdog")
        watcher = c["matching_process"](owner) if owner else None
        if watcher:
            assert "watch" in owner["command"] and str(root) in owner["command"]
            watcher.terminate()
            watcher.wait(15)
        with c["campaign_lock"](root):
            state = read(root / "progress.json")
            assert aerial_finished(state) and not any(v["status"] == "running" for v in state["tasks"].values())
            if not committed:
                active = read(root / "parallel/execution_overlay.json")
                assert active == read(directory / "previous_overlay.json") or (
                    record.get("status") == "preparing" and active == candidate)
                assert marker.read_text(encoding="utf-8") == expected
                for filename in ("progress.json", "parallel/execution_overlay.json", "parallel/trial_selection.json",
                                 "parallel/trial_snapshots.json", "parallel/gpu_samples.json", "PAUSE"):
                    path = root / filename
                    backup = directory / (path.name + ".before")
                    if path.exists() and not backup.exists():
                        backup.write_bytes(path.read_bytes())
                # Journal before replacing any files; interrupted installation
                # can safely repeat while the dataset PAUSE still blocks work.
                write(directory / "handoff_status.json", {"status": "preparing", "updated_at": c["utc_now"]()})
                trials = []
                for budget in (20, 30):
                    options = [v["task"] for v in state["tasks"].values() if v["task"]["dataset"] == "travel_uav"
                               and v["task"]["image_budget"] == budget and v["status"] in ("pending", "retry_pending")]
                    assert len(options) >= 2
                    trials.extend(options[:2])
                write(root / "parallel/trial_selection.json", {"tasks": trials, "started_at": c["utc_now"]()})
                write(root / "parallel/trial_snapshots.json", {"snapshots": []})
                write(root / "parallel/gpu_samples.json", {"samples": []})
                state.update(status="paused", reason="manual_interrupt")
                write(root / "progress.json", state)
                write(root / "parallel/execution_overlay.json", candidate)
                write(directory / "handoff_status.json", {"status": "committed", "updated_at": c["utc_now"]()})
            # Removing only the owned dataset marker preserves an intervening stop.
            if marker.exists():
                assert marker.read_text(encoding="utf-8") == expected
                marker.unlink()
            c["atomic_write_text"](root / "parallel/watchdog/RESUME", "Travel environment ready; continue authorized fixed samples")
        powershell("-File", str(REPO / "tests/manual/install_campaign_watchdog.ps1"), "-OutputRoot", str(root), "-Workers", "2")
        write(directory / "handoff_status.json", {"status": "installed", "updated_at": c["utc_now"](),
                                                  "recovery_revision": candidate["recovery_revision"]})
        return "installed"
    finally:
        powershell("-Command", "Enable-ScheduledTask -TaskName '" + task + "' | Out-Null")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--tick", action="store_true")
    args = parser.parse_args()
    assert Path(sys.executable).resolve() == Path(r"F:\conda_env\flyagent\python.exe").resolve()
    root = args.output.resolve()
    directory = root / "parallel/travel_continuation_10245"
    directory.mkdir(exist_ok=True)
    if args.prepare:
        prepare(root, directory)
    elif args.tick:
        print(tick(root, directory))
