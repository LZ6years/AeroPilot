"""One-shot guard for a completed pilot with pre-fix GPU audit contamination.

Never interrupts flights. Archives historical GPU samples and reruns the same
controller's technical gate using only its new classified observations. Only
the exact forbidden-GPU-only gate failure and two scored pilots are eligible.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(root):
    from tests.manual.install_recovery_revision import powershell
    d = root / "parallel/gpu23_20261017"
    done = d / "trial_audit_resume.json"
    if done.exists():
        return
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    state = read(root / "progress.json")
    if state.get("reason") != "parallel_technical_check" or (root / "PAUSE").exists():
        return
    check = read(root / "parallel/trial_check.json")
    if set(check.get("errors", [])) != {"forbidden_remote_gpu"}:
        return
    overlay = read(root / "parallel/execution_overlay.json")
    if overlay["recovery_revision"] != "gpu23_d917e990c163b8bd":
        return
    c = runpy.run_path(overlay["controller_script"], run_name="gpu_trial_recheck")
    for name in ("dispatcher.json", "parallel/host_process.json"):
        if c["matching_process"](read(root / name)):
            return
    trials = read(root / "parallel/trial_selection.json")["tasks"]
    if len(trials) != 2 or any(not any(v["task"] == t and v["status"] == "scored" for v in state["tasks"].values()) for t in trials):
        return
    samples = read(root / "parallel/gpu_samples.json")
    fresh = [s for s in samples["samples"] if s.get("owned") and all("all_observed_gpus" in i for i in s["owned"])]
    if not fresh or any(c["gpu_violation"](s) for s in fresh):
        return
    owner = read(root / "parallel/watchdog/status.json").get("watchdog")
    watch = c["matching_process"](owner) if owner else None
    if watch:
        watch.terminate()
        watch.wait(15)
    with c["campaign_lock"](root):
        state = read(root / "progress.json")
        assert state.get("reason") == "parallel_technical_check" and not (root / "PAUSE").exists()
        assert not any(v["status"] == "running" for v in state["tasks"].values())
        c["write_json"](d / "gpu_samples_before_pilot_recheck.json", samples)
        c["write_json"](root / "parallel/gpu_samples.json", {"samples": fresh})
        c["write_json"](d / "progress_before_pilot_recheck.json", state)
        state.update(status="paused", reason="manual_interrupt")
        c["write_json"](root / "progress.json", state)
        c["atomic_write_text"](root / "parallel/watchdog/RESUME", "Recheck completed GPU 2/3 pilot with classified audit samples")
    powershell("-File", str(Path(__file__).with_name("install_campaign_watchdog.ps1")), "-OutputRoot", str(root), "-Workers", "2")
    c["write_json"](done, {"status": "installed", "at": c["utc_now"](), "preserved_pilots": trials})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
