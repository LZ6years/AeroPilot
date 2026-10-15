"""Frozen, serial AerialVLN-S pilot and split campaigns with durable recovery.

prepare: freeze source/config and exact manifest without flight. run: pilot then
seen automatically, or unseen independently. child: one original-start attempt.
The supervisor owns recovery; model inputs and navigation policy are unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import psutil

from src.agent.config import load_model_config
from src.agent.config.model import read_env_file
from src.llm import DefaultModelClient, ModelRequest, ProviderRequestError
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner, _summary
from evaluation.benchmarks.aerial_vln.recovery import RecoveryQueue, write_json
from evaluation.benchmarks.core.process_review import write_process_review
from evaluation.benchmarks.core.report_io import atomic_write_text
from evaluation.benchmarks.core.supervision import cleanup_owned, cleanup_verified, supervise

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("H:/FlyAgent-Eval/aerialvln_s_20261007")
PLAN = ROOT / "runs/experiments/aerialvln_s_full_eval_plan_20261006"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment_diagnostics():
    """Verify the user-required interpreter and actual CUDA environment."""
    import torch
    info = {"sys_executable": sys.executable, "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("openai", "httpx2", "transformers", "numpy", "pillow", "psutil")}}
    if os.name == "nt" and Path(sys.executable).resolve() != Path("F:/conda_env/flyagent/python.exe").resolve():
        raise RuntimeError("Campaign requires the explicit flyagent interpreter")
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_tensor_probe"] = float(torch.ones(1, device="cuda").item())
    return info


def provider_probe(env_path, output, *, injected_tls=0, service_check=False):
    """Small real request; optional deterministic TLS faults before real recovery."""
    settings = load_model_config(env_path).to_provider_settings()
    if service_check:
        settings = replace(settings, timeout_sec=30, retry_deadline_sec=30,
                           provider_connection_retries=0, provider_timeout_retries=0)
    client = DefaultModelClient(settings)
    trace = {}
    if injected_tls:
        import ssl
        original = client._request_chat_stream if settings.stream else client._request_chat
        calls = 0

        def fault_then_request(*args):
            nonlocal calls
            calls += 1
            if calls <= injected_tls:
                raise ssl.SSLEOFError("local recovery-validation TLS fault")
            return original(*args)

        if settings.stream:
            client._request_chat_stream = fault_then_request
        else:
            client._request_chat = fault_then_request
    try:
        result = client.request(ModelRequest([{"role": "user", "content": "Reply with exactly OK."}]), transport_trace=trace)
        record = {"success": bool(result.text and result.text.strip()), "transport": trace,
                  "injected_tls_failures": injected_tls, "model": settings.model_name}
    except ProviderRequestError as exc:
        record = {"success": False, "error": exc.to_dict(), "transport": trace}
    finally:
        client._reset_connection()
    write_json(output, record)
    return record["success"]


def prepare(output, plan_dir=PLAN):
    """Freeze dirty source, knowledge and configuration without credentials."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Campaign already prepared; use run to resume the frozen manifest")
    for name in ("provider_success.json", "provider_recovery.json"):
        if not (output / name).is_file() or not _json(output / name).get("success"):
            raise ValueError("Run and pass the real provider probe before freezing")
    if not (output / "verification.json").is_file() or not _json(output / "verification.json").get("passed"):
        raise ValueError("Deterministic recovery tests must pass before freezing")
    for key, value in read_env_file(ROOT / ".env").items():
        if key.startswith("FLYAGENT_"):
            os.environ.setdefault(key, value)
    info = environment_diagnostics()
    write_json(output / "environment.json", info)
    if not info["cuda_available"]:
        raise RuntimeError("CUDA unavailable in flyagent; campaign was not launched")
    frozen = output / "source"
    files = []
    for folder in ("src", "plugins", "configs", "evaluation", "memory", "tests/manual"):
        files.extend(path for path in (ROOT / folder).rglob("*") if path.is_file()
                     and path.suffix in {".py", ".json", ".md", ".txt", ".ps1"}
                     and not any(part in {"outputs", "__pycache__", ".git", "node_modules"} for part in path.parts))
    files.extend(path for path in (ROOT / "AGENT.md", ROOT / "AGENTS.md", ROOT / "pyproject.toml",
                                  ROOT / "requirements.txt", ROOT / "tests/__init__.py") if path.is_file())
    hashes = {}
    for path in sorted(set(files)):
        relative = path.relative_to(ROOT)
        destination = frozen / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write_json(output / "source_hashes.json", hashes)
    configs = {}
    tasks = []
    counts = {}
    for split in ("val_seen", "val_unseen"):
        data = _json(Path(plan_dir) / f"{split}_full_config.json")
        data.update(dataset_root=str((ROOT / data["dataset_root"]).resolve()),
                    map_config_path=str(frozen / data["map_config_path"]),
                    env_path=str(ROOT / ".env"), output_dir=str(output / split),
                    run_root=str(output / split / "runs"),
                    report_profile="paper_process" if split == "val_seen" else "simple_process",
                    enable_action_planner_debug=False, generate_html_review=True, save_artifacts=True)
        config = AerialVlnEvalConfig.from_dict(data)
        prepared = AerialVlnBenchmarkRunner(config).prepare_evaluation(require_execution_ready=False)
        configs[split] = config.to_dict()
        counts[split] = len(prepared.cases)
        tasks.extend({"split": split, "episode_id": case.episode_id, "scene_id": case.scene_id,
                      "difficulty": prepared.selection.assignment_by_episode[case.episode_id].difficulty}
                     for case in prepared.cases)
        write_json(output / f"{split}_validation.json", prepared.validation_dict())
        write_json(output / f"{split}_config.json", config.to_dict())
        from tests.manual.aerial_vln_remote_readiness import inspect_remote_readiness
        readiness = inspect_remote_readiness(config)
        write_json(output / f"{split}_remote_readiness.json", readiness)
        if not readiness["ready"]:
            raise RuntimeError(f"Read-only remote preflight failed for {split}: {readiness['errors']}")
    if counts != {"val_seen": 333, "val_unseen": 531}:
        raise ValueError(f"Unexpected split population: {counts}")
    pilot = _json(Path(plan_dir) / "pilot_cases.json")["cases"]
    batches = _json(Path(plan_dir) / "batch_manifest.json")["batches"]
    by_key = {RecoveryQueue.key(task): task for task in tasks}
    pilot = [by_key[RecoveryQueue.key(task)] for task in pilot]
    ordered = [by_key[f'{batch["split"]}/{episode_id}'] for batch in batches for episode_id in batch["episode_ids"]]
    if len(ordered) != 864 or len({RecoveryQueue.key(task) for task in ordered}) != 864:
        raise ValueError("Batch manifest coverage is incomplete or duplicated")
    if len(pilot) != 15 or len({task["scene_id"] for task in pilot}) != 13:
        raise ValueError("Pilot must cover the confirmed 15 cases and 13 scenes")
    model = load_model_config(ROOT / ".env").redacted_summary()
    manifest = {"version": 1, "configs": configs, "tasks": ordered, "pilot": pilot,
                "batches": batches, "counts": counts, "model": model,
                "verification": _json(output / "verification.json"),
                "provider_validation": {name: _json(output / name) for name in
                                        ("provider_success.json", "provider_recovery.json")},
                "source_hashes": hashes, "environment": info,
                "policy": {"episode_attempts": 3, "episode_deadline_sec": 7200,
                           "no_progress_sec": 900, "checkpoint_every": 25,
                           "minimum_free_gib": 10, "storage_multiplier": 2,
                           "automatic_stage": "pilot_then_seen", "unseen_independent": True}}
    manifest["fingerprint"] = _digest(manifest)
    write_json(output / "manifest.json", manifest)
    RecoveryQueue(output / "progress.json", ordered, manifest["fingerprint"])
    (output / "temp").mkdir(exist_ok=True)
    write_json(output / "storage_initial.json", {"free_bytes": shutil.disk_usage(output).free})
    print(json.dumps({"prepared": str(output), "counts": counts, "pilot": len(pilot),
                      "fingerprint": manifest["fingerprint"]}), flush=True)


@contextmanager
def campaign_lock(output):
    """OS releases the lock on crash; only one host dispatcher can own the queue."""
    path = Path(output) / "campaign.lock"
    with path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _attempt_dir(output, task, number):
    if "dataset" in task and "image_budget" in task:
        return Path(output) / task["dataset"] / f'images_{task["image_budget"]}' / task["split"] / "attempts" / task["episode_id"] / f"attempt_{number:02d}"
    return Path(output) / task["split"] / "attempts" / task["episode_id"] / f"attempt_{number:02d}"


def child(output, split, episode_id, attempt):
    manifest = _json(Path(output) / "manifest.json")
    task = next(task for task in manifest["tasks"] if task["split"] == split and task["episode_id"] == episode_id)
    directory = _attempt_dir(output, task, attempt)
    info = environment_diagnostics()
    write_json(directory / "environment.json", info)
    if not info["cuda_available"]:
        raise RuntimeError("CUDA unavailable in the experiment child")
    config = replace(AerialVlnEvalConfig.from_dict(manifest["configs"][split]),
                     episode_ids=(episode_id,), output_dir=str(directory / "evaluation"),
                     run_root=str(directory / "runs"))
    runner = AerialVlnBenchmarkRunner(config)
    prepared = runner.prepare_evaluation(require_execution_ready=True)
    result = runner.run_case(prepared.cases[0], directory / "evaluation",
                             assignment=prepared.selection.assignment_by_episode[episode_id],
                             provenance=prepared.provenance)
    write_json(directory / "result.json", result)
    print(json.dumps({"episode_id": episode_id, "status": result["status"],
                      "infrastructure_error": result.get("infrastructure_error")}), flush=True)


def _cleanup(output, manifest, task, attempt):
    directory = _attempt_dir(output, task, attempt["number"])
    process_path = directory / "process.json"
    descendants = _json(process_path).get("descendants", []) if process_path.exists() else []
    if process_path.exists():
        owner = _json(process_path)
        if owner.get("status") == "running":
            try:
                process = psutil.Process(owner["pid"])
                if process.create_time() != owner.get("created") or process.cmdline() != owner.get("command"):
                    raise ValueError("Interrupted child ownership cannot be verified")
                process.kill()
                process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
    receipts = cleanup_owned(directory / "runs", manifest["configs"][task["split"]]["map_config_path"], descendants)
    verified = cleanup_verified(receipts)
    write_json(directory / "cleanup.json", {"verified": verified, "receipts": receipts})
    return verified


def _render_reports(output, queue):
    """Report errors remain independent of flight results and can be repaired."""
    if any("image_budget" in item["task"] for item in queue.state["tasks"].values()):
        from evaluation.experiments.image_budget_report import write_report
        return write_report(output, queue.state)
    for split in ("val_seen", "val_unseen"):
        scored = [item["result"] for item in queue.state["tasks"].values()
                  if item["task"]["split"] == split and item["result"] is not None]
        summary = _summary(scored)
        summary.update(dataset_split=split, requested_case_count=sum(item["task"]["split"] == split for item in queue.state["tasks"].values()),
                       campaign_status=queue.state["status"], campaign_reason=queue.state["reason"])
        directory = Path(output) / split
        directory.mkdir(exist_ok=True)
        write_json(directory / "summary.json", summary)
        rows = []
        for result in scored:
            case_dir = Path(result["run_dir"]).parents[1] / "evaluation/cases" / result["episode_id"]
            path = case_dir / "review.html"
            rows.append('<tr><td><a href="' + Path(os.path.relpath(path, directory)).as_posix() + '">' +
                        html.escape(result["episode_id"]) + '</a></td><td>' + str(result.get("difficulty")) +
                        '</td><td>' + str(result.get("metrics", {}).get("success")) + '</td></tr>')
        atomic_write_text(directory / "index.html", '<!doctype html><meta charset="utf-8"><h1>' + split +
                          '</h1><p>Valid scored episodes: ' + str(len(scored)) + '</p><pre>' +
                          html.escape(json.dumps(summary["metrics"], indent=2)) + '</pre><table>' + ''.join(rows) + '</table>')


def storage_check(output, queue, tasks):
    """Budget retained bytes twice plus one observed temporary peak; keep 10 GiB."""
    sizes = []
    peaks = []
    for item in queue.state["tasks"].values():
        for attempt in item["attempts"]:
            if attempt.get("retained_bytes"):
                sizes.append(attempt["retained_bytes"])
                peaks.append(attempt.get("temporary_depth_bytes", 0))
    remaining = sum(queue.state["tasks"][RecoveryQueue.key(task)]["status"] != "scored" for task in tasks)
    free = shutil.disk_usage(output).free
    # Before the pilot there is no empirical estimate; require the minimum.
    per_case = max(sizes) if sizes else 0
    required = max(10 * 1024**3, remaining * per_case * 2 + max(peaks, default=0))
    record = {"free_bytes": free, "required_bytes": required, "remaining_cases": remaining,
              "observed_max_retained_bytes": per_case, "observed_cases": len(sizes), "passed": free >= required}
    write_json(Path(output) / "storage_check.json", record)
    return record["passed"]


def run(output, stage):
    output = Path(output).resolve()
    manifest = _json(output / "manifest.json")
    for relative, expected in manifest["source_hashes"].items():
        if hashlib.sha256((output / "source" / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    if load_model_config(manifest["configs"]["val_seen"]["env_path"]).redacted_summary() != manifest["model"]:
        raise ValueError("Effective model config changed since freeze")
    env = read_env_file(manifest["configs"]["val_seen"]["env_path"])
    for key, value in env.items():
        if key.startswith("FLYAGENT_"):
            os.environ.setdefault(key, value)
    os.environ.update(TEMP=str(output / "temp"), TMP=str(output / "temp"))
    os.chdir(output / "source")
    queue = RecoveryQueue(output / "progress.json", manifest["tasks"], manifest["fingerprint"])
    stop = lambda: (output / "PAUSE").exists()
    with campaign_lock(output):
        write_json(output / "dispatcher.json", {"pid": os.getpid(), "stage": stage, "interpreter": sys.executable})
        if not queue.recover_interrupted(lambda task, attempt: _cleanup(output, manifest, task, attempt)):
            return
        seen = [task for task in manifest["tasks"] if task["split"] == "val_seen"]
        unseen = [task for task in manifest["tasks"] if task["split"] == "val_unseen"]

        def probe():
            return provider_probe(manifest["configs"]["val_seen"]["env_path"], output / "service_check.json", service_check=True)

        def execute(task, attempt):
            directory = _attempt_dir(output, task, attempt["number"])
            command = [sys.executable, "-u", "-X", "faulthandler", "-m",
                       "evaluation.benchmarks.aerial_vln.campaign", "child", "--output", str(output),
                       "--split", task["split"], "--episode-id", task["episode_id"], "--attempt", str(attempt["number"])]
            process = supervise(command, directory, stop=stop)
            try:
                verified = _cleanup(output, manifest, task, attempt)
            except Exception as exc:
                write_json(directory / "cleanup.json", {"verified": False, "error": str(exc)})
                verified = False
            result_path = directory / "result.json"
            # run_case saves scoring before reporting: adopt it after a report crash.
            if not result_path.is_file():
                result_path = directory / "evaluation/cases" / task["episode_id"] / "case_result.json"
            result = _json(result_path) if result_path.is_file() else {}
            valid = bool(result.get("metrics")) and not result.get("infrastructure_error") and not result.get("metric_error")
            valid = valid and not process.get("reason")
            if valid and verified:
                case_dir = directory / "evaluation/cases" / task["episode_id"]
                try:
                    result["report"] = write_process_review(case_dir, result,
                        profile=manifest["configs"][task["split"]]["report_profile"])
                    result.pop("report_error", None)
                except Exception as exc:
                    result["report_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
                write_json(case_dir / "case_result.json", result)
            # Delete only this attempt's generated depth after score/report/cleanup.
            depth_bytes = 0
            if valid and verified and result.get("report") and not result.get("report_error"):
                owned = (directory / "runs").resolve()
                for path in owned.rglob("*.npy"):
                    if owned not in path.resolve().parents or not any("depth" in part.lower() for part in path.parts):
                        continue
                    depth_bytes += path.stat().st_size
                    path.unlink()
            retained = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            receipt = {"result": result, "cleanup_verified": verified, "valid_score": valid,
                       "process": process, "directory": str(directory), "retained_bytes": retained,
                       "temporary_depth_bytes": depth_bytes}
            write_json(directory / "receipt.json", receipt)
            print(json.dumps({"episode": RecoveryQueue.key(task), "attempt": attempt["number"],
                              "valid_score": valid, "cleanup_verified": verified, "bytes": retained}), flush=True)
            return receipt

        if not storage_check(output, queue, manifest["pilot"] if stage == "seen" else unseen):
            queue.pause("storage_budget")
            return
        if stage == "seen":
            if not queue.run(manifest["pilot"], execute, probe, checkpoint=lambda: storage_check(output, queue, seen),
                             stop=stop, after_episode=lambda: _render_reports(output, queue)):
                _render_reports(output, queue)
                return
            pilot_items = [queue.state["tasks"][RecoveryQueue.key(task)] for task in manifest["pilot"]]
            if any(item["result"].get("report_error") or not item["result"].get("report") for item in pilot_items):
                queue.pause("pilot_report_pending")
                _render_reports(output, queue)
                return
            if not storage_check(output, queue, seen):
                queue.pause("pilot_storage_budget")
                _render_reports(output, queue)
                return
            write_json(output / "pilot_gate.json", {"passed": True, "cases": 15, "navigation_sr_is_gate": False})
            tasks = seen
        else:
            if not (output / "pilot_gate.json").is_file():
                queue.pause("pilot_gate_required")
                return
            tasks = unseen
        queue.run(tasks, execute, probe, checkpoint=lambda: storage_check(output, queue, tasks),
                  stop=stop, after_episode=lambda: _render_reports(output, queue))
        if queue.state["status"] == "stage_complete":
            queue.state.update(status=f"{stage}_complete", reason=None)
            queue.save()
        _render_reports(output, queue)
        print(json.dumps({"status": queue.state["status"], "reason": queue.state["reason"]}), flush=True)


def repair_reports(output):
    output = Path(output).resolve()
    manifest = _json(output / "manifest.json")
    queue = RecoveryQueue(output / "progress.json", manifest["tasks"], manifest["fingerprint"])
    with campaign_lock(output):
        for item in queue.state["tasks"].values():
            if item["result"] is None:
                continue
            result = item["result"]
            directory = Path(result["run_dir"]).parents[1] / "evaluation/cases" / result["episode_id"]
            result["report"] = write_process_review(directory, result,
                profile=manifest["configs"][item["task"]["split"]]["report_profile"])
            result.pop("report_error", None)
            write_json(directory / "case_result.json", result)
            queue.save()
        _render_reports(output, queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "child", "repair-reports", "probe"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--split")
    parser.add_argument("--episode-id")
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "run":
        run(args.output, args.stage)
    elif args.command == "child":
        child(args.output, args.split, args.episode_id, args.attempt)
    elif args.command == "repair-reports":
        repair_reports(args.output)
    elif args.command == "probe":
        args.output.mkdir(parents=True, exist_ok=True)
        normal = provider_probe(ROOT / ".env", args.output / "provider_success.json")
        recovered = normal and provider_probe(ROOT / ".env", args.output / "provider_recovery.json", injected_tls=2)
        print(json.dumps({"normal": normal, "recovered": recovered}), flush=True)
        return 0 if normal and recovered else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
