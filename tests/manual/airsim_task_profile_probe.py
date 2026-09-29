"""Profile historical TravelUAV case 13: stabilization OR read concurrency.

This is an end-to-end diagnostic, not a success-rate study or an identical
action replay. Child processes use frozen source/configuration and independent
scenes; change only the selected execution setting. Infrastructure, cleanup or review
failures pause the pair. No task wall-clock cap or model override is introduced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.core import BenchmarkAuditReader, EpisodeAuditCursor, write_audit_json
from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner
from src.agent.config import load_model_config
from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
from tests.manual.airsim_phase2_probe import BASELINE, MAPS, stage_summary, verify_cleanup
from tests.manual.experiment_sources import source_hashes
from tests.manual.export_benchmark_review import export_case, validate_bundle
from tests.manual.process_diagnostics import supervise, write_json
from tests.manual.task_profile import TaskProfile, install_profile, summarize

CONFIG = "evaluation/benchmarks/travel_uav/configs/travel_uav_test_seen_quick.example.json"
INDEX = ".external_refs/TravelUAV/data/travel_uav_index.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_cases(manifest, experiment="stabilization"):
    """Select historical case 13 by its frozen position and verify its identity."""
    case = manifest["cases"][12]
    if case["benchmark"] != "travel" or case["scene"] != "ModernCityMap":
        raise ValueError("Historical case 13 no longer identifies ModernCityMap")
    if experiment == "read-only-concurrency":
        return [{**case, "reuse": False, "read_workers": workers} for workers in (1, 4)]
    if experiment != "stabilization":
        raise ValueError("Unknown task experiment")
    return [{**case, "reuse": enabled, "read_workers": 1} for enabled in (False, True)]


class ProfileRunner(TravelUavBenchmarkRunner):
    """Experimental flag injection; production runner defaults remain intact."""

    def __init__(self, config, *, reuse, read_workers=1):
        super().__init__(config, read_only_execution_config=ReadOnlyExecutionConfig(read_workers))
        self.reuse = reuse

    def _make_runtime(self, run_dir, recorder):
        runtime = super()._make_runtime(run_dir, recorder)
        runtime.stabilizer.config = replace(runtime.stabilizer.config,
                                           reuse_verified_post_action_state=self.reuse)
        write_json(Path(run_dir) / "effective_stabilization.json", runtime.stabilizer.config.to_dict())
        write_json(Path(run_dir) / "effective_read_execution.json", self.read_only_execution_config.to_dict())
        return runtime


def make_runner(directory, manifest, item):
    config = replace(TravelUavEvalConfig.from_file(ROOT / CONFIG),
                     map_config_path=str(ROOT / "configs/maps/travel_uav_remote.local"),
                     index_path=str(ROOT / INDEX), expected_index_sha256=manifest["dataset_sha256"],
                     case_ids=(item["case_id"],), limit_per_difficulty=None,
                     output_dir=str(directory / "evaluation_output"), run_root=str(directory / "runs"),
                     env_path=manifest["env_path"])
    runner = ProfileRunner(config, reuse=item["reuse"], read_workers=item.get("read_workers", 1))
    if runner._model_config().redacted_summary() != manifest["model_config"]:
        raise ValueError("Model configuration changed since experiment freeze")
    return runner


def child(directory, number):
    """All model decisions and actions use the ordinary benchmark runner."""
    manifest = read_json(directory.parent / "manifest.json")
    item = manifest["cases"][number - 1]
    runner = make_runner(directory, manifest, item)
    write_json(directory / "experiment.json", {"case": item, "config": runner.config.to_dict(),
               "model": runner._model_config().redacted_summary(),
               "thinking_policy": runner._model_config().thinking_mode or "provider default"})
    profile = TaskProfile(directory / "timings.jsonl")
    try:
        with install_profile(profile), profile.span("experiment.case"):
            result = runner.run(eval_id="evaluation")
        write_json(directory / "result.json", result)
    finally:
        write_json(directory / "timing_summary.json", summarize(profile.path))


def freeze(root, review, experiment="stabilization"):
    """Copy executable inputs, never credentials or historical experiment outputs."""
    root.mkdir(parents=True, exist_ok=False)
    review.mkdir(parents=True, exist_ok=False)
    historical = read_json(BASELINE / "manifest.json")
    if sha(ROOT / INDEX) != historical["dataset_sha256"]["travel"]:
        raise ValueError("TravelUAV dataset differs from historical manifest")
    sources = set(source_hashes()) | {CONFIG, INDEX, "tests/__init__.py",
                                     "configs/maps/travel_uav_remote.local"}
    sources.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "plugins").rglob("*.md"))
    snapshot = root / "source"
    hashes = {}
    for relative in sorted(sources):
        original = ROOT / relative
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = original.read_bytes()
        target.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    model = load_model_config(ROOT / ".env")
    if not model.api_key or not model.model_name or not model.base_url:
        raise ValueError("Model endpoint, model and API key must be configured")
    manifest = {"cases": pair_cases(historical, experiment), "experiment": experiment, "dataset_sha256": sha(ROOT / INDEX),
                "source_snapshot": str(snapshot), "snapshot_hashes": hashes,
                "env_path": str(ROOT / ".env"), "model_config": model.redacted_summary(),
                "review_root": str(review), "baseline_manifest_sha256": sha(BASELINE / "manifest.json"),
                "limitations": "One full task per condition. Model choices and trajectories may differ. "
                                "Profiling hooks add overhead. No success-rate or identical-action speed claim."}
    write_json(root / "manifest.json", manifest)
    return manifest


def verify_frozen(manifest):
    snapshot = Path(manifest["source_snapshot"])
    if any(sha(snapshot / name) != value for name, value in manifest["snapshot_hashes"].items()):
        raise ValueError("Frozen executable inputs changed; no further scene will start")
    if load_model_config(manifest["env_path"]).redacted_summary() != manifest["model_config"]:
        raise ValueError("Frozen model configuration changed; no further scene will start")


def partial_case(directory, case):
    """Build a separate audit from persisted records after a native child exit."""
    source = directory / "partial"
    source.mkdir(exist_ok=False)
    runs = list((directory / "runs").glob("*"))
    if len(runs) == 1:
        audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(runs[0]))
        write_audit_json(source / "audit.json", audit, ())
    else:
        # The renderer can still show a truthful no-score page before scene start.
        audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(directory / "no_run"))
        write_audit_json(source / "audit.json", audit, ())
    write_json(source / "case_result.json", {"episode_id": case["case_id"], "status": "error",
               "completed": False, "metrics": {}, "error": "Child exited without a final result",
               "infrastructure_error": {"error_type": "IncompleteChild"}})
    return source


def audit_counts(source):
    audit = read_json(source / "audit.json")
    calls = Counter(call.get("tool_name") for event in audit.get("agent_transcript", [])
                    if event.get("kind") == "assistant_turn"
                    for call in event.get("payload", {}).get("tool_calls", []))
    usage = Counter()
    for decision in audit.get("planner_debug", []):
        for model_round in decision.get("model_rounds", []):
            for key, value in ((model_round.get("response") or {}).get("usage") or {}).items():
                if key in ("prompt_tokens", "completion_tokens", "total_tokens") and isinstance(value, int):
                    usage[key] += value
    return {"tool_calls": dict(calls), "token_usage": dict(usage)}


def can_continue(status, result):
    """Navigation failures continue; infrastructure, missing result or cleanup pause."""
    return (status.get("exit_code") == 0 and bool(result)
            and status.get("cleanup", {}).get("verified") is True
            and not status.get("cleanup_error") and not status.get("review_error")
            and not result.get("infrastructure_error"))


def model_preflight(root, manifest):
    """One saved RGB/native-tool request verifies access; returned calls never run."""
    from src.llm import DefaultModelClient, ModelRequest
    from tests.manual.tool_selection_ablation_probe import restore_request, wire_check
    candidates = sorted((BASELINE / "cases").glob("13_*/runs/*/debug/tool_loop/*/decision_0001.json"))
    if not candidates:
        raise ValueError("Historical case 13 first multimodal request is unavailable")
    source = candidates[0]
    record = read_json(source)["model_rounds"][0]
    run = next(parent for parent in source.parents if (parent / "debug").is_dir())
    payload = restore_request(record["request"], record["request_media"], run)
    config = load_model_config(manifest["env_path"])
    parameters = {k: v for k, v in payload.items() if k not in ("model", "messages", "tools", "thinking")}
    if config.thinking_mode:
        parameters["thinking"] = {"type": config.thinking_mode}
    status = {"source": str(source), "source_sha256": sha(source), "tools_executed": False,
              "config": config.redacted_summary(), "status": "running"}
    write_json(root / "model_preflight.json", status)
    trace = {}
    try:
        response = DefaultModelClient(config.to_provider_settings()).request(
            ModelRequest(messages=payload["messages"], tools=payload["tools"], parameters=parameters),
            transport_trace=trace)
        status.update(request=response.request_snapshot, response=response.response_snapshot,
                      media_sha256=[item["sha256"] for item in record["request_media"]],
                      returned_model=response.data.get("model"))
        status["wire_check"] = wire_check(response.response_snapshot, payload)
        if status["wire_check"]["issues"]:
            raise ValueError("Model preflight native tool protocol rejected; see saved diagnostics")
        status["status"] = "passed"
    except Exception as exc:
        status.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        status["transport"] = trace
        write_json(root / "model_preflight.json", status)


def refresh_index(root, review, rows, *, state, error=None):
    summary = {"state": state, "rows": rows, "remaining": [i for i in (1, 2) if i > len(rows)],
               "error": error, "recommend_enable": False,
               "accounting": "Compare distributions within like stages; never add inclusive parents and children "
                             "or concurrent logger work to task wall time. Full tasks may follow different routes."}
    write_json(root / "summary.json", summary)
    write_json(review / "summary.json", summary)
    links = []
    for row in rows:
        number = f'{row["index"]:02d}'
        if (review / "cases" / number / "review.html").exists():
            links.append(f'<li>reuse={row["reuse"]}, read_workers={row.get("read_workers", 1)}: <a href="cases/{number}/review.html">时间线</a> · '
                         f'<a href="cases/{number}/audit_review.html">完整审计</a></li>')
    page = '<!doctype html><meta charset="utf-8"><h1>AirSim 完整任务耗时对照</h1>'
    page += '<p>ModernCityMap 同案例配对；条件见各行。运行中的案例尚无最终评分。</p><ul>' + ''.join(links) + '</ul>'
    page += '<p><a href="summary.json">结果与性能数据</a></p><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>'
    (review / "index.html").write_text(page, encoding="utf-8")
    # Case renderer's shared navigation targets ../../review.html.
    (review / "review.html").write_text(page, encoding="utf-8")


def offline_ready(directory):
    """Fail closed: partial offline batches cannot authorize engine trials."""
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    rows = summary.get("conditions", [])
    expected = {(sample, workers) for sample in range(1, 5) for workers in (1, 2, 4)}
    return (summary.get("state") == "passed" and not summary.get("error")
            and len(rows) == 12 and {(r["sample"], r["workers"]) for r in rows} == expected
            and all(r["exit_code"] == 0 and r["completed_batches"] == 5
                    and r["started_calls"] == 10 and r["incomplete_spans"] == 0
                    and r["peak_active_device"] and all(v == 1 for v in r["peak_active_device"].values())
                    and not any(r["remaining_active_device"].values()) for r in rows))


def wait_offline(directory, status_path):
    """Wait for the identified offline parent; never wait forever after its exit."""
    import psutil
    owner = read_json(Path(directory) / "parent.json")
    while True:
        try:
            summary = read_json(Path(directory) / "summary.json")
        except json.JSONDecodeError:
            summary = {}  # A concurrent atomic-sized report write may be incomplete.
        if summary.get("state") == "passed":
            if not offline_ready(directory):
                raise ValueError("Offline passed label lacks complete invariant evidence")
            write_json(status_path, {"state": "offline_passed", "offline_root": str(directory)})
            return
        if summary.get("state") == "paused":
            raise RuntimeError("Offline probe paused; no scene will start")
        try:
            process = psutil.Process(owner["pid"])
            alive = process.create_time() == owner["created"] and process.is_running()
        except psutil.NoSuchProcess:
            alive = False
        if not alive:
            raise RuntimeError("Offline parent exited without passing; no scene will start")
        time.sleep(2)


def wait_existing_process(pid, status_path):
    """Respect another experiment's ownership; observe its process tree only."""
    import psutil
    try:
        original = psutil.Process(pid)
        owned = {pid: original.create_time()}
    except psutil.NoSuchProcess:
        return
    write_json(status_path, {"state": "waiting_existing_experiment", "pid": pid})
    while owned:
        for identity, created in list(owned.items()):
            try:
                process = psutil.Process(identity)
                if process.create_time() != created or not process.is_running():
                    del owned[identity]
                    continue
                for child in process.children(recursive=True):
                    owned[child.pid] = child.create_time()
            except psutil.NoSuchProcess:
                del owned[identity]
        if owned:
            time.sleep(2)


def execute(root, review, *, prepare_only=False, experiment="stabilization"):
    """Parent remains alive through native failures and verifies scene ownership."""
    import psutil
    from src.runtime.configs.map_config import load_map_config
    from src.sim_server.remote.control import RemoteSceneControl

    manifest = freeze(root, review, experiment)
    rows = []
    refresh_index(root, review, rows, state="preflight")
    try:
        packages, _ = load_map_config(MAPS)
        package = next(p for p in packages if p.scene_id == manifest["cases"][0]["scene"])
        preflight = {"disk_free_bytes": shutil.disk_usage(root).free,
                     "system_available_bytes": psutil.virtual_memory().available}
        write_json(root / "preflight.json", preflight)
        if preflight["disk_free_bytes"] < 10 * 1024 ** 3:
            raise ValueError("Less than 10 GiB free before experiment")
        preflight["remote"] = RemoteSceneControl().preflight(package)
        write_json(root / "preflight.json", preflight)
        if prepare_only:
            refresh_index(root, review, rows, state="prepared_only")
            return
        model_preflight(root, manifest)
        snapshot = Path(manifest["source_snapshot"])
        for number, item in enumerate(manifest["cases"], 1):
            verify_frozen(manifest)
            refresh_index(root, review, rows, state=f"running_{number:02d}")
            directory = root / f"{number:02d}"
            command = [sys.executable, "-u", "-X", "faulthandler", "-c",
                       "import os,runpy,sys; os.chdir(sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1), run_name='__main__')",
                       str(snapshot), str(snapshot / "tests/manual/airsim_task_profile_probe.py"),
                       "child", "--root", str(directory), "--number", str(number)]
            status = supervise(command, directory, monitor_gpu=True)  # No task-level wall-clock cap.
            try:
                status["cleanup"] = verify_cleanup(directory, item, status["descendants"],
                                                     snapshot / "configs/maps/travel_uav_remote.local")
            except Exception as exc:
                status["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json(directory / "supervision.json", status)
            result_path = directory / "result.json"
            payload = read_json(result_path) if result_path.exists() else {}
            results = payload.get("results", [])
            result = results[0] if len(results) == 1 and results[0].get("episode_id") == item["case_id"] else {}
            profile = summarize(directory / "timings.jsonl")
            write_json(directory / "timing_summary.json", profile)
            row = {"index": number, "reuse": item["reuse"], "read_workers": item.get("read_workers", 1), "result": result,
                   "elapsed_seconds": status["elapsed_seconds"], "threads": profile["threads"],
                   "incomplete_spans": profile["incomplete"], "airsim_stages": stage_summary(directory)}
            try:
                source = Path(result["case_dir"]) if result else partial_case(directory, item)
                row.update(audit_counts(source))
                row["review"] = export_case(source, review, f"{number:02d}")
                row["resource_validation"] = validate_bundle(review)
                # Both pages must load actual timeline images before a second scene.
                from tests.manual.review_browser_probe import inspect as inspect_browser
                (review / "browser_validation").mkdir(exist_ok=True)
                pages = [f"cases/{number:02d}/{name}.html" for name in ("review", "audit_review")]
                if result and not result.get("infrastructure_error"):
                    browsers = {}
                    for name, executable in (("chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe"),
                                             ("edge", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")):
                        browsers[name] = inspect_browser(Path(executable), review, pages, name)
                    write_json(review / "browser_validation" / f"{number:02d}.json", browsers)
            except Exception as exc:
                status["review_error"] = {"type": type(exc).__name__, "message": str(exc)}
            row["supervision"] = status
            write_json(directory / "supervision.json", status)
            rows.append(row)
            proceed = can_continue(status, result)
            refresh_index(root, review, rows, state="completed" if number == 2 and proceed else "paused")
            print(json.dumps({"index": number, "continue": proceed, "result_status": result.get("status")}), flush=True)
            if not proceed:
                break
    except Exception as exc:
        refresh_index(root, review, rows, state="paused", error={"type": type(exc).__name__, "message": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "child"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--number", type=int, choices=(1, 2))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--experiment", choices=("stabilization", "read-only-concurrency"), default="stabilization")
    parser.add_argument("--after-offline", type=Path, help="Wait for a recorded offline probe to pass before freezing/starting scenes")
    parser.add_argument("--after-process", type=int, help="Wait for an existing experiment and its local descendants without stopping them")
    args = parser.parse_args()
    if args.mode == "child":
        if args.number is None:
            parser.error("child requires --number")
        child(args.root.resolve(), args.number)
    else:
        if args.review_root is None:
            parser.error("run requires --review-root")
        if args.after_offline:
            if args.experiment != "read-only-concurrency":
                parser.error("--after-offline requires --experiment read-only-concurrency")
            gate_path = args.root.resolve().with_suffix(".gate.json")
            write_json(gate_path, {"state": "waiting_offline", "offline_root": str(args.after_offline.resolve())})
            try:
                wait_offline(args.after_offline.resolve(), gate_path)
            except Exception as exc:
                write_json(gate_path, {"state": "paused", "error": type(exc).__name__, "message": str(exc)})
                raise
        if args.after_process:
            wait_existing_process(args.after_process, args.root.resolve().with_suffix(".gate.json"))
        execute(args.root.resolve(), args.review_root.resolve(), prepare_only=args.prepare_only, experiment=args.experiment)


if __name__ == "__main__":
    main()
