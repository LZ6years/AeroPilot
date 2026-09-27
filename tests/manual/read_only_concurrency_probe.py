"""Replay frozen, accepted dual-vision calls; never invoke a model or simulator.

Each sample/worker condition has a fresh supervised process and five calls of
the same batch. First-use costs are separate from warm comparisons. Historical
records are selection evidence, not expected outputs for the current backend.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.perception.frame_payload import frame_bytes, frame_metadata, load_rgb_frame, load_depth_frame
from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
from src.runtime.control.read_only_executor import ReadJob, ReadOnlyBatchExecutor
from src.tools.dispatcher import ToolDispatcher
from src.plugins.contracts.skill import SKILL_ARTIFACT_REFS_KEY
from tests.manual.experiment_sources import source_hashes
from tests.manual.process_diagnostics import supervise, write_json
from tests.manual.task_profile import TaskProfile, install_profile, summarize
from tests.manual.export_benchmark_review import validate_bundle
from tests.manual.gpu_resource_monitor import summarize_resources
from tests.manual.gpu_probe_environment import (check_interpreter, cache_sources, verify_weights,
    environment, validate_environment, other_project_jobs)

BASELINE = ROOT / "tests/manual_outputs/benchmark_20_flash_20260923_v2"
VISION = "skill.visual_grounding.locate"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accepted_pairs(audit):
    accepted = {e["payload"]["tool_call_id"] for e in audit["agent_transcript"]
                if e["kind"] == "tool_result" and e["payload"]["status"] == "ok"}
    for event in audit["agent_transcript"]:
        if event["kind"] != "assistant_turn":
            continue
        calls = event["payload"].get("tool_calls", [])
        if (len(calls) == 2 and all(c["tool_name"] == VISION and c["tool_call_id"] in accepted for c in calls)
                and len({c["args"]["observation_id"] for c in calls}) == 1):
            yield calls


def select_samples(baseline):
    """Deterministic historical selection: Aerial detection + depth, two Travel."""
    selected, seen = [], set()
    for directory in sorted((baseline / "cases").iterdir()):
        benchmark = "aerial" if "_aerial_" in directory.name else "travel"
        for path in sorted(directory.glob("evaluation_output/evaluation/cases/*/audit.json")):
            audit = read(path)
            for calls in accepted_pairs(audit):
                group = ("aerial_depth" if any(c["args"]["include_distance"] for c in calls) else "aerial_detection") if benchmark == "aerial" else "travel"
                if group != "travel" and group in seen:
                    continue
                if group == "travel" and sum(x["benchmark"] == "travel" for x in selected) == 2:
                    continue
                selected.append({"benchmark": benchmark, "group": group, "audit": str(path.resolve()),
                                 "case_id": path.parent.name, "calls": calls})
                seen.add(group)
    if len(selected) != 4 or sum(s["benchmark"] == "aerial" for s in selected) != 2:
        raise ValueError("Need two genuine accepted dual-vision batches per benchmark")
    return selected


def import_samples(path, root):
    """Copy exact frozen inputs; never search historical runs for replacements."""
    original = read(path)
    samples = copy.deepcopy(original["samples"])
    if len(samples) != 4 or [s["number"] for s in samples] != [1, 2, 3, 4]:
        raise ValueError("Expected four ordered frozen samples")
    if sorted(s["benchmark"] for s in samples) != ["aerial", "aerial", "travel", "travel"]:
        raise ValueError("Expected two samples per benchmark")
    identities = set()
    for item in samples:
        if len(item["calls"]) != 2 or any(c["tool_name"] != VISION for c in item["calls"]):
            raise ValueError("Expected original dual-vision calls")
        identity = (item["case_id"], item["observation"]["observation_id"], tuple(c["tool_call_id"] for c in item["calls"]))
        if identity in identities:
            raise ValueError("Duplicate frozen sample")
        identities.add(identity)
        if any(sha(p) != value for p, value in item["input_hashes"].items()):
            raise ValueError("Historical frozen images changed")
        directory = root / "inputs" / f'{item["number"]:02d}'
        directory.mkdir(parents=True)
        hashes = {}
        for modality, frame in item["observation"]["frames"].items():
            source = Path(frame["path"])
            if str(source) not in item["input_hashes"]:
                raise ValueError("Frame missing frozen hash")
            target = directory / (modality + source.suffix)
            shutil.copyfile(source, target)
            frame["path"] = str(target)
            hashes[str(target)] = sha(target)
        item["input_hashes"] = hashes
        load_rgb_frame(item["observation"]["frames"]["rgb"])
        if any(c["args"]["include_distance"] for c in item["calls"]):
            load_depth_frame(item["observation"]["frames"]["depth"], require_metric=True)
    return samples, original["baseline_manifest_sha256"]


def freeze(root, review, baseline=BASELINE, *, input_manifest=None, require_cuda=False):
    root.mkdir(parents=True, exist_ok=False)
    review.mkdir(parents=True, exist_ok=False)
    sources = set(source_hashes()) | {"tests/__init__.py"}
    sources.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "plugins").rglob("*.md"))
    snapshot, hashes = root / "source", {}
    for relative in sorted(sources):
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        hashes[relative] = sha(target)
    if input_manifest:
        samples, baseline_hash = import_samples(input_manifest, root)
    else:
        samples, baseline_hash = select_samples(baseline), sha(baseline / "manifest.json")
    for number, item in enumerate(samples if not input_manifest else [], 1):
        directory = root / "inputs" / f"{number:02d}"
        directory.mkdir(parents=True)
        audit = read(item["audit"])
        obs_id = item["calls"][0]["args"]["observation_id"]
        frames, input_hashes = {}, {}
        for frame in audit["frames"]:
            if frame["observation_id"] != obs_id or frame["modality"] not in {"rgb", "depth"}:
                continue
            modality = frame["modality"]
            path = directory / ("rgb.png" if modality == "rgb" else "depth.npy")
            path.write_bytes(frame_bytes(frame))
            metadata = {k: v for k, v in frame_metadata(frame).items()
                        if k not in {"base64", "path", "artifact_path", "artifact_ref", "metadata"}}
            frames[modality] = {**metadata, "path": str(path)}
            input_hashes[str(path)] = sha(path)
        load_rgb_frame(frames["rgb"])
        if any(c["args"]["include_distance"] for c in item["calls"]):
            load_depth_frame(frames["depth"], require_metric=True)
        item.update(number=number, audit_sha256=sha(item["audit"]), input_hashes=input_hashes,
                    observation={"observation_id": obs_id, "frames": frames})
    manifest = {"samples": samples, "source_snapshot": str(snapshot), "snapshot_hashes": hashes,
                "review_root": str(review), "repeats": 5, "workers": [1, 2, 4],
                "baseline_manifest_sha256": baseline_hash,
                "input_manifest": str(input_manifest) if input_manifest else None,
                "input_manifest_sha256": sha(input_manifest) if input_manifest else None,
                "require_cuda": require_cuda, "expected_executable": str(Path(sys.executable).resolve()),
                "limitations": "Offline saved observations; no navigation score. First iteration is cold. "
                "Travel dual calls both request distance. GPU device execution remains serial."}
    write_json(root / "manifest.json", manifest)
    return manifest


class NoEnvironment:
    def __getattr__(self, name):
        raise AssertionError("Offline probe cannot access environment: " + name)


def public_signature(result):
    """Only output locations differ; compare the full public numerical result."""
    data = result["data"]
    return {"ok": result["ok"], "message": result["message"],
            "observation_id": data.get("observation_id"), "query": data.get("query"),
            "include_distance": data.get("include_distance"),
            "candidates": data.get("candidates"), "error": data.get("error")}


def child(directory, number, workers, *, kind="comparison"):
    manifest = read(directory.parent / "manifest.json")
    item = manifest["samples"][number - 1]
    check_interpreter(manifest["expected_executable"])
    if manifest.get("require_cuda"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    preflight_started = time.perf_counter()
    env = environment(directory, require_cuda=manifest.get("require_cuda", False))
    env["preflight_sec"] = time.perf_counter() - preflight_started
    write_json(directory / "environment.json", env)
    validate_environment(env, require_cuda=manifest.get("require_cuda", False), expected=manifest.get("environment"))
    if manifest.get("cuda_trace_repeats"):
        from tests.manual.cuda_timeline import require_cuda_profiler
        require_cuda_profiler(directory)
    if manifest.get('sam_coordinate_ab'):
        from tests.manual.sam_coordinate_probe import validate_coordinates
        validate_coordinates(directory)
    dispatcher = ToolDispatcher(NoEnvironment())
    calls = copy.deepcopy(item["calls"])
    if kind == "mixed":
        from src.plugins.skills.knowledge_service import KnowledgeService
        dispatcher.knowledge_service = KnowledgeService(dispatcher.plugin_registry.skill_catalog)
        name = manifest["mixed_knowledge_name"]
        calls += [{"tool_call_id": f"constructed_knowledge_{i}", "tool_name": "knowledge.load", "args": {"name": name}} for i in (1, 2)]
    config = ReadOnlyExecutionConfig(workers, 2, {VISION: 2})
    profile = TaskProfile(directory / "timings.jsonl")
    rows = []
    write_json(directory / "execution_config.json", config.to_dict())
    with install_profile(profile):
        for repeat in range(manifest["repeats"] if kind == "comparison" else 1):
            output = directory / f"r{repeat + 1}"
            output.mkdir()
            profile.decision.set(repeat + 1)
            jobs = []
            for call in calls:
                args = copy.deepcopy(call["args"])
                context = {"latest_observation": copy.deepcopy(item["observation"]),
                           "run_dir": str(output), "tool_call_id": call["tool_call_id"]}
                jobs.append(ReadJob(call["tool_call_id"], call["tool_name"],
                    dispatcher.read_execution_hints(call["tool_name"], args),
                    lambda c=call, a=args, ctx=context: dispatcher.call("offline", "uav0", c["tool_name"], a, ctx)))
            def event(name, payload):
                profile.emit({"event": name, "decision": repeat + 1, **payload})
            profiled = repeat + 1 in manifest.get("cuda_trace_repeats", [])
            event_profiled = repeat + 1 in manifest.get("cuda_event_repeats", [])
            diagnostic = nullcontext()
            coordinate_candidate = manifest.get('sam_coordinate_ab') and repeat > 0 and (repeat - 1) % 4 in (1, 2)
            if coordinate_candidate:
                from tests.manual.sam_coordinate_probe import candidate_coordinates
                diagnostic = candidate_coordinates()
            if profiled:
                from tests.manual.cuda_timeline import capture_batch
                diagnostic = capture_batch(output)
            elif event_profiled:
                from tests.manual.cuda_event_probe import capture_events
                diagnostic = capture_events(output, sam_modules=manifest.get('sam_module_events', False))
            started = time.perf_counter()
            with diagnostic:
                with profile.span("experiment.batch"):
                    results = ReadOnlyBatchExecutor(config, event).execute(jobs)
                if profiled or event_profiled:
                    # Preserve completed tool outputs even if trace export fails.
                    write_json(output / "tool_results.json", [asdict(result) for result in results])
            row = {"repeat": repeat + 1, "cold": repeat == 0, "wall_sec": time.perf_counter() - started,
                   "profiled": profiled or event_profiled,
                   "coordinate_variant": 'device' if coordinate_candidate else 'original',
                   "tool_call_ids": [c["tool_call_id"] for c in calls], "constructed_input": kind == "mixed",
                   "results": [asdict(result) for result in results]}
            rows.append(row)
            write_json(output / "results.json", row)
            print(json.dumps({"sample": number, "workers": workers, "repeat": repeat + 1,
                              "seconds": row["wall_sec"], "ok": all(r.ok for r in results)}), flush=True)
            if not all(r.ok for r in results):
                raise RuntimeError("Capability failure; preserve evidence and stop")
            if any(r.data.get("audit_details", {}).get(key) for r in results
                   for key in ("segmentation_error", "depth_error")):
                raise RuntimeError("Backend fallback/error occurred; preserve evidence and stop")
            if kind == "mixed" and (not results[2].data.get("body_included") or results[3].data.get("body_included")):
                raise RuntimeError("Knowledge delivery ordering failed")
    write_json(directory / "results.json", rows)
    write_json(directory / "timing_summary.json", summarize(profile.path))


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)


def condition_summary(directory):
    rows = read(directory / "results.json") if (directory / "results.json").exists() else sorted(
        (read(p) for p in directory.glob("r*/results.json")), key=lambda row: row["repeat"])
    profile = summarize(directory / "timings.jsonl")
    spans, events = profile.get("spans", []), profile.get("device_events", [])
    active, peaks = {}, {}
    for event in events:
        if event["event"] in {"device_execution_started", "device_execution_finished"}:
            device = event["device"]
            active[device] = active.get(device, 0) + (1 if event["event"] == "device_execution_started" else -1)
            peaks[device] = max(peaks.get(device, 0), active[device])
    warm = [r["wall_sec"] for r in rows if not r["cold"] and not r.get("profiled", False)]
    queued, waits, calls = {}, [], []
    for event in events:
        key = (event.get("decision"), event.get("tool_call_id"))
        if event["event"] == "read_call_queued" and "monotonic_sec" in event:
            queued[key] = event["monotonic_sec"]
        if event["event"] == "read_call_started" and key in queued and "monotonic_sec" in event:
            waits.append(event["monotonic_sec"] - queued[key])
        if event["event"] in {"read_call_started", "read_call_completed", "read_call_failed"}:
            calls.append(event)
    memory = {}
    for event in events:
        for key, value in event.get("torch_memory", {}).items():
            memory[key] = max(memory.get(key, value), value)
    stage_names = ("vision.grounding.load", "vision.sam.load", "vision.grounding.forward", "vision.sam.forward",
                   "vision.image_read", "vision.depth_read", "vision.depth_estimate", "vision.artifact_write")
    return {"completed_batches": len(rows), "submitted_calls": sum(e["event"] == "read_call_queued" for e in events),
            "started_calls": sum(e["event"] == "read_call_started" for e in events),
            "forward_calls": {backend: sum(s["name"] == f"vision.{backend}.forward" for s in spans)
                              for backend in ("grounding", "sam")},
            "model_loads": sum(e["event"] == "model_loaded" for e in events),
            "peak_active_device": peaks, "remaining_active_device": active,
            "device_wait_sec": sum(e.get("wait_sec", 0) for e in events if e["event"] == "device_execution_started"),
            "cold_sec": rows[0]["wall_sec"] if rows else None,
            "warm_p50_sec": statistics.median(warm) if warm else None, "warm_p95_sec": percentile(warm, .95),
            "incomplete_spans": len(profile.get("incomplete", [])), "queue_wait_sec": sum(waits),
            "stage_work_sec": {name: sum(s["ended"] - s["time"] for s in spans if s["name"] == name) for name in stage_names},
            "loaded_models": [e for e in events if e["event"] == "model_loaded"],
            "call_events": calls, "torch_memory_maxima": memory or None,
            "resources": summarize_resources(directory)}


def render(root, manifest, conditions, *, state, error=None):
    """Portable, complete replay evidence (not a fabricated navigation audit)."""
    review = Path(manifest["review_root"])
    summary = {"state": state, "error": error, "conditions": conditions,
               "limitations": manifest["limitations"], "require_cuda": manifest.get("require_cuda", False),
               "comparison_batches": sum(c["completed_batches"] for c in conditions if c.get("kind", "comparison") == "comparison"),
               "environment": manifest.get("environment"), "production_default_changed": False}
    write_json(root / "summary.json", summary)
    write_json(review / "summary.json", summary)
    links = []
    for item in manifest["samples"]:
        number = item["number"]
        target = review / "cases" / f"{number:02d}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["observation"]["frames"]["rgb"]["path"], target / "rgb.png")
        title = f'{item["benchmark"]} {item["case_id"]} / {item["observation"]["observation_id"]}'
        sections = [f'<h1>{html.escape(title)}</h1><p>离线只读重放；无导航评分。{html.escape(state)}</p>',
                    '<a href="../../index.html">总索引</a><div class="phase-timeline"><img src="rgb.png" width="720">']
        complete = {"sample": item, "conditions": []}
        for condition in [c for c in conditions if c["sample"] == number]:
            directory = root / condition["directory"]
            results = read(directory / "results.json") if (directory / "results.json").exists() else [read(p) for p in sorted(directory.glob("r*/results.json"))]
            record = {**condition, "results": results}
            complete["conditions"].append(record)
            label = condition.get("kind", "comparison") + ' / ' + str(condition["workers"]) + ' 线程'
            sections.append('<h2>' + html.escape(label) + '</h2><pre>' + html.escape(json.dumps(condition, ensure_ascii=False, indent=2)) + '</pre>')
            for row in results:
                sections.append(f'<h3>第 {row["repeat"]} 次，{row["wall_sec"]:.3f} 秒</h3>')
                for call_index, result in enumerate(row["results"]):
                    sections.append('<pre>' + html.escape(json.dumps(public_signature(result), ensure_ascii=False, indent=2)) + '</pre>')
                    for artifact in result["data"].get(SKILL_ARTIFACT_REFS_KEY, []):
                        source = Path(artifact)
                        name = f'{directory.name}_r{row["repeat"]}_c{call_index}_{source.name[-20:]}'
                        shutil.copyfile(source, target / name)
                        sections.append(f'<img src="{html.escape(name)}" width="720">')
            for name in ("timings.jsonl", "gpu_resources.jsonl", "resources.jsonl", "process.json", "stdout.log", "stderr.log", "environment.json", "execution_config.json", "windows_gpu_memory.jsonl", "windows_gpu_memory.stderr.log", "windows_gpu_monitor.json"):
                if (directory / name).is_file():
                    copy_name = f'{directory.name}_{name}'
                    shutil.copyfile(directory / name, target / copy_name)
                    sections.append(f'<p><a href="{copy_name}">{copy_name}</a></p>')
        sections.append('</div>')
        write_json(target / "audit.json", complete)
        common = '<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:30px;max-width:1100px}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        (target / "review.html").write_text(common + ''.join(sections) + '<a href="audit_review.html">完整审计</a>', encoding="utf-8")
        (target / "audit_review.html").write_text(common + ''.join(sections) + '<a href="audit.json">完整原始数据</a><pre>' + html.escape(json.dumps(complete, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
        links.append(f'<li>{html.escape(title)} <a href="cases/{number:02d}/review.html">时间线</a> <a href="cases/{number:02d}/audit_review.html">完整审计</a></li>')
    (review / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>只读并发离线对照</h1><p>' + html.escape(state) + '</p><ul>' + ''.join(links) + '</ul><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
    write_json(review / "resource_validation.json", validate_bundle(review))


def validate_condition(summary, rows, *, require_cuda=False, admission=False, expected_batches=5):
    """A successful process exit alone is insufficient proof of GPU execution."""
    if summary["completed_batches"] != expected_batches or summary["incomplete_spans"]:
        raise RuntimeError("Incomplete batches or timing spans")
    peaks = summary["peak_active_device"]
    if not peaks or any(v != 1 for v in peaks.values()) or any(summary["remaining_active_device"].values()):
        raise ValueError("Device concurrency invariant failed")
    if require_cuda:
        if any(not d.startswith("cuda:") for d in peaks):
            raise ValueError("CPU fallback detected")
        for model in summary["loaded_models"]:
            if not model.get("parameter_devices") or any(not d.startswith("cuda:") for d in model["parameter_devices"]):
                raise ValueError("Model parameters were not verified on CUDA")
        if not summary["loaded_models"]:
            raise ValueError("No actual model device evidence")
        backends = [m.get("backend") for m in summary["loaded_models"] if m.get("backend")]
        if len(backends) != len(set(backends)):
            raise ValueError("Backend model loaded more than once in one condition")
    if admission:
        if any(summary["forward_calls"].get(backend, 0) < 1 for backend in ("grounding", "sam")):
            raise ValueError("Admission requires both GroundingDINO and SAM forward calls")
        candidates = [c for row in rows for result in row["results"] for c in result["data"].get("candidates", [])]
        if not candidates or not any(c["distance_status"] not in {"not_requested", "metric_depth_unavailable"} for c in candidates):
            raise ValueError("Admission has no completed depth measurements")
        artifacts = [Path(p) for row in rows for result in row["results"] for p in result["data"].get(SKILL_ARTIFACT_REFS_KEY, [])]
        if not artifacts or any(not p.is_file() for p in artifacts):
            raise ValueError("Admission evidence image missing")


def validate_mixed(summary, rows, workers):
    """Check same-entry delivery and actual completion order, not just totals."""
    row = rows[0]
    if len(row["results"]) != 4 or not row["constructed_input"]:
        raise ValueError("Mixed batch must explicitly contain four constructed calls")
    first, second = (r["data"] for r in row["results"][2:])
    if not first.get("body_included") or second.get("body_included") or first["name"] != second["name"]:
        raise ValueError("Knowledge incremental delivery mismatch")
    events = summary["call_events"]
    ends = {e["tool_call_id"]: e["monotonic_sec"] for e in events if e["event"] == "read_call_completed"}
    starts = {e["tool_call_id"]: e["monotonic_sec"] for e in events if e["event"] == "read_call_started"}
    if len(ends) != 4 or ends["constructed_knowledge_1"] > starts["constructed_knowledge_2"]:
        raise ValueError("Knowledge calls were not completed in submission order")
    if workers == 4 and ends["constructed_knowledge_1"] >= max(ends[c] for c in row["tool_call_ids"][:2]):
        raise ValueError("Knowledge did not complete while capability work was in flight")


def compare_with_serial(results_by_worker):
    """Check repeat stability immediately; thread equivalence against worker=1."""
    for rows in results_by_worker.values():
        first = [public_signature(r) for r in rows[0]["results"]]
        if any([public_signature(r) for r in row["results"]] != first for row in rows):
            raise ValueError("Public results differ across repeats")
    if 1 in results_by_worker:
        baseline = [public_signature(r) for r in results_by_worker[1][0]["results"]]
        if any([public_signature(r) for r in rows[0]["results"]] != baseline for rows in results_by_worker.values()):
            raise ValueError("Public results differ from GPU serial baseline")


def execute(root, review, *, input_manifest=None, require_cuda=False):
    check_interpreter()
    manifest = freeze(root, review, input_manifest=input_manifest, require_cuda=require_cuda)
    import psutil
    write_json(root / "parent.json", {"pid": os.getpid(), "created": psutil.Process().create_time()})
    conditions = []
    render(root, manifest, conditions, state="prepared")
    try:
        snapshot = Path(manifest["source_snapshot"])
        if require_cuda:
            settings = read(snapshot / "plugins/capabilities/visual_grounding/locate/capability.json")["settings"]
            manifest["weight_sources"] = cache_sources(settings)
            from src.plugins import PluginRegistry
            from src.plugins.skills.knowledge_service import KnowledgeService
            catalog = KnowledgeService(PluginRegistry.from_config().skill_catalog).catalog()
            manifest["mixed_knowledge_name"] = sorted(c["name"] for c in catalog)[0]
            write_json(root / "manifest.json", manifest)

        def run_condition(item, workers, kind="comparison"):
            if any(sha(snapshot / name) != value for name, value in manifest["snapshot_hashes"].items()):
                raise ValueError("Frozen source changed")
            if any(sha(path) != value for path, value in item["input_hashes"].items()):
                raise ValueError("Frozen images changed")
            if require_cuda:
                blockers = other_project_jobs(ROOT)
                write_json(root / "resource_preflight.json", {"blocking_jobs": blockers})
                if blockers:
                    raise RuntimeError("Another project experiment is active; pause instead of contaminating timing")
                verify_weights(manifest["weight_sources"])
            directory = root / (f's{item["number"]}w{workers}' if kind == "comparison" else f'{kind}w{workers}')
            render(root, manifest, conditions, state="running")
            command = [sys.executable, "-u", "-X", "faulthandler", "-c",
                       "import os,runpy,sys; os.chdir(sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1), run_name='__main__')",
                       str(snapshot), str(snapshot / "tests/manual/read_only_concurrency_probe.py"),
                       "child", "--root", str(directory), "--sample", str(item["number"]), "--workers", str(workers), "--kind", kind]
            status = supervise(command, directory, monitor_gpu=True)
            summary = condition_summary(directory)
            condition = {"sample": item["number"], "workers": workers, "kind": kind, "directory": directory.name,
                         "exit_code": status["exit_code"], "process_elapsed_sec": status["elapsed_seconds"], **summary}
            conditions.append(condition)
            render(root, manifest, conditions, state="running")
            if status["exit_code"]:
                raise RuntimeError(f"{directory.name}: child failed; inspect stderr and environment records")
            rows = read(directory / "results.json")
            validate_condition(summary, rows, require_cuda=require_cuda, admission=kind == "admission",
                               expected_batches=5 if kind == "comparison" else 1)
            if kind == "mixed":
                validate_mixed(summary, rows, workers)
            if not manifest.get("environment"):
                manifest["environment"] = read(directory / "environment.json")
                write_json(root / "environment.json", manifest["environment"])
                write_json(root / "manifest.json", manifest)
            print(json.dumps({k: condition[k] for k in ("sample", "workers", "kind", "completed_batches", "warm_p50_sec", "forward_calls")}), flush=True)
            return rows

        admission_item = next(s for s in manifest["samples"] if s["group"] == "aerial_depth")
        if require_cuda:
            run_condition(admission_item, 1, "admission")
        serial_results = {}
        for item in manifest["samples"]:
            # Rotate order to reduce a fixed worker-order environmental bias.
            offset = (item["number"] - 1) % 3
            order = manifest["workers"][offset:] + manifest["workers"][:offset]
            compared = {}
            for workers in order:
                compared[workers] = run_condition(item, workers)
                compare_with_serial(compared)
            serial_results[item["number"]] = compared[1]
        if require_cuda:
            for workers in (1, 2, 4):
                rows = run_condition(admission_item, workers, "mixed")
                visual_rows = [{**row, "results": row["results"][:2]} for row in rows]
                compare_with_serial({1: serial_results[admission_item["number"]], "mixed": visual_rows})
        from tests.manual.review_browser_probe import inspect
        (review / "browser_validation").mkdir(exist_ok=True)
        pages = [f"cases/{number:02d}/{page}.html" for number in (1, 3) for page in ("review", "audit_review")]
        for name, executable in (("chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe"),
                                 ("edge", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")):
            write_json(review / "browser_validation" / f"{name}.json", inspect(Path(executable), review, pages, name))
        render(root, manifest, conditions, state="passed")
    except BaseException as exc:
        render(root, manifest, conditions, state="paused", error={"type": type(exc).__name__, "message": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "child"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--sample", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--workers", type=int, choices=(1, 2, 4))
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--kind", choices=("comparison", "admission", "mixed"), default="comparison")
    args = parser.parse_args()
    if args.mode == "run":
        if args.review_root is None:
            parser.error("run requires --review-root")
        execute(args.root.resolve(), args.review_root.resolve(),
                input_manifest=args.input_manifest.resolve() if args.input_manifest else None, require_cuda=args.require_cuda)
    else:
        if args.sample is None or args.workers is None:
            parser.error("child requires --sample and --workers")
        child(args.root.resolve(), args.sample, args.workers, kind=args.kind)


if __name__ == "__main__":
    main()
