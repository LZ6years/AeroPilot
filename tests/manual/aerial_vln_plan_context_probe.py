"""Freeze selected real AerialVLN flights and compare immutable evidence by cohort.

Evaluation-only orchestration: production runners own navigation and scoring;
the existing supervisor owns subprocesses and scene cleanup. Historical readers
never regenerate requests or pick a better navigation outcome. Commands are
prepare/admit/run/report, with a private child command for isolated execution.
Launch with F:\\conda_env\\flyagent\\python.exe; all children inherit sys.executable.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from urllib.parse import quote

WORKSPACE = Path(__file__).resolve().parents[2]
# Execution/admission use frozen dependencies. Reporting from the workspace uses
# current offline readers; their hashes distinguish derived reports from flights.
# The original .env remains outside the snapshot and is never copied.
if "--root" in sys.argv and sys.argv[1:2] != ["report"]:
    _candidate = Path(sys.argv[sys.argv.index("--root") + 1]).resolve() / "source"
    if _candidate.is_dir():
        sys.path.insert(0, str(_candidate))
sys.path.insert(0 if not sys.path or "source" not in sys.path[0] else 1, str(WORKSPACE))

from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.metrics import SUCCESS_DISTANCE
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
from evaluation.benchmarks.aerial_vln.campaign import campaign_lock, environment_diagnostics
from evaluation.benchmarks.core.report_io import atomic_write_text
from src.agent.config import load_model_config, read_env_file
from tests.manual.process_diagnostics import supervise
from tests.manual.supervised_flight_probe import cleanup_owned
from tests.manual.aerial_vln_plan_mode import PLAN_MODES, admission_passed, freeze_plan_mode

DEFAULT_BASELINE = Path("H:/FlyAgent-Eval/aerialvln_s_20261007")
DEFAULT_ROOT = Path("H:/FlyAgent-Eval/avln_plan_context_fix_10_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
DEFAULT_PREVIOUS = Path("H:/FlyAgent-Eval/avln_plan_context_10_20261010_001628")
SKIP = {"source", "source_snapshot", "snapshot", "untracked", "__pycache__",
        "review_assets", "artifacts", "frames", "provider_inputs", ".git", "node_modules"}
METRICS = ("success", "oracle_success", "ndtw", "sdtw", "distance_to_goal",
           "path_length", "uav_action_steps")
# Review annotation only; the formal metric suite keeps its own success rule.
NEAR_BOUNDARY_DISTANCE = 25.0


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_experiment_progress(root):
    """Project native parallel leases for report readers without writing state."""
    root = Path(root)
    progress = read(root / "progress.json")
    if "cases" in progress:
        return progress
    from evaluation.benchmarks.aerial_vln.campaign import _attempt_dir
    manifest = read(root / "manifest.json")
    cases = []
    for task in manifest["tasks"]:
        item = progress["tasks"][f'{task["split"]}/{task["episode_id"]}']
        attempts = [{**attempt, "directory": str(_attempt_dir(root, task, attempt["number"]))}
                    for attempt in item["attempts"]]
        cases.append({"task": task, "status": "completed" if item["status"] == "scored" else item["status"],
                      "attempts": attempts})
    return {"status": "complete" if progress["status"] == "seen_complete" else progress["status"],
            "cases": cases, "native_status": progress["status"]}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_acceptance_reader():
    """Load a derived audit reader without replacing frozen flight modules.

    Like report rendering, audits may repair reading of captured labels. Record
    the reader hash in gates; child navigation always imports its frozen source.
    """
    return current_report_reader("aerial_vln_plan_context_acceptance")


def current_report_reader(filename):
    """Import only evaluation readers; frozen navigation dependencies stay owned by children."""
    name = "_current_report_reader_" + filename
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, WORKSPACE / "tests/manual" / (filename + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def select_cases(ids, progress):
    """Preserve user-selected identities/order; outcomes never drive sampling."""
    if not isinstance(ids, list) or len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("Exactly ten distinct episode IDs are required")
    by_id = {v["task"]["episode_id"]: v for v in progress["tasks"].values()
             if v["task"]["split"] == "val_seen"}
    if any(cid not in by_id or by_id[cid]["status"] != "scored" for cid in ids):
        raise ValueError("Every episode must have a scored val_seen baseline")
    return [dict(by_id[cid]["task"]) for cid in ids]


def select_hard_cases(progress, seed=20261011, scenes=(2, 5, 10, 12, 14)):
    """Evaluation sampling uses identities only; scores never enter the ordering."""
    selected = []
    for scene in scenes:
        pool = [v["task"] for v in progress["tasks"].values()
                if v["status"] == "scored" and v["task"]["split"] == "val_seen"
                and v["task"]["difficulty"] == "high" and int(v["task"]["scene_id"]) == scene]
        if len(pool) < 2:
            raise ValueError(f"Scene {scene} has fewer than two completed high cases")
        ordered = sorted(pool, key=lambda t: hashlib.sha256(
            f"{seed}:{scene}:high:{t['episode_id']}".encode()).hexdigest())
        selected.extend(dict(t) for t in ordered[:2])
    return selected


def select_cohorts(case_ids, hard_ids, progress, *, hard_only=False):
    """Select locked cohorts without scores, preserving existing CLI defaults."""
    if hard_only and hard_ids is None:
        raise ValueError("--hard-only requires --hard-case-ids-file")
    original = [] if hard_only else select_cases(case_ids, progress)
    hard = select_cases(hard_ids, progress) if hard_ids is not None else []
    if hard and hard != select_hard_cases(progress):
        raise ValueError("Hard case file differs from the locked identity-only sampling rule")
    if {t["episode_id"] for t in original} & {t["episode_id"] for t in hard}:
        raise ValueError("Original and hard cohorts overlap")
    cohorts = []
    if original:
        cohorts.append({"name": "original10", "episode_ids": [t["episode_id"] for t in original]})
    if hard:
        cohorts.append({"name": "hard10", "episode_ids": [t["episode_id"] for t in hard]})
    return original + hard, cohorts


def select_subset(tasks, cohorts, ids):
    """Filter a validated cohort while preserving its locked identity order.

    Full hard sampling runs first. A focused regression list is selection
    metadata, never a new score-based sampler or a model instruction.
    """
    if not isinstance(ids, list) or not ids or any(not isinstance(cid, str) or not cid for cid in ids):
        raise ValueError("Subset must be a non-empty list of episode IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Subset contains duplicate episode IDs")
    known = {task["episode_id"] for task in tasks}
    if not set(ids) <= known:
        raise ValueError("Subset contains unknown episode IDs outside the selected cohorts")
    selected = [task for task in tasks if task["episode_id"] in ids]
    if [task["episode_id"] for task in selected] != ids:
        raise ValueError("Subset must preserve the locked cohort order")
    groups = [{**group, "source_episode_ids": list(group["episode_ids"]),
               "episode_ids": [cid for cid in group["episode_ids"] if cid in ids]}
              for group in cohorts if any(cid in ids for cid in group["episode_ids"])]
    return selected, groups


def match_previous_cohort(previous, tasks, *, allow_subset=False):
    """Match a complete cohort inside a larger run; reject missing/reordered IDs."""
    if previous.get("status") != "complete":
        raise ValueError("Previous experiment must be complete")
    rows = previous.get("cases", [])
    ids = {task["episode_id"] for task in tasks}
    selected = [row for row in rows if row["task"]["episode_id"] in ids]
    if (not allow_subset and len(rows) != len(tasks)) or [r["task"] for r in selected] != tasks:
        raise ValueError("Previous experiment must have identical ordered cohort tasks")
    if any(row.get("status") not in {"completed", "model_error"}
           or not row.get("attempts") or not row["attempts"][-1].get("result") for row in selected):
        raise ValueError("Previous cohort must have terminal outcomes with recorded results")
    return selected


def freeze_plugins(root, hashes, profile):
    """Only the new snapshot default is replaced; workspace/history stay intact."""
    target = root / "source/configs/plugins.json"
    shutil.copy2(profile, target)
    hashes["configs/plugins.json"] = digest(target)


def freeze_source(workspace, destination):
    """Copy executable/configuration text, including every compaction prompt."""
    suffixes = {".py", ".json", ".md", ".txt", ".ps1", ".toml"}
    files = []
    for folder in ("src", "plugins", "configs", "evaluation", "memory", "tests/manual"):
        files.extend(p for p in (workspace / folder).rglob("*") if p.is_file()
                     and p.suffix in suffixes
                     and not set(p.relative_to(workspace).parts) & (SKIP | {"outputs"}))
    files.extend(workspace / p for p in ("AGENT.md", "AGENTS.md", "pyproject.toml",
                                       "requirements.txt", "tests/__init__.py"))
    hashes = {}
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = digest(target)
    return hashes


def guard(root):
    """Fail before further calls if frozen source, cases or provider drift."""
    manifest = read(root / "manifest.json")
    for relative, value in manifest["source_hashes"].items():
        if digest(root / "source" / relative) != value:
            raise ValueError("Frozen source changed: " + relative)
    if digest(Path(manifest["case_file"])) != manifest["case_file_sha256"]:
        raise ValueError("Selected episode file changed")
    for key in ("hard_case_file", "subset_case_file", "plugins_config"):
        if manifest.get(key) and digest(Path(manifest[key])) != manifest[key + "_sha256"]:
            raise ValueError(f"Frozen experiment input changed: {key}")
    if digest(Path(manifest["baseline"]) / "manifest.json") != manifest["baseline_manifest_sha256"]:
        raise ValueError("Historical baseline manifest changed")
    if manifest.get("previous_root") and digest(Path(manifest["previous_root"]) / "progress.json") != manifest["previous_progress_sha256"]:
        raise ValueError("Previous experiment results changed")
    for reference in manifest.get("reference_roots", []):
        for name in ("manifest", "progress"):
            if digest(Path(reference["root"]) / (name + ".json")) != reference[name + "_sha256"]:
                raise ValueError(f"Reference experiment {name} changed: {reference['root']}")
    if load_model_config(manifest["config"]["env_path"]).redacted_summary() != manifest["model"]:
        raise ValueError("Provider configuration differs from the frozen baseline profile")
    return manifest


def prepare(root, baseline, case_file, previous_root=None, plugins_config=None,
            hard_case_ids_file=None, hard_only=False, plan_mode="optional", reference_roots=(),
            subset_case_ids_file=None):
    if plan_mode not in PLAN_MODES:
        raise ValueError(f"Unknown plan mode: {plan_mode}")
    if root.exists():
        root = root.with_name(root.name + "_" + datetime.now().strftime("%H%M%S"))
    if root.exists():
        raise ValueError("Output already exists")
    prior = read(baseline / "manifest.json")
    baseline_progress = read(baseline / "progress.json")
    tasks, cohorts = select_cohorts(None if hard_only else read(case_file),
        read(hard_case_ids_file) if hard_case_ids_file is not None else None,
        baseline_progress, hard_only=hard_only)
    if subset_case_ids_file is not None:
        tasks, cohorts = select_subset(tasks, cohorts, read(subset_case_ids_file))
    previous_covers_all = False
    if previous_root is not None:
        previous = read(previous_root / "progress.json")
        # Existing combined runs compare their original ten to the previous ten.
        # Hard-only runs instead match those same high cases inside the prior 20.
        previous_covers_all = len(previous.get("cases", [])) == len(tasks) or hard_only or subset_case_ids_file is not None
        previous_tasks = tasks if previous_covers_all else tasks[:10]
        match_previous_cohort(previous, previous_tasks, allow_subset=hard_only or subset_case_ids_file is not None)
    references = []
    for reference in dict.fromkeys(Path(p).resolve() for p in reference_roots):
        match_previous_cohort(read(reference / "progress.json"), tasks, allow_subset=True)
        references.append({"root": str(reference), "manifest_sha256": digest(reference / "manifest.json"),
                           "progress_sha256": digest(reference / "progress.json")})
    profile = load_model_config(WORKSPACE / ".env").redacted_summary()
    if profile != prior["model"]:
        raise ValueError("Current provider profile must match the full baseline")
    plugins_config = Path(plugins_config or WORKSPACE / "configs/plugins.json").resolve()
    plugins = read(plugins_config)
    if plugins_config == WORKSPACE / "configs/plugins.json" and plugins != read(baseline / "source/configs/plugins.json"):
        raise ValueError("Default plugin configuration differs from the baseline")
    root.mkdir(parents=True)
    info = environment_diagnostics()
    write(root / "environment.json", info)
    if not info["cuda_available"]:
        raise RuntimeError("CUDA is unavailable; real perception was not launched")
    hashes = freeze_source(WORKSPACE, root / "source")
    freeze_plugins(root, hashes, plugins_config)
    overlays = freeze_plan_mode(root, hashes, plan_mode, digest)
    config = dict(prior["configs"]["val_seen"])
    config.update(map_config_path=str(root / "source/configs/maps/aerial_vln_s_remote.example.json"),
                  context_config_path=str(root / "source/configs/context.json"),
                  env_path=str(WORKSPACE / ".env"), episode_ids=[t["episode_id"] for t in tasks],
                  output_dir=str(root / "evaluation"), run_root=str(root / "runs"),
                  report_profile="audit", enable_action_planner_debug=True)
    manifest = {"schema_version": "aerial_plan_context_cohorts_v2",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "workspace_root": str(WORKSPACE), "baseline": str(baseline),
                "baseline_manifest_sha256": digest(baseline / "manifest.json"),
                "case_file": str(hard_case_ids_file if hard_only else case_file),
                "case_file_sha256": digest(hard_case_ids_file if hard_only else case_file),
                "plugins_config": str(plugins_config), "plugins_config_sha256": digest(plugins_config),
                "model": profile, "plugins": plugins, "config": config, "tasks": tasks,
                "source_hashes": hashes, "environment": info,
                "plan_mode": plan_mode, "experiment_overrides": overlays, "reference_roots": references,
                "source_changes": [p for p in sorted(set(hashes) | set(prior["source_hashes"]))
                                   if hashes.get(p) != prior["source_hashes"].get(p)],
                "limits": "Current-version comparison; other working-tree changes prevent causal attribution."}
    manifest["selection_mode"] = "hard_only" if hard_only else ("original_plus_hard" if len(cohorts) == 2 else "original_only")
    if subset_case_ids_file is not None:
        manifest.update(selection_mode=manifest["selection_mode"] + "_subset",
            subset_case_file=str(subset_case_ids_file), subset_case_file_sha256=digest(subset_case_ids_file))
    manifest["cohorts"] = [{**cohort,
        "primary_baseline": str(previous_root or baseline) if previous_covers_all or cohort["name"] == "original10" else str(baseline),
        "secondary_baseline": str(baseline)} for cohort in cohorts]
    manifest["overall_primary_baseline"] = str(previous_root if previous_covers_all else baseline)
    from evaluation.benchmarks.aerial_vln.protocol import AERIAL_VLN_PROTOCOL
    manifest["protocol_variant"] = AERIAL_VLN_PROTOCOL
    manifest["place_memory_admission_required"] = "capabilities.place_memory" in plugins.get("plugins", {}).get("enabled", [])
    if hard_case_ids_file is not None:
        manifest.update(hard_case_file=str(hard_case_ids_file), hard_case_file_sha256=digest(hard_case_ids_file),
            hard_sampling={"seed": 20261011, "scenes": [2, 5, 10, 12, 14], "per_scene": 2,
                "key": "SHA256(seed:scene:high:episode_id)", "uses_scores": False})
    manifest["technical_gate_indices"] = list(range(min(2, len(tasks))))
    if len(cohorts) == 2:
        first_next = len(cohorts[0]["episode_ids"])
        if first_next not in manifest["technical_gate_indices"]:
            manifest["technical_gate_indices"].append(first_next)
    write(root / "manifest.json", manifest)
    if previous_root is not None:
        previous_hashes = read(previous_root / "manifest.json").get("source_hashes", {})
        manifest.update(previous_root=str(previous_root), previous_progress_sha256=digest(previous_root / "progress.json"),
            source_changes_from_previous=[p for p in sorted(set(hashes) | set(previous_hashes))
                                          if hashes.get(p) != previous_hashes.get(p)])
        write(root / "manifest.json", manifest)
    (root / "working_tree.patch").write_bytes(subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "diff", "--binary", "HEAD"], cwd=WORKSPACE))
    atomic_write_text(root / "git_status.txt", subprocess.check_output(
        ["git", "status", "--short"], cwd=WORKSPACE).decode("utf-8"))
    write(root / "progress.json", {"status": "prepared", "cases": [
        {"task": t, "status": "pending", "attempts": []} for t in tasks]})
    report(root, refresh_history=True)
    return root


def load_credentials(manifest):
    """Resolve configured secrets in memory; never serialize values."""
    for k, v in read_env_file(manifest["config"]["env_path"]).items():
        if k.startswith("FLYAGENT_"):
            os.environ.setdefault(k, v)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def admit(root):
    from src.agent.planning.task_plan import TaskPlan
    from src.llm import ModelRequest, create_model_client
    from src.plugins import PluginRegistry
    from src.tools.registry import default_registry
    from tests.manual.aerial_vln_remote_readiness import inspect_remote_readiness
    from tests.manual.taskplan_recommendation_probe import validate_tool_args
    manifest = guard(root)
    load_credentials(manifest)
    if manifest.get("previous_root"):
        from tests.manual.aerial_vln_context_module_audit import replay_history
        previous_root = Path(manifest["previous_root"])
        previous_comparison = read(previous_root / "comparison.json")
        expected = previous_comparison["current_diagnostics"]["navigation_usage"]["captured_requests"]
        historical = replay_history(root, previous_root, write, digest, expected_requests=expected)
        if not historical["passed"]:
            raise RuntimeError("Historical request replay failed; inspect historical_request_replay.json")
    config = AerialVlnEvalConfig.from_dict(manifest["config"])
    readiness = inspect_remote_readiness(config)
    write(root / "remote_readiness.json", readiness)
    if not readiness["ready"]:
        raise RuntimeError("Read-only remote readiness failed; inspect remote_readiness.json")
    registry = default_registry(PluginRegistry.from_config(root / "source/configs/plugins.json"))
    tools = [s.to_provider_tool() for s in registry.model_specs()]
    write(root / "provider_tools.json", tools)
    if "capabilities.pixel_depth" in manifest.get("plugins", {}).get("plugins", {}).get("enabled", []):
        from tests.manual.aerial_vln_pixel_place_audit import admit_saved_evidence
        sandbox = admit_saved_evidence(root, Path(manifest["previous_root"]), write, digest)
        if not sandbox["passed"]:
            raise RuntimeError("Pixel/place sandbox admission failed; navigation not started")
    previous = read(Path(manifest["baseline"]) / "progress.json")
    first = next(v["result"] for v in previous["tasks"].values()
                 if v["task"] == manifest["tasks"][0])
    images = sorted(Path(first["run_dir"]).glob("artifacts/**/rgb/*.png"))
    if not images:
        raise ValueError("No original real RGB exists for multimodal admission")
    image_path = images[0]
    shutil.copy2(image_path, root / "admission_rgb.png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    from tests.manual.aerial_vln_plan_context_admission import run_plan_admission, run_no_plan_admission
    model = load_model_config(config.env_path)
    client = create_model_client(model.to_provider_settings())
    admission_runner = run_no_plan_admission if manifest.get("plan_mode") == "off" else run_plan_admission
    result = admission_runner(root, client, model, tools,
        "data:image/png;base64," + encoded, str(image_path), digest(image_path), write)
    if manifest.get("place_memory_admission_required"):
        from tests.manual.place_memory_admission import run_place_admission
        # The image must be the sandbox observation, not a different baseline's RGB.
        sandbox_record = read(root / "pixel_place_admission.json")
        photo_ref = sandbox_record["inspect_place"]["data"]["photos"][0]["ref"]
        image = root / "capability_admission" / photo_ref
        run_place_admission(root, client, model, tools,
            "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii"), write)
    return result


def next_attempt(row, resume):
    """Navigation/model outcomes are terminal; only clean interruptions resume."""
    if row["status"] in {"completed", "model_error"}:
        return None
    if row["attempts"]:
        if not resume or not row["attempts"][-1].get("cleanup_verified"):
            raise ValueError("Interrupted attempt requires verified cleanup before resumption")
        if len(row["attempts"]) >= 3:
            raise ValueError("Three infrastructure attempts exhausted")
    return len(row["attempts"]) + 1


def child(root, index, number):
    manifest = guard(root)
    load_credentials(manifest)
    directory = root / "cases" / f"{index + 1:02d}" / f"attempt_{number:02d}"
    info = environment_diagnostics()
    write(directory / "environment.json", info)
    if not info["cuda_available"]:
        raise RuntimeError("CUDA unavailable in isolated child")
    data = dict(manifest["config"])
    data.update(episode_ids=[manifest["tasks"][index]["episode_id"]],
                output_dir=str(directory / "evaluation_output"), run_root=str(directory / "runs"))
    runner = AerialVlnBenchmarkRunner(AerialVlnEvalConfig.from_dict(data))
    result = runner.run(eval_id="evaluation")
    write(directory / "result.json", result["results"][0])


def audit_gate(root, manifest, row):
    """Re-audit completed evidence on resume, preserving previous gate receipts."""
    from src.runtime.configs.map_config import load_map_config
    from tests.manual.aerial_vln_case_gate import audit_case
    result = row["attempts"][-1]["result"]
    directory = Path(row["attempts"][-1]["directory"])
    packages, _ = load_map_config(manifest["config"]["map_config_path"])
    package = next(p for p in packages if p.map_id == manifest["config"]["scene_map"][row["task"]["scene_id"]])
    gate = audit_case(Path(result["audit_path"]).parent, result, package)
    reader = current_acceptance_reader()
    require_map = "capabilities.place_memory" in manifest.get("plugins", {}).get("plugins", {}).get("enabled", [])
    request_issues = []
    for path in Path(result["run_dir"]).glob("debug/tool_loop/*/decision_*.json"):
        for model_round in read(path).get("model_rounds", []):
            issues, _ = reader.request_checks(model_round, require_map=require_map,
                                             plan_mode=manifest.get("plan_mode", "optional"),
                                             require_place_list=manifest.get("place_memory_admission_required", False))
            request_issues.extend({"source": str(path), "reason": issue} for issue in issues)
    gate.update(request_issues=request_issues, request_reader_sha256=digest(Path(reader.__file__)),
        controller_reader_sha256=digest(Path(__file__)))
    gate["passed"] = gate["passed"] and not request_issues
    previous = directory / "native_audit.json"
    if previous.exists():
        archive = directory / "native_audit_history" / (digest(previous) + ".json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            shutil.copy2(previous, archive)
    write(previous, gate)
    return gate


def run(root, resume=False):
    manifest = guard(root)
    load_credentials(manifest)
    admission = read(root / "admission.json")
    if not admission_passed(admission, manifest.get("plan_mode", "optional")):
        raise ValueError("Successful real-image native-tool admission required")
    if not (root / "historical_request_replay.json").exists() or not read(root / "historical_request_replay.json")["passed"]:
        raise ValueError("Successful historical request replay required before navigation")
    if "capabilities.pixel_depth" in manifest.get("plugins", {}).get("plugins", {}).get("enabled", []):
        if not read(root / "pixel_place_admission.json")["passed"]:
            raise ValueError("Successful pixel/place admission required before navigation")
    if manifest.get("place_memory_admission_required"):
        if read(root / "place_memory_admission.json").get("status") != "passed":
            raise ValueError("Successful place-memory provider admission required")
    with campaign_lock(root):
        progress = read(root / "progress.json")
        progress["status"] = "running"
        write(root / "progress.json", progress)
        # An audit-reader repair must verify the existing terminal case before
        # another flight. Terminal navigation outcomes are never rerun here.
        for index in manifest.get("technical_gate_indices", [0, 1]):
            row = progress["cases"][index]
            if row["status"] in {"completed", "model_error"} and not audit_gate(root, manifest, row)["passed"]:
                progress["status"] = "paused_audit"
                write(root / "progress.json", progress)
                return report(root)
        for index, row in enumerate(progress["cases"]):
            number = next_attempt(row, resume)
            if number is None:
                continue
            guard(root)
            if shutil.disk_usage(root).free < 5 * 1024**3:
                raise RuntimeError("Less than 5 GiB available before next flight")
            directory = root / "cases" / f"{index + 1:02d}" / f"attempt_{number:02d}"
            row["status"] = "running"
            attempt = {"number": number, "directory": str(directory), "status": "running"}
            row["attempts"].append(attempt)
            write(root / "progress.json", progress)
            print(f"START {index + 1}/{len(manifest['tasks'])} {row['task']['episode_id']} attempt {number}", flush=True)
            process = supervise([sys.executable, "-u", "-X", "faulthandler",
                                 str(root / "source/tests/manual/aerial_vln_plan_context_probe.py"),
                                 "child", "--root", str(root), "--index", str(index),
                                 "--attempt", str(number)], directory, monitor_gpu=True)
            try:
                cleanup = cleanup_owned(directory / "runs", manifest["config"]["map_config_path"],
                                        process.get("descendants", []))
                verified = bool(cleanup) and all(c["remote_stop"].get("cleanup_verified")
                                                and c["endpoint"].get("status") == "unreachable" for c in cleanup)
                write(directory / "cleanup.json", {"verified": verified, "receipts": cleanup})
            except Exception as exc:
                verified = False
                write(directory / "cleanup.json", {"verified": False, "error": str(exc)})
            result = read(directory / "result.json") if (directory / "result.json").exists() else None
            status = "completed"
            infrastructure = (result or {}).get("infrastructure_error") or {}
            if not verified or process["exit_code"] != 0 or result is None or (infrastructure and infrastructure.get("stage") != "provider"):
                status = "infrastructure_error"
            elif result.get("status") == "error" or infrastructure.get("stage") == "provider":
                status = "model_error"
            attempt.update(status=status, cleanup_verified=verified, process=process, result=result)
            row["status"] = status
            write(directory / "receipt.json", attempt)
            write(root / "progress.json", progress)
            report(root)
            print(json.dumps({"index": index + 1, "status": status, "cleanup_verified": verified,
                              "metrics": result.get("metrics") if result else None}, ensure_ascii=False), flush=True)
            if status == "infrastructure_error":
                progress["status"] = "paused"
                write(root / "progress.json", progress)
                return report(root)
            if index in manifest.get("technical_gate_indices", [0, 1]):
                gate = audit_gate(root, manifest, row)
                if not gate["passed"]:
                    progress["status"] = "paused_audit"
                    write(root / "progress.json", progress)
                    return report(root)
        progress["status"] = "complete"
        write(root / "progress.json", progress)
    return report(root)


def token_summary(usages):
    """Missing provider usage is unknown, not a fabricated zero."""
    values = []
    for u in usages:
        if not isinstance(u, dict):
            continue
        value = u.get("input_tokens", u.get("prompt_tokens"))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(value)
    outputs = [u.get("output_tokens", u.get("completion_tokens")) for u in usages if isinstance(u, dict)]
    outputs = [v for v in outputs if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {"requests": len(usages), "requests_with_usage": len(values),
            "input_tokens_sum": sum(values) if values else None,
            "input_tokens_mean": statistics.mean(values) if values else None,
            "input_tokens_peak": max(values) if values else None,
            "requests_with_output_usage": len(outputs),
            "output_tokens_sum": sum(outputs) if outputs else None,
            "output_tokens_mean": statistics.mean(outputs) if outputs else None,
            "output_tokens_peak": max(outputs) if outputs else None}


def run_stats(result):
    """Read captured calls/results; do not infer model progress from the route."""
    run_dir = Path(result.get("run_dir") or "__missing_run__")
    transcript = run_dir / "memory/agent_transcript.jsonl"
    if not transcript.exists():
        return {"available": False, "navigation_usage": token_summary([]),
                "compression_usage": token_summary([])}
    events = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
    from tests.manual.flight_teaching_metrics import analyze
    teaching = analyze(events)
    from tests.manual.aerial_vln_pixel_place_audit import audit_capability_records
    # Fixtures/legacy logs without sensor stores retain unknown integrity; they
    # must not be interpreted as having verified photos or exact pixel bytes.
    integrity = audit_capability_records(run_dir, events) if (run_dir / "memory/frames.jsonl").exists() else None
    calls = {}; plan_calls = 0; flights = 0; last_update = 0; gaps = []; timeline = []
    physical_windows = 0; last_window_update = 0; window_gaps = []
    pixel = Counter(); places = Counter(); capability_timeline = []; previous_plan = {}
    current_observation = None
    trace_path = Path(result["audit_path"]).parent / "trace.json" if result.get("audit_path") else None
    physical_trace = read(trace_path) if trace_path and trace_path.exists() else {}
    traced_windows = {step["tool_call_id"]: window["window_index"]
        for window in physical_trace.get("physical_windows", [])
        for step in window.get("ordered_steps", []) if step.get("tool_call_id")}
    seen_windows = set(); observed_windows = 0
    from src.tools.diagnostics import legacy_result_diagnostic
    rejections = []; diagnostics = []; knowledge = Counter(); measurement = Counter()
    active_turn = None; incidents = set(); flight_statuses = Counter()
    for event in events:
        p = event.get("payload", {})
        if event.get("kind") == "state_checkpoint":
            current_observation = p.get("observation_id")
        if event.get("kind") == "state_checkpoint" and p.get("reason") == "physical_window_completed":
            observed_windows += 1
            if not traced_windows:
                physical_windows = observed_windows
        if event.get("kind") in {"assistant_turn", "provider_turn_rejected"}:
            active_turn = p.get("turn_id") or event.get("id")
        if event.get("kind") == "protocol_feedback" and p.get("diagnostic"):
            diagnosis = p["diagnostic"]
            diagnostics.append({"event_kind": "protocol_feedback", "diagnostic": diagnosis})
            incidents.add((diagnosis["phase"], diagnosis["code"], diagnosis.get("source_tool_call_id") or active_turn))
        if event.get("kind") == "assistant_turn":
            for call in p.get("tool_calls", []):
                calls[call.get("tool_call_id")] = call
                if call.get("tool_name") == "runtime.update_plan":
                    plan_calls += 1
        if event.get("kind") == "tool_result":
            if p.get("tool_call_id") in traced_windows:
                seen_windows.add(traced_windows[p["tool_call_id"]])
                physical_windows = len(seen_windows)
            diagnosis = legacy_result_diagnostic(p)
            if diagnosis is not None:
                diagnostics.append({"tool_call_id": p.get("tool_call_id"), "tool_name": p.get("tool_name"),
                                    "event_kind": "tool_result", "diagnostic": diagnosis})
                incidents.add((diagnosis["phase"], diagnosis["code"], diagnosis.get("source_tool_call_id") or active_turn))
            if p.get("tool_name") == "knowledge.load":
                knowledge[p.get("status", "unknown")] += 1
            if p.get("tool_name") == "skill.visual_grounding.measure_visible_depth":
                measurement["calls"] += 1
                measurement["execution_" + p.get("status", "unknown")] += 1
                data = p.get("data") or {}
                if isinstance(data.get("candidates"), list):
                    valid = sum(c.get("distance_status") == "ok" and c.get("distance_m") is not None for c in data["candidates"])
                    measurement["calls_with_measurement_data"] += 1
                    measurement["valid_distance_count"] += valid
                    measurement["calls_with_valid_distance"] += bool(valid)
                    for c in data["candidates"]:
                        measurement[c.get("distance_status", "unknown")] += 1
            if p.get("tool_name") == "skill.pixel_depth.measure_pixel_depth":
                data = p.get("data") or {}
                pixel["calls"] += 1
                pixel["execution_" + p.get("status", "unknown")] += 1
                if "distance_status" in data:
                    valid = data["distance_status"] == "ok" and data.get("distance_m") is not None
                    pixel["calls_with_measurement_data"] += 1
                    pixel["calls_with_valid_distance"] += valid
                    pixel["valid_distance_count"] += valid
                    pixel[data["distance_status"]] += 1
                measurement.update({"calls": 1, "execution_" + p.get("status", "unknown"): 1})
                if "distance_status" in data:
                    measurement.update({"calls_with_measurement_data": 1,
                        "calls_with_valid_distance": int(valid), "valid_distance_count": int(valid), data["distance_status"]: 1})
            if p.get("tool_name", "").startswith("skill.place_memory."):
                action = p["tool_name"].rsplit(".", 1)[-1]
                places[action + "_calls"] += 1
                places[action + "_" + p.get("status", "unknown")] += 1
            if p.get("tool_name", "").startswith(("skill.place_memory.", "skill.pixel_depth.")):
                capability_timeline.append({"event_id": event.get("id"), "tool_call_id": p.get("tool_call_id"),
                    "tool": p["tool_name"], "status": p.get("status"),
                    "args": calls.get(p.get("tool_call_id"), {}).get("args"), "data": p.get("data"),
                    "diagnostic": diagnosis, "model_inputs": p.get("model_inputs"),
                    "artifact_refs": p.get("artifact_refs")})
            if p.get("tool_name", "").startswith("uav.") and p.get("status") in {"ok", "adjusted"}:
                flights += 1
            if p.get("tool_name", "").startswith("uav."):
                flight_statuses[p.get("status", "unknown")] += 1
            if p.get("tool_name") == "runtime.update_plan":
                call = calls.get(p.get("tool_call_id"), {})
                if p.get("status") == "ok":
                    gaps.append(flights - last_update); last_update = flights
                    window_gaps.append(physical_windows - last_window_update); last_window_update = physical_windows
                    snapshot = (p.get("data") or {}).get("plan", {})
                    before_steps = {s["id"]: s for s in previous_plan.get("steps", [])}
                    corrections = [s["id"] for s in snapshot.get("steps", []) if s["id"] in before_steps
                                   and s.get("description") != before_steps[s["id"]].get("description")]
                    timeline.append({"event_id": event.get("id"), "flight_actions_before": flights,
                                     "tool_call_id": p.get("tool_call_id"),
                                     "observation_id": current_observation,
                                     "physical_windows_before": physical_windows, "description_corrections": corrections,
                                     "incremental": call.get("args", {}).get("operation") == "patch" or "updates" in call.get("args", {}),
                                     "args": call.get("args"), "data": p.get("data")})
                    previous_plan = snapshot
                else:
                    rejections.append({"event_id": event.get("id"), "message": p.get("message"),
                                       "args": call.get("args"), "diagnostic": diagnosis})
    nav = []; round_count = 0; delivered_maps = []; delivered_evidence = []; image_counts = []
    for path in sorted(run_dir.glob("debug/tool_loop/*/decision_*.json")):
        decision = read(path)
        for r in decision.get("model_rounds", []):
            round_count += 1
            nav.append((r.get("response") or {}).get("usage"))
            media = current_acceptance_reader().delivered_media(r) if "request_media" in r else None
            if isinstance(media, list):
                image_counts.append(len(media))
                delivered_maps.append([m for m in media if m.get("label", "").startswith("CONTEXT IMAGE ")
                    or m.get("context_image_classification") == "captured_latest_group"])
                delivered_evidence.extend(m for m in media if m.get("timeline_role") == "TOOL EVIDENCE")
    comp = {}
    for path in run_dir.glob("debug/history_compaction_calls/*.json"):
        value = read(path)
        key = value.get("call_id", str(path))
        if key not in comp or value.get("status") != "started":
            comp[key] = value.get("usage")
    final = (timeline[-1].get("data") or {}).get("plan", {}) if timeline else {}
    feedback = [e.get("payload", {}).get("message") for e in events if e.get("kind") == "protocol_feedback"]
    observed_rounds = sum(e.get("kind") in {"assistant_turn", "provider_turn_rejected"} for e in events)
    process_path = run_dir.parents[1] / "process.json"
    elapsed = read(process_path).get("elapsed_seconds") if process_path.exists() else None
    total_windows = physical_trace.get("physical_window_count", observed_windows)
    window_source = "explicit_trace_windows" if "physical_window_count" in physical_trace else "resulting_checkpoints"
    if not physical_trace and flight_statuses and not observed_windows:
        total_windows = None
        window_gaps = [None for _ in window_gaps]
        window_source = "unavailable_window_association"
    return {"available": True, **teaching, "plan_calls": plan_calls, "plan_commits": len(timeline),
            "incremental_commits": sum(t["incremental"] for t in timeline),
            "replace_commits": sum(not t["incremental"] for t in timeline),
            "description_correction_count": sum(len(t["description_corrections"]) for t in timeline),
            "final_revision": final.get("revision", 0), "final_plan": final or None,
            "first_plan_physical_windows_before": timeline[0]["physical_windows_before"] if timeline and total_windows is not None else None,
            "plan_created_before_first_physical_window": timeline[0]["physical_windows_before"] == 0 if timeline and total_windows is not None else None,
            "plan_rejections": rejections, "plan_timeline": timeline,
            "flight_actions_between_updates": gaps[1:], "flight_actions_after_last_update": flights-last_update,
            "physical_windows": total_windows, "observed_physical_windows": observed_windows,
            "physical_windows_source": window_source, "physical_windows_between_updates": window_gaps[1:],
            "physical_windows_after_last_update": total_windows-last_window_update if total_windows is not None else None,
            "unfinished_final_steps": [s for s in final.get("steps", []) if s.get("status") in {"pending", "in_progress"}],
            "planner_rounds": round_count if nav else observed_rounds,
            "planner_rounds_source": "captured_requests" if nav else "recorded_native_responses",
            "protocol_rejections": len(feedback), "rejection_messages": feedback,
            "elapsed_seconds": elapsed,
            "diagnostics": diagnostics,
            "diagnostic_counts": dict(Counter(d["diagnostic"]["code"] for d in diagnostics)),
            "diagnostic_incident_counts": dict(Counter(code for phase, code, source in incidents)),
            "diagnostic_counting": "Event counts include affected calls and protocol feedback; incidents deduplicate phase/code/source within a turn.",
            "flight_action_statuses": dict(flight_statuses),
            "plan_gap_counting": "Explicit trace windows count started interactions, including failures; resulting checkpoints are an independent observed count. A window does not prove movement occurred. No-plan cases are not labeled stale.",
            "knowledge_load_results": dict(knowledge), "measurement": dict(measurement),
            "pixel_measurement": dict(pixel), "place_memory": dict(places), "capability_timeline": capability_timeline,
            "capability_integrity": integrity,
            "context_images": {"requests_with_media_records": len(delivered_maps),
                "requests_with_unique_map": sum(len(m) == 1 for m in delivered_maps),
                "delivery_rate": sum(len(m) == 1 for m in delivered_maps) / len(delivered_maps) if delivered_maps else None,
                "tool_result_caption_requests": sum(any(m.get("context_label_variant") for m in group) for group in delivered_maps),
                "map_versions": list({m["sha256"]: m for group in delivered_maps for m in group}.values()),
                "peak_request_images": max(image_counts) if image_counts else None},
            "measurement_valid_rate": measurement["calls_with_valid_distance"] / measurement["calls_with_measurement_data"] if measurement["calls_with_measurement_data"] else None,
            "navigation_usage": token_summary(nav), "compression_usage": token_summary(list(comp.values()))}


def result_identity(result, source):
    """Re-exported copies share an actual run ID, not merely an episode ID."""
    run_id = str(result.get("run_dir") or "").replace("\\", "/").rstrip("/").split("/")[-1]
    return (result.get("episode_id"), run_id or str(Path(source).resolve()))


def deduplicate(records):
    unique = {}
    for record in records:
        key = result_identity(record["result"], record["source"])
        if key in unique:
            unique[key]["source_aliases"].append(record["source"])
        else:
            unique[key] = {**record, "source_aliases": [record["source"]]}
    return list(unique.values())


def history_inventory(roots, excluded, valid_episode_ids=None):
    """Read finalized results and campaign attempts, without large audit files."""
    records = []; errors = []; metadata_cache = {}; excluded = excluded.resolve()
    excluded_fixtures = []
    def metadata(path, boundary):
        nearest_experiment = None
        dataset = {}
        for parent in (path.parent, *path.parents):
            if not parent.is_relative_to(boundary):
                break
            candidate = parent / "manifest.json"
            if candidate.exists():
                if candidate not in metadata_cache:
                    try: metadata_cache[candidate] = read(candidate)
                    except (OSError, ValueError): metadata_cache[candidate] = {}
                m = metadata_cache[candidate]
                if nearest_experiment is None and (m.get("benchmark") == "aerial_vln" or m.get("source_hashes")):
                    nearest_experiment = parent
                    config = m.get("config") or next(iter((m.get("configs") or {}).values()), {})
                    if not isinstance(config, dict): config = {}
                    dataset = {"benchmark": m.get("benchmark"),
                               "dataset_version": m.get("dataset_version", config.get("dataset_version")),
                               "scene_map": m.get("scene_map", config.get("scene_map"))}
                profile = m.get("model_config", m.get("model", {}))
                if isinstance(profile, str): profile = {"model_name": profile}
                if isinstance(profile, dict) and profile.get("model_name"):
                    plugin_path = next((parent / s / "configs/plugins.json" for s in ("source", "source_snapshot", "snapshot")
                                        if (parent / s / "configs/plugins.json").exists()), None)
                    enabled = read(plugin_path).get("plugins", {}).get("enabled") if plugin_path else None
                    return {**dataset, "experiment": str(parent), "model": profile.get("model_name"),
                            "api_protocol": profile.get("api_protocol"), "base_url": profile.get("base_url"),
                            "plugins": enabled, "source_revision": m.get("fingerprint"),
                            "model_profile": {k:v for k,v in profile.items() if k not in {"api_key", "token", "password"}}}
        return {**dataset, "experiment": str(nearest_experiment or path.parent), "model": None, "api_protocol": None,
                "plugins": None, "source_revision": None}
    for boundary in roots:
        if not boundary.exists():
            continue
        for base, dirs, files in os.walk(boundary):
            path = Path(base)
            dirs[:] = [d for d in dirs if d not in SKIP and not (path/d).resolve().is_relative_to(excluded)]
            if path.resolve().is_relative_to(excluded):
                continue
            for name in ("case_result.json", "progress.json", "summary.json"):
                if name not in files: continue
                source = path / name
                try: x = read(source)
                except (OSError, ValueError) as exc:
                    errors.append({"path": str(source), "error": str(exc)}); continue
                values = []
                if name == "case_result.json": values = [x]
                elif name == "progress.json" and isinstance(x.get("tasks"), dict):
                    for task in x["tasks"].values():
                        for a in task.get("attempts", []):
                            if isinstance(a.get("result"), dict): values.append(a["result"])
                        if isinstance(task.get("result"), dict): values.append(task["result"])
                elif name == "summary.json":
                    values = [c for c in x.get("cases", []) if isinstance(c,dict) and "episode_id" in c]
                for r in values:
                    if not r.get("episode_id") or "metrics" not in r or "__" in r["episode_id"]:
                        continue
                    meta = metadata(source,boundary)
                    outside_baseline = valid_episode_ids is not None and r["episode_id"] not in valid_episode_ids
                    if outside_baseline and not (meta.get("benchmark") == "aerial_vln"
                                                and meta.get("dataset_version") and len(r["episode_id"]) == 30
                                                and r["episode_id"].isalnum() and r["episode_id"].isupper()):
                        excluded_fixtures.append({"source":str(source), "episode_id":r["episode_id"],
                                                  "reason":"identity_not_in_baseline_or_declared_historical_dataset"})
                        continue
                    # A summary lacking run identity is only an index when a
                    # canonical case result exists; it is not another flight.
                    if name == "summary.json" and not r.get("run_dir") and (path / "cases" / r["episode_id"] / "case_result.json").exists():
                        continue
                    r = dict(r)
                    r.setdefault("dataset_split", x.get("dataset_split", (x.get("dataset") or {}).get("split", "unknown")))
                    records.append({"source": str(source), "result": r, **meta,
                                    "outside_primary_dataset": outside_baseline})
    canonical = [r for r in records if r["result"].get("run_dir")]
    indexes = []; flights = []
    for record in records:
        r = record["result"]
        is_index = (Path(record["source"]).name == "summary.json" and not r.get("run_dir")
                    and any(c["result"]["episode_id"] == r["episode_id"]
                            and c["result"].get("dataset_split") in {r.get("dataset_split"), "unknown"}
                            and (c["experiment"] == record["experiment"]
                                 or Path(c["source"]).is_relative_to(Path(record["source"]).parent))
                            for c in canonical))
        if is_index:
            indexes.append({"source": record["source"], "episode_id": r["episode_id"],
                            "reason": "summary_without_run_identity_indexes_existing_flights"})
        else:
            flights.append(record)
    return {"records": deduplicate(flights), "summary_indexes": indexes,
            "excluded_fixtures": excluded_fixtures, "read_errors": errors, "roots": [str(p) for p in roots]}


def near_boundary_stop(result):
    """Return a tri-state review label without inferring legacy termination.

    Experimental status (e.g. ``completed``) is not mission completion. Require
    the recorded mission flags/error and finite horizontal distance; absent facts
    remain unknown even if other facts suggest failure. Never recalculate SR.
    """
    if not isinstance(result, dict):
        return None
    distance = (result.get("metrics") or {}).get("distance_to_goal")
    if (isinstance(distance, bool) or not isinstance(distance, (int, float))
            or not math.isfinite(distance) or distance < 0):
        return None
    if (not isinstance(result.get("completed"), bool)
            or not isinstance(result.get("stopped"), bool) or "error" not in result):
        return None
    normal = (result["completed"] and not result["stopped"]
              and result["error"] is None and not result.get("infrastructure_error"))
    return normal and SUCCESS_DISTANCE < distance <= NEAR_BOUNDARY_DISTANCE


def arrival_distance_review(result):
    """Report signed distance to the official threshold without rescoring SR.

    Positive gap is outside 20 m; negative gap is inside. A scored failure in
    the five-metre exterior band is a near miss even if it reported blocked.
    Missing scores/distances remain unknown, never become inferred successes.
    """
    metrics = (result or {}).get("metrics") or {}
    distance = metrics.get("distance_to_goal")
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0:
        return {"distance_m": None, "gap_to_20m_m": None, "excess_over_20m_m": None, "near_boundary_failure": None}
    gap = distance - SUCCESS_DISTANCE
    success = metrics.get("success")
    return {"distance_m": distance, "gap_to_20m_m": gap, "excess_over_20m_m": max(0.0, gap),
            "near_boundary_failure": (success is False and 0 < gap <= 5.0) if isinstance(success, bool) else None}


def annotated_result(result):
    """Copy a report view, leaving raw results and cached inventories unchanged."""
    return {**result, "near_boundary_stop": near_boundary_stop(result),
            "arrival_distance_review": arrival_distance_review(result)} if result is not None else None


def aggregate(rows):
    """Counts retain outcomes/errors; numeric aggregates expose coverage."""
    result = {"cases": len(rows), "statuses": dict(Counter(r.get("status", "unknown") for r in rows)),
              "infrastructure_errors": sum(bool(r.get("infrastructure_error")) for r in rows)}
    labels = [near_boundary_stop(row) for row in rows]
    result["near_boundary_stop"] = {"count": labels.count(True),
                                    "known_count": sum(v is not None for v in labels),
                                    "unknown_count": labels.count(None)}
    for key in METRICS:
        values = [r.get("metrics", {}).get(key) for r in rows]
        values = [float(v) for v in values if isinstance(v, (int,float))]
        result[key] = {"count": len(values), "mean": statistics.mean(values) if values else None,
                       "sum": sum(values) if values else None}
    return result


def stats_overview(stats):
    """Aggregate recorded diagnostics with coverage, never fill missing usage."""
    result = {"cases": len(stats)}
    for key in ("plan_calls", "plan_commits", "incremental_commits", "replace_commits", "description_correction_count",
                "physical_windows", "planner_rounds", "elapsed_seconds"):
        values = [s.get(key) for s in stats if isinstance(s.get(key), (int, float))]
        result[key] = {"count": len(values), "sum": sum(values) if values else None,
                       "mean": statistics.mean(values) if values else None}
    known_rejections = [s["plan_rejections"] for s in stats if "plan_rejections" in s]
    result["plan_rejections"] = sum(len(v) for v in known_rejections) if known_rejections else None
    result["plan_rejections_coverage"] = len(known_rejections)
    result["plan_usage"] = {"known_cases": sum(s.get("available", False) for s in stats),
        "cases_using_plan": sum(bool(s.get("plan_calls")) for s in stats),
        "cases_with_commits": sum(bool(s.get("plan_commits")) for s in stats),
        "creation_timing_known_cases": sum(s.get("plan_created_before_first_physical_window") is not None for s in stats),
        "cases_created_before_first_physical_window": sum(s.get("plan_created_before_first_physical_window") is True for s in stats)}
    maps = [s["context_images"] for s in stats if "context_images" in s]
    map_total = sum(m["requests_with_media_records"] for m in maps)
    map_unique = sum(m["requests_with_unique_map"] for m in maps)
    result["context_images"] = {"requests_with_media_records": map_total,
        "requests_with_unique_map": map_unique, "delivery_rate": map_unique / map_total if map_total else None,
        "tool_result_caption_requests": sum(m.get("tool_result_caption_requests", 0) for m in maps) if maps else None}
    for key in ("diagnostic_counts", "diagnostic_incident_counts", "knowledge_load_results", "knowledge_by_name",
                "measurement", "pixel_measurement", "place_memory", "flight_action_statuses"):
        known = [s[key] for s in stats if key in s]
        counter = Counter()
        for value in known:
            counter.update(value)
        result[key] = {"cases_with_records": len(known), "counts": dict(counter) if known else None}
    measurements = result["measurement"]["counts"] or {}
    denominator = measurements.get("calls_with_measurement_data", 0)
    result["measurement_valid_rate"] = measurements.get("calls_with_valid_distance", 0) / denominator if denominator else None
    choices = [s["movement_choices"] for s in stats if "movement_choices" in s]
    movement = {"cases_with_records": len(choices),
        "requested_planar_over_20m": sum(s["requested_planar_over_20m"] for s in choices) if choices else None,
        "notice": "Command lengths, not actual displacement or clearance. 20 m is a diagnostic threshold only."}
    for key in ("requested_planar", "requested_vertical_abs", "requested_yaw_abs"):
        distributions = [s[key] for s in choices]
        count = sum(d["count"] for d in distributions)
        peaks = [d["peak"] for d in distributions if d["peak"] is not None]
        movement[key] = {"count": count,
            "mean": sum(d["mean"] * d["count"] for d in distributions if d["count"]) / count if count else None,
            "peak": max(peaks) if peaks else None}
    axes = Counter()
    for choice in choices:
        axes.update(choice["axis_choices"])
    movement["axis_choices"] = dict(axes) if choices else None
    maxima = [s["max_calls_in_observed_window"] for s in stats if s.get("max_calls_in_observed_window") is not None]
    result.update(movement_choices=movement, max_calls_in_observed_window=max(maxima) if maxima else None)
    for key in ("navigation_usage", "compression_usage"):
        usages = [s.get(key, {}) for s in stats]
        known = [u for u in usages if u.get("input_tokens_sum") is not None]
        total = sum(u["input_tokens_sum"] for u in known) if known else None
        count = sum(u.get("requests_with_usage", 0) for u in usages)
        peaks = [u["input_tokens_peak"] for u in known if u.get("input_tokens_peak") is not None]
        result[key] = {"cases_with_usage": len(known), "captured_requests": sum(u.get("requests", 0) for u in usages),
                       "requests_with_usage": count, "input_tokens_sum": total,
                       "input_tokens_mean": total / count if count else None,
                       "input_tokens_peak": max(peaks) if peaks else None}
        output_known = [u for u in usages if u.get("output_tokens_sum") is not None]
        output_total = sum(u["output_tokens_sum"] for u in output_known) if output_known else None
        output_count = sum(u.get("requests_with_output_usage", 0) for u in usages)
        output_peaks = [u["output_tokens_peak"] for u in output_known if u.get("output_tokens_peak") is not None]
        result[key].update(cases_with_output_usage=len(output_known), requests_with_output_usage=output_count,
            output_tokens_sum=output_total, output_tokens_mean=output_total / output_count if output_count else None,
            output_tokens_peak=max(output_peaks) if output_peaks else None)
    return result


def report(root, refresh_history=False):
    manifest = read(root / "manifest.json")
    progress = read_experiment_progress(root)
    inventory_path = root / "history_inventory.json"
    if refresh_history or not inventory_path.exists():
        workspace = Path(manifest["workspace_root"])
        roots = [workspace / "evaluation/benchmarks/aerial_vln/outputs", workspace / "runs",
                 Path(manifest["baseline"]).parent]
        prior_progress = read(Path(manifest["baseline"]) / "progress.json")
        valid_ids = {v["task"]["episode_id"] for v in prior_progress["tasks"].values()}
        write(inventory_path, history_inventory(roots, root.resolve(), valid_ids))
    inventory = read(inventory_path)
    inventory["records"] = [{**r, "result": annotated_result(r["result"])} for r in inventory["records"]]
    baseline = read(Path(manifest["baseline"]) / "progress.json")
    prior = {v["task"]["episode_id"]:annotated_result(v["result"]) for v in baseline["tasks"].values()
             if v["task"]["split"] == "val_seen" and v["status"] == "scored"}
    secondary = dict(prior)
    if manifest.get("secondary_root"):
        selected = read_experiment_progress(Path(manifest["secondary_root"]))
        secondary.update({r["task"]["episode_id"]: annotated_result(r["attempts"][-1]["result"])
                          for r in selected["cases"] if r["attempts"]})
    if manifest.get("previous_root"):
        previous = read_experiment_progress(Path(manifest["previous_root"]))
        prior.update({r["task"]["episode_id"]: annotated_result(r["attempts"][-1]["result"]) for r in previous["cases"]})
    cohorts = manifest.get("cohorts") or [{"name": "original10", "episode_ids": [r["task"]["episode_id"] for r in progress["cases"]],
        "primary_baseline": manifest.get("previous_root") or manifest["baseline"], "secondary_baseline": manifest["baseline"]}]
    cohort_by_id = {cid: cohort for cohort in cohorts for cid in cohort["episode_ids"]}
    pairs = []
    for index,row in enumerate(progress["cases"]):
        cid = row["task"]["episode_id"]
        current = annotated_result(row["attempts"][-1].get("result")) if row["attempts"] else None
        if row["status"] not in {"completed", "model_error"}: current = None
        old = (secondary if cohort_by_id[cid]["primary_baseline"] == manifest["baseline"] else prior)[cid]
        deltas = {k:float(current["metrics"][k])-float(old["metrics"][k])
                  for k in METRICS if current and isinstance(current.get("metrics",{}).get(k),(int,float))
                  and isinstance(old.get("metrics",{}).get(k),(int,float))}
        pairs.append({"index": index+1, "task": row["task"], "status": row["status"],
                      "cohort": cohort_by_id[cid]["name"], "baseline_source": cohort_by_id[cid]["primary_baseline"],
                      "current": current, "baseline": old, "metric_deltas": deltas,
                      "current_stats": run_stats(current) if current else None,
                      "baseline_stats": run_stats(old), "attempts": [
                          {**a, "result": annotated_result(a.get("result"))} for a in row["attempts"]],
                      "secondary_baseline": secondary[cid],
                      "secondary_stats": run_stats(secondary[cid]),
                      "secondary_metric_deltas": {k: float(current["metrics"][k]) - float(secondary[cid]["metrics"][k])
                          for k in METRICS if current and isinstance(current.get("metrics", {}).get(k), (int, float))
                          and isinstance(secondary[cid].get("metrics", {}).get(k), (int, float))},
                      "history": [r for r in inventory["records"] if r["result"]["episode_id"]==cid
                                  and r["result"].get("dataset_split") in {"val_seen","unknown"}]})
    current_rows=[p["current"] for p in pairs if p["current"] is not None]
    # Legacy manifests explicitly identify cohort sources; use the common full
    # baseline only when they differ. Never select a source by sample count.
    sources = {c["primary_baseline"] for c in cohorts}
    overall_source = manifest.get("overall_primary_baseline") or (next(iter(sources)) if len(sources) == 1 else manifest["baseline"])
    common_baseline = overall_source == manifest["baseline"]
    for pair in pairs:
        pair["overall_baseline"] = pair["secondary_baseline"] if common_baseline else pair["baseline"]
    matched_old=[p["secondary_baseline"] if common_baseline else p["baseline"] for p in pairs if p["current"] is not None]
    transitions = Counter()
    for p in pairs:
        old_success = (p["secondary_baseline"] if common_baseline else p["baseline"]).get("metrics", {}).get("success")
        new_success = (p["current"] or {}).get("metrics", {}).get("success")
        if isinstance(new_success, bool) and isinstance(old_success, bool):
            transitions[("success" if old_success else "failure") + "_to_" +
                        ("success" if new_success else "failure")] += 1
        else:
            transitions["unscored_or_pending"] += 1
    groups={}
    current_by_episode = {p["task"]["episode_id"]: p["current"] for p in pairs if p["current"]}
    for r in inventory["records"]:
        group=(r["experiment"],r.get("model"),r.get("api_protocol"),r["result"].get("protocol_variant"),
               tuple(r.get("plugins") or ()),r.get("source_revision"),r.get("dataset_version"))
        groups.setdefault(group,[]).append(r["result"])
    comparison={"schema_version":"aerial_plan_context_comparison_v1", "status":progress["status"],
                "selected_cases":len(pairs), "terminal_cases":len(current_rows), "pairs":pairs,
                "current":aggregate(current_rows), "matched_baseline":aggregate(matched_old),
                "success_transitions": dict(transitions),
                "current_diagnostics": stats_overview([p["current_stats"] for p in pairs if p["current_stats"]]),
                "baseline_diagnostics": stats_overview([p["secondary_stats"] if common_baseline else p["baseline_stats"] for p in pairs]),
                "ten_case_baseline":aggregate([p["secondary_baseline"] if common_baseline else p["baseline"] for p in pairs]),
                "primary_baseline_root": overall_source,
                "overall_comparison_contract": "All selected cases paired to one full-baseline version; cohort primary comparisons remain separate." if common_baseline else "Selected cohort primary comparison.",
                "secondary_baseline_root": manifest.get("secondary_root", manifest["baseline"]),
                "secondary_ten_case_baseline":aggregate([p["secondary_baseline"] for p in pairs]),
                "secondary_baseline_diagnostics":stats_overview([p["secondary_stats"] for p in pairs]),
                "primary_background": aggregate([v["result"] for v in baseline["tasks"].values()
                                                 if v["task"]["split"] == "val_seen" and v["status"] == "scored"]),
                "historical_groups":[{"experiment":k[0],"model":k[1],"api_protocol":k[2],
                                      "protocol_variant":k[3],"plugins":list(k[4]) or None,
                                      "source_revision":k[5],"dataset_version":k[6],"summary":aggregate(v)} for k,v in groups.items()],
                "history_scope": {"unique_attempts":len(inventory["records"]),
                                  "summary_indexes":len(inventory.get("summary_indexes", [])),
                                  "outside_primary_dataset":sum(r.get("outside_primary_dataset",False) for r in inventory["records"]),
                                  "excluded_fixtures":len(inventory.get("excluded_fixtures", []))},
                "api_admission": read(root / "admission.json") if (root / "admission.json").exists() else None,
                "report_generator_sha256": digest(Path(__file__)),
                "report_generated_at": datetime.now(timezone.utc).isoformat(),
                "arrival_annotation_contract": {
                    "field": "near_boundary_stop", "lower_distance_exclusive": SUCCESS_DISTANCE,
                    "upper_distance_inclusive": NEAR_BOUNDARY_DISTANCE,
                    "requires": "finite horizontal distance; completed=true, stopped=false, error=null; no infrastructure error",
                    "missing_information": None, "changes_formal_scoring": False,
                    "note": "Recorded flights retain their frozen protocol; later arrival guidance never rewrites historical requests."},
                "history_read_errors":inventory["read_errors"],"limits":manifest["limits"]}
    comparison["plan_mode"] = manifest.get("plan_mode", "optional")
    comparison["execution_condition"] = manifest.get("retest")
    comparison["distance_review_contract"] = {
        "field": "arrival_distance_review", "gap_formula": "horizontal_distance_m - 20",
        "near_boundary_failure": "recorded success=false and 20 < horizontal_distance_m <= 25",
        "includes_blocked_outcomes": True, "missing_information": None, "changes_formal_scoring": False}
    # Provider failures retain trajectories but may have no navigation score.
    # Keep them as terminal outcomes, never silently turn missing SR into False.
    scored_pairs = [p for p in pairs if isinstance((p["current"] or {}).get("metrics", {}).get("success"), bool)
                    and isinstance(p["overall_baseline"].get("metrics", {}).get("success"), bool)]
    comparison["terminal_unscored_cases"] = [p["index"] for p in pairs if p["current"] is not None
        and not isinstance(p["current"].get("metrics", {}).get("success"), bool)]
    comparison["scored_pair_comparison"] = {
        "indices": [p["index"] for p in scored_pairs],
        "current": aggregate([p["current"] for p in scored_pairs]),
        "baseline": aggregate([p["overall_baseline"] for p in scored_pairs]),
        "notice": "Same-case comparison restricted to known SR on both sides; unscored terminal outcomes remain in the full experiment."}
    comparison["experiment_overrides"] = manifest.get("experiment_overrides", [])
    comparison["references"] = []
    for reference in manifest.get("reference_roots", []):
        selected = match_previous_cohort(read(Path(reference["root"]) / "progress.json"),
                                        manifest["tasks"], allow_subset=True)
        ref_pairs = []
        for p, old_row in zip(pairs, selected):
            old_result = annotated_result(old_row["attempts"][-1]["result"])
            ref_pairs.append({"index": p["index"], "task": p["task"], "result": old_result,
                              "stats": run_stats(old_result), "metric_deltas": {
                k: float(p["current"]["metrics"][k]) - float(old_result["metrics"][k])
                for k in METRICS if p["current"] and isinstance(p["current"].get("metrics", {}).get(k), (int, float))
                and isinstance(old_result.get("metrics", {}).get(k), (int, float))}})
        comparison["references"].append({**reference, "pairs": ref_pairs,
            "summary": aggregate([p["result"] for p in ref_pairs]),
            "diagnostics": stats_overview([p["stats"] for p in ref_pairs])})
    comparison["cohorts"] = []
    for cohort in cohorts:
        rows = [p for p in pairs if p["cohort"] == cohort["name"]]
        complete = [p for p in rows if p["current"] is not None]
        scored = [p for p in complete if isinstance(p["current"].get("metrics", {}).get("success"), bool)
                  and isinstance(p["baseline"].get("metrics", {}).get("success"), bool)]
        transitions_by_group = Counter()
        for p in rows:
            old_sr = p["baseline"].get("metrics", {}).get("success")
            new_sr = (p["current"] or {}).get("metrics", {}).get("success")
            key = (("success" if old_sr else "failure") + "_to_" + ("success" if new_sr else "failure")) if isinstance(old_sr, bool) and isinstance(new_sr, bool) else "unscored_or_pending"
            transitions_by_group[key] += 1
        group_stats = stats_overview([p["current_stats"] for p in complete])
        comparison["cohorts"].append({**cohort, "selected_cases": len(rows), "terminal_cases": len(complete),
            "current": aggregate([p["current"] for p in complete]), "baseline": aggregate([p["baseline"] for p in rows]),
            "scored_pair_comparison": {"indices": [p["index"] for p in scored],
                "current": aggregate([p["current"] for p in scored]),
                "baseline": aggregate([p["baseline"] for p in scored])},
            "matched_baseline": aggregate([p["baseline"] for p in complete]),
            "secondary_baseline_summary": aggregate([p["secondary_baseline"] for p in rows]),
            "success_transitions": dict(transitions_by_group), "current_diagnostics": group_stats,
            "baseline_diagnostics": stats_overview([p["baseline_stats"] for p in rows]),
            "plan_review_notice": "本轮关闭在线计划；计划覆盖不作为验收指标。" if comparison["plan_mode"] == "off" else
                "导航中计划验证覆盖不足" if cohort["name"] == "hard10" and group_stats["plan_usage"]["cases_with_commits"] < min(3, len(rows)) else
                "Plan receipts and unfinished steps need human semantic review; the program does not judge route correctness."})
    # Compare every overlapping attempt; never choose the best historical run.
    # Unmatched episodes contribute only to each group's background summary.
    for group, values in zip(comparison["historical_groups"], groups.values()):
        # Non-overlapping flights also retain individual labels, but never enter
        # paired metric deltas or success transitions for these selected cases.
        group["arrival_annotations"] = [{"episode_id": old.get("episode_id"),
            "run_dir": old.get("run_dir"), "status": old.get("status"),
            "distance_to_goal": (old.get("metrics") or {}).get("distance_to_goal"),
            "near_boundary_stop": old["near_boundary_stop"]} for old in values]
        matched = [old for old in values if old.get("episode_id") in current_by_episode
                   and old.get("dataset_split", "unknown") in {"val_seen", "unknown"}]
        group["overlap_cases"] = len({old["episode_id"] for old in matched})
        group["overlap_attempts"] = len(matched)
        group["overlap_summary"] = aggregate(matched)
        group["paired_attempts"] = [{"episode_id": old["episode_id"], "historical_run": old.get("run_dir"),
            "historical_status": old.get("status"), "infrastructure_error": old.get("infrastructure_error"),
            "current_run": current_by_episode[old["episode_id"]].get("run_dir"),
            "historical_metrics": old.get("metrics"), "current_metrics": current_by_episode[old["episode_id"]].get("metrics"),
            "historical_near_boundary_stop": old["near_boundary_stop"],
            "current_near_boundary_stop": current_by_episode[old["episode_id"]]["near_boundary_stop"],
            "metric_deltas": {key: float(current_by_episode[old["episode_id"]]["metrics"][key]) - float(old["metrics"][key])
                for key in METRICS if isinstance((old.get("metrics") or {}).get(key), (int, float))
                and isinstance((current_by_episode[old["episode_id"]].get("metrics") or {}).get(key), (int, float))}}
            for old in matched]
    plan_rows = [p for p in pairs if (p.get("current_stats") or {}).get("plan_commits")]
    comparison["plan_review_summary"] = {
        "cases_using_committed_plan": len(plan_rows),
        "cases_ending_with_unfinished_steps": [p["index"] for p in plan_rows if p["current_stats"]["unfinished_final_steps"]],
        "scored_success_with_unfinished_steps": [p["index"] for p in plan_rows
            if p["current_stats"]["unfinished_final_steps"] and (p["current"].get("metrics") or {}).get("success") is True],
        "scored_failure_with_all_steps_completed": [p["index"] for p in plan_rows
            if p["current_stats"].get("final_plan", {}).get("steps")
            and all(s["status"] == "completed" for s in p["current_stats"]["final_plan"]["steps"])
            and (p["current"].get("metrics") or {}).get("success") is False],
        "cancelled_final_steps": sum(s["status"] == "cancelled" for p in plan_rows
            for s in p["current_stats"]["final_plan"]["steps"]),
        "route_semantics": "Not automatically evaluated. Plan progress is model-authored; compare receipts with the linked actions and observations.",
        "termination": "Terminal tools end the loop. Raw plans are not automatically completed or rewritten afterward."}
    review_path = root / "review_notes.json"
    comparison["record_review"] = read(review_path) if review_path.is_file() else None
    comparison["record_review_sha256"] = digest(review_path) if review_path.is_file() else None
    write(root / "comparison.json",comparison)
    case_reader = current_report_reader("aerial_vln_case_diagnostics")
    comparison["report_reader_hashes"] = {name: digest(WORKSPACE / "tests/manual" / (name + ".py")) for name in (
        "aerial_vln_plan_context_probe", "aerial_vln_case_diagnostics", "aerial_vln_context_module_audit",
        "aerial_vln_plan_context_acceptance", "aerial_vln_pixel_place_audit")}
    write(root / "comparison.json", comparison)
    case_reader.write_case_diagnostics(root, pairs, link)
    from tests.manual.aerial_vln_pixel_place_audit import write_place_report
    write_place_report(root, pairs, write)
    render_report(root,comparison)
    from tests.manual.aerial_vln_context_module_audit import write_module_report
    historical = read(root / "historical_request_replay.json") if (root / "historical_request_replay.json").exists() else None
    write_module_report(root, pairs, historical, write)
    return comparison


def link(root,path):
    path=Path(path)
    try: return quote(os.path.relpath(path,root).replace("\\","/"),safe="/:.")
    except ValueError: return path.resolve().as_uri()


def metric_rate(summary, metric):
    """Display known metric outcomes, not the number of attempted flights."""
    total = summary[metric]["sum"]
    value = "未知" if total is None else f"{total:g}"
    return f'{value}/{summary[metric]["count"]}'


def render_report(root, data):
    """Render only recorded evidence; readable UTF-8 and no embedded images."""
    def fmt(v): return "未知" if v is None else f"{v:.4f}" if isinstance(v,float) else str(v)
    def arrival(result):
        label = near_boundary_stop(result)
        return "临界到达" if label is True else "未知" if label is None else "无"
    def arrival_count(summary):
        label = summary["near_boundary_stop"]
        return f'{label["count"]} 例（已知 {label["known_count"]}，未知 {label["unknown_count"]}）'
    n = data["selected_cases"]
    cohort_names = {group["name"] for group in data.get("cohorts", [])}
    cohort_description = ("困难十例：锁定的 high 清单；主对照和第二对照均使用下方明确记录的实验来源及相同案例。"
        if cohort_names == {"hard10"} else
        "原十例（8 个低、2 个中）及十例 high；分组和总体分别使用报告明确记录的主对照来源。"
        if n == 20 else "原十例：8 个低难度、2 个中难度。")
    if any(group.get("source_episode_ids") for group in data.get("cohorts", [])):
        cohort_description = f"锁定清单的有序子集，共 {n} 例；与上轮同案例配对，保留源清单身份和哈希。"
    scope_notice = ("相同冻结导航源码、相同每例GPU通道，仅取消任务重叠；单次复测仍不能排除模型和渲染波动。"
        if (data.get("execution_condition") or {}).get("kind") == "serial_same_navigation_source"
        else "整体代码存在其他差异，不能单独归因于地点图、像素测距或计划修改。")
    rows=[]; md=[f"# {n} 例真实 AerialVLN 对比", "",f"状态：{data['status']}；终态：{data['terminal_cases']}/{n}。", "",
                cohort_description + scope_notice,
                "", "|序号|案例|状态|旧 SR|新 SR|旧 nDTW|新 nDTW|Δ距离(m)|旧临界到达|新临界到达|第二对照临界到达|终点距离(m)|距20米差值(m)|临界失败|", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    findings = []
    if (data.get("execution_condition") or {}).get("kind") == "serial_same_navigation_source":
        findings.append('本组为同冻结导航源码的串行复测：每例沿用主对照GPU/端口通道，同一时刻最多一个导航任务。两个宿主工作线程仅保留通道可用性，不代表两个任务并行；未包含导航后照片送达补修。')
    findings.append('距20米差值=终点水平距离−20米；正值表示超出，负值表示在范围内。临界失败单列已评分失败且20<距离≤25米的案例，包括主动停止；不修改正式SR，未知评分不补零。')
    unavailable_depth = sum(len(((pair.get("current_stats") or {}).get("capability_integrity") or {}).get("unavailable_checks", []))
                            for pair in data["pairs"])
    if unavailable_depth:
        findings.append(f'{unavailable_depth} 次像素测距的原始深度数组已由并行控制器在生成报告后清理；保留回执、RGB和标记图，但原始深度数值复核为未知，不算核验通过。')
    findings.append('临界到达仅标记正常结束且 20 < 最终水平距离 ≤ 25 米；缺少距离或明确结束信息时为未知。'
                    '正式 SR 与成功／失败转换沿用原 20 米规则。约 10 米建议不反向作用于历史请求；各次导航实际使用的说明以其冻结请求为准。')
    findings.append(f'当前临界到达 {arrival_count(data["current"])}；'
                    f'主对照 {arrival_count(data["ten_case_baseline"])}；'
                    f'第二对照 {arrival_count(data["secondary_ten_case_baseline"])}。')
    findings.append(f'报告生成器 SHA-256：{data["report_generator_sha256"]}；生成时间：{data["report_generated_at"]}。')
    if data.get("terminal_unscored_cases"):
        paired = data["scored_pair_comparison"]
        findings.append('已终止但没有有效SR评分的案例：' + ', '.join(map(str, data["terminal_unscored_cases"]))
            + '。保留错误，不填零、不重跑；表中分母为该指标有效评分数。')
        findings.append(f'有效配对 {len(paired["indices"])} 例：SR {metric_rate(paired["baseline"], "success")} → '
            f'{metric_rate(paired["current"], "success")}；Oracle '
            f'{metric_rate(paired["baseline"], "oracle_success")} → {metric_rate(paired["current"], "oracle_success")}。'
            f'这不是完整{n}例的成功率估计；未评分案例仍在实验终态和成本统计中。')
    if (data["current"]["success"]["count"] == data["selected_cases"]
            and data["matched_baseline"]["success"]["count"] == data["selected_cases"]):
        current = data["current"]; prior = data["matched_baseline"]
        sr_delta = current["success"]["sum"] - prior["success"]["sum"]
        direction = "提高" if sr_delta > 0 else "下降" if sr_delta < 0 else "持平"
        findings.append(f'相同 {n} 例 SR {fmt(prior["success"]["sum"])}/{n} → {fmt(current["success"]["sum"])}/{n}；'
                        f'Oracle SR {fmt(prior["oracle_success"]["sum"])}/{n} → {fmt(current["oracle_success"]["sum"])}/{n}；'
                        f'nDTW {fmt(prior["ndtw"]["mean"])} → {fmt(current["ndtw"]["mean"])}。'
                        f'SR {direction}。这组小样本不能证明某一项修改的因果收益。')
        old_key = "overall_baseline"
        regressed = [str(p["index"]) for p in data["pairs"] if p[old_key].get("metrics", {}).get("success") is True
                     and (p["current"] or {}).get("metrics", {}).get("success") is False]
        improved = [str(p["index"]) for p in data["pairs"] if p[old_key].get("metrics", {}).get("success") is False
                    and (p["current"] or {}).get("metrics", {}).get("success") is True]
        findings.append(f'失败→成功案例：{", ".join(improved) or "无"}；成功→失败案例：{", ".join(regressed) or "无"}。')
        close_blocked = [str(p["index"]) for p in data["pairs"] if (p["current"] or {}).get("stop_source") == "runtime.report_blocked"
                         and (p["current"] or {}).get("metrics", {}).get("distance_to_goal", float("inf"))
                         < (p["current"] or {}).get("metrics", {}).get("success_distance", 0)]
        if close_blocked:
            findings.append('案例 '+", ".join(close_blocked)+' 终点位于评分成功半径内，但模型报告阻塞，SR 仍为失败。评分接近不代表指令路线和地标身份已经被模型正确判断。')
    diagnostics = data["current_diagnostics"]
    measurement = (diagnostics.get("measurement") or {}).get("counts") or {}
    if measurement.get("calls"):
        findings.append(f'测距工具执行 {measurement["calls"]} 次，执行成功 {measurement.get("execution_ok", 0)} 次，'
                        f'有效标量距离 {measurement.get("valid_distance_count", 0)} 个。'
                        '执行成功不能解释为测量有效，表面射线距离也不是目标水平距离。')
        if "inconsistent_depth_surfaces" in measurement:
            findings.append(f'旧候选测距记录中的深度表面不一致候选：{measurement["inconsistent_depth_surfaces"]} 个。')
    old_usage = data["baseline_diagnostics"]["navigation_usage"].get("input_tokens_sum")
    new_usage = diagnostics["navigation_usage"].get("input_tokens_sum")
    if old_usage and new_usage is not None:
        findings.append(f'导航实际输入 token：{old_usage:,} → {new_usage:,}（{(new_usage/old_usage-1)*100:+.2f}%）；'
                        f'模型轮数：{data["baseline_diagnostics"]["planner_rounds"]["sum"]} → {diagnostics["planner_rounds"]["sum"]}。'
                        '这是当前版本整体请求成本，不能单独归因于参数精简。')
    secondary = data["secondary_ten_case_baseline"]
    findings.append(f'旧全量实验同 {n} 例：SR {fmt(secondary["success"]["sum"])}/{n}，'
                    f'Oracle SR {fmt(secondary["oracle_success"]["sum"])}/{n}。逐例指标和差值保存在 comparison.json。')
    findings.append(f'地点图实际送达率：{fmt(diagnostics["context_images"]["delivery_rate"])}；'
                    f'像素测距：{json.dumps(diagnostics["pixel_measurement"]["counts"], ensure_ascii=False)}；'
                    f'地点保存／查阅：{json.dumps(diagnostics["place_memory"]["counts"], ensure_ascii=False)}。')
    if diagnostics["context_images"].get("tool_result_caption_requests"):
        findings.append(f'地点图有 {diagnostics["context_images"]["tool_result_caption_requests"]} 份请求使用工具结果图片标签。'
            '地图身份按原始请求图片哈希与该次已捕获的 latest_group 选择核验；原标签保留，未将旧请求改写为 CONTEXT IMAGE。')
    if diagnostics["plan_calls"]["sum"]:
        creation = diagnostics["plan_usage"]
        findings.append(f'首个物理窗口前提交计划：{creation["cases_created_before_first_physical_window"]} 例；'
            f'创建时机可读 {creation["creation_timing_known_cases"]} 例。窗口记录不能证明模型内部核对或阶段语义正确。')
        findings.append(f'正式导航计划调用 {diagnostics["plan_calls"]["sum"]} 次，成功提交 '
                        f'{diagnostics["plan_commits"]["sum"]} 次，拒绝 {diagnostics["plan_rejections"]} 次。'
                        '原始参数及拒绝原因在各案例时间线中保留；不能用单元测试通过代替真实接口可用性。')
        unfinished = [p for p in data["pairs"] if (p.get("current_stats") or {}).get("unfinished_final_steps")]
        if unfinished:
            endings = [f'{p["index"]}（未完成 {len(p["current_stats"]["unfinished_final_steps"])} 步；'
                f'最后提交后 {fmt(p["current_stats"].get("physical_windows_after_last_update"))} 个窗口）' for p in unfinished]
            findings.append('结束时仍有未完成计划步骤的案例：' + '、'.join(endings) + '。'
                '这是收尾状态与提交间隔的记录，不是对自然语言路线正确性的自动判定。')
        review = data.get('plan_review_summary') or {}
        if review.get('scored_success_with_unfinished_steps'):
            findings.append('评分成功但计划仍有未完成步骤的案例：' +
                '、'.join(str(v) for v in review['scored_success_with_unfinished_steps']) + '。终止工具结束回合后不会自动补写计划。')
        if review.get('scored_failure_with_all_steps_completed'):
            findings.append('计划全部步骤标记完成但评分失败的案例：' +
                '、'.join(str(v) for v in review['scored_failure_with_all_steps_completed']) + '。计划完成状态不是环境证据。')
    md[5:5] = ["", *findings, ""]
    cohort_html = '<h2>分组主对照</h2><table><tr><th>分组</th><th>终态</th><th>旧 SR / Oracle</th><th>新 SR / Oracle</th><th>主对照来源</th></tr>'
    cohort_md = ["", "## 分组主对照", "", "|分组|终态|旧 SR / Oracle|新 SR / Oracle|主对照|", "|---|---|---|---|---|"]
    cohort_pair_notes = []
    for group in data.get("cohorts", []):
        size = group["selected_cases"]
        cells = [group["name"], f'{group["terminal_cases"]}/{size}',
            f'{metric_rate(group["baseline"], "success")} / {metric_rate(group["baseline"], "oracle_success")}',
            f'{metric_rate(group["current"], "success")} / {metric_rate(group["current"], "oracle_success")}', group["primary_baseline"]]
        cohort_md.append("|" + "|".join(cells) + "|")
        cohort_html += "<tr>" + "".join("<td>" + html.escape(c) + "</td>" for c in cells) + "</tr>"
        paired = group.get("scored_pair_comparison")
        if paired and len(paired["indices"]) != size:
            paired_note = (f'{group["name"]} 有效配对 {len(paired["indices"])} 例：SR '
                f'{metric_rate(paired["baseline"], "success")} → {metric_rate(paired["current"], "success")}；'
                f'Oracle {metric_rate(paired["baseline"], "oracle_success")} → '
                f'{metric_rate(paired["current"], "oracle_success")}。其余案例不补造评分。')
            cohort_pair_notes.append(paired_note)
            cohort_html += '<tr><td colspan="5">' + html.escape(paired_note) + '</td></tr>'
        if group["name"] == "hard10":
            findings.append(group["plan_review_notice"])
            cohort_md.extend(["", group["plan_review_notice"], ""])
    cohort_md.extend(["", *cohort_pair_notes, ""])
    cohort_html += '</table><p>逐例表的旧值使用所属分组主对照；' + (
        '跨组汇总使用报告中明确记录的主对照版本。' if len(data.get("cohorts", [])) > 1 else '本组汇总使用同一主对照版本。') + '</p>'
    cost_header = ['分组', '旧 / 新 nDTW', '旧 / 新物理窗口', '旧 / 新模型轮数', '旧 / 新输入 token', '新计划覆盖 / 提交']
    cohort_html += '<h2>分组路径、请求成本与计划覆盖</h2><table><tr>' + ''.join('<th>' + c + '</th>' for c in cost_header) + '</tr>'
    cohort_md.extend(['', '## 分组路径、请求成本与计划覆盖', '', '|' + '|'.join(cost_header) + '|', '|---|---|---|---|---|---|'])
    for group in data.get('cohorts', []):
        before, after = group['baseline_diagnostics'], group['current_diagnostics']
        def totals(key):
            return fmt(before[key]['sum']) + ' / ' + fmt(after[key]['sum'])
        def tokens(view):
            usage = view['navigation_usage']
            return fmt(usage['input_tokens_sum']) + f'（usage覆盖 {usage["cases_with_usage"]} 例）'
        cells = [group['name'], fmt(group['baseline']['ndtw']['mean']) + ' / ' + fmt(group['current']['ndtw']['mean']),
            totals('physical_windows'), totals('planner_rounds'), tokens(before) + ' / ' + tokens(after),
            f'{after["plan_usage"]["cases_with_commits"]}/{group["terminal_cases"]} 例 / {fmt(after["plan_commits"]["sum"])} 次']
        cohort_md.append('|' + '|'.join(cells) + '|')
        cohort_html += '<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>'
    cohort_html += '</table><p>输入 token 使用实际 API usage；缺失为未知，覆盖不足时不能作为全组总成本比较。压缩请求另列。</p>'
    table_index = next(i for i, line in enumerate(md) if line.startswith("|序号|"))
    md[table_index:table_index] = [*cohort_md, ""]
    # Explicit reference roots supplement the primary and full-baseline pairs.
    # Each is one recorded version, not a best-of selection from all history.
    reference_html = '<h2>显式版本对照</h2><p>当前计划模式：' + html.escape(data.get('plan_mode', 'optional')) + '</p>'
    reference_md = ['', '## 显式版本对照', '', '当前计划模式：' + data.get('plan_mode', 'optional'), '',
        '|版本|SR|Oracle|nDTW|平均终点距离|物理窗口|模型轮数|输入token|输出token|',
        '|---|---|---|---|---|---|---|---|---|']
    reference_html += '<table><tr>' + ''.join('<th>'+c+'</th>' for c in
        ['版本', 'SR', 'Oracle', 'nDTW', '平均终点距离', '物理窗口', '模型轮数', '输入token', '输出token']) + '</tr>'
    versions = [(Path(data['primary_baseline_root']).name, data['ten_case_baseline'], data['baseline_diagnostics']),
                *[(Path(r['root']).name, r['summary'], r['diagnostics']) for r in data.get('references', [])],
                ('当前：' + data.get('plan_mode', 'optional'), data['current'], data['current_diagnostics'])]
    for label, summary, stats in versions:
        cells = [label, metric_rate(summary, 'success'),
            metric_rate(summary, 'oracle_success'), fmt(summary['ndtw']['mean']),
            fmt(summary['distance_to_goal']['mean']), fmt(stats['physical_windows']['sum']),
            fmt(stats['planner_rounds']['sum']), fmt(stats['navigation_usage']['input_tokens_sum']),
            fmt(stats['navigation_usage'].get('output_tokens_sum'))]
        reference_md.append('|'+'|'.join(cells)+'|')
        reference_html += '<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>'
    reference_html += '</table>'
    for reference in data.get('references', []):
        reference_html += '<p>'+html.escape(Path(reference['root']).name)+'：'
        for p in reference['pairs']:
            target = link(root, p['result']['run_dir']) if p['result'].get('run_dir') else None
            if target:
                reference_html += f'<a href="{html.escape(target, quote=True)}">案例 {p["index"]}</a> '
        reference_html += '</p>'
    md[table_index:table_index] = [*reference_md, '']
    cohort_html += reference_html
    record_review_html = ''
    review_md = []
    if data.get('record_review'):
        review_md = ['', '## 记录审阅', '', '以下分析与自动指标分开，原始请求、计划及评分保持不变。', '']
        record_review_html = '<h2>记录审阅</h2><p>分析与自动指标分开；原始请求、计划及评分保持不变。</p>'
        for note in data['record_review'].get('notes', []):
            review_md.extend([f'案例 {note["case"]}：{note["title"]}。', '', note['recorded_facts'], '', note['assessment'], '', note['suggested_next_change'], ''])
            record_review_html += '<h3>案例 ' + str(note['case']) + '：' + html.escape(note['title']) + '</h3>'
            record_review_html += ''.join('<p>' + html.escape(note[key]) + '</p>' for key in ('recorded_facts', 'assessment', 'suggested_next_change'))
            for i, target in enumerate(note.get('evidence_links', []), 1):
                review_md.append(f'[证据 {i}]({target})')
                record_review_html += f'<a href="{html.escape(target, quote=True)}">证据 {i}</a> '
        review_md.extend(['', '[完整记录审阅及来源](review_notes.json)。', ''])
    details=[]
    for p in data["pairs"]:
        old=p["baseline"].get("metrics",{}); new=(p["current"] or {}).get("metrics",{})
        result=p["current"]
        outcome = f'{p["status"]} / {result.get("status", "unknown")}' if result else p["status"]
        cells=[p["index"],p["task"]["episode_id"],outcome,old.get("success"),new.get("success"),
               old.get("ndtw"),new.get("ndtw"),p["metric_deltas"].get("distance_to_goal"),
               arrival(p["baseline"]),arrival(result),arrival(p["secondary_baseline"])]
        distance_review = arrival_distance_review(result)
        cells.extend(distance_review[key] for key in ("distance_m", "gap_to_20m_m", "near_boundary_failure"))
        md.append("|"+"|".join(fmt(v) for v in cells)+"|")
        review=None
        if result:
            review=Path(result["audit_path"]).parent / "review.html"
        cells[1]=f'<a href="#case{p["index"]}">{html.escape(str(cells[1]))}</a>'
        rows.append("<tr>"+"".join("<td>"+(v if i==1 else html.escape(fmt(v)))+"</td>" for i,v in enumerate(cells))+"</tr>")
        links=[]
        diagnostic_page = root / f'case_diagnostics_{p["index"]:02d}.html'
        if diagnostic_page.exists():
            links.append(f'<a href="{link(root,diagnostic_page)}">计划、测距与地图时间线</a>')
        if review and review.exists(): links.append(f'<a href="{link(root,review)}">轨迹、图片、完整模型请求</a>')
        if result:
            transcript=Path(result["run_dir"])/"memory/agent_transcript.jsonl"
            links.append(f'<a href="{link(root,transcript)}">原始 Transcript</a>')
            stats = p["current_stats"] or {}
            for media in (stats.get("context_images") or {}).get("map_versions", []):
                if media.get("asset_ref"):
                    links.append(f'<a href="{link(root,Path(result["run_dir"])/media["asset_ref"])}">地点图 {html.escape(media.get("frame_id", ""))}</a>')
            for receipt in stats.get("capability_timeline", []):
                for model_input in receipt.get("model_inputs") or []:
                    ref = model_input.get("media_ref") or {}
                    if ref.get("ref"):
                        links.append(f'<a href="{link(root,Path(result["run_dir"])/ref["ref"])}">{html.escape(receipt["tool"])} {html.escape(ref.get("frame_id", ""))}</a>')
        old_review = (p["baseline"].get("report") or {}).get("path")
        if old_review and Path(old_review).is_file():
            links.append(f'<a href="{link(root,old_review)}">同案例旧基线轨迹</a>')
        for attempt in p["attempts"]:
            directory = Path(attempt["directory"])
            for name, label in (("cleanup.json", "清理"), ("process.json", "进程"), ("result.json", "原始结果")):
                target = directory / name
                if target.is_file():
                    links.append(f'<a href="{link(root,target)}">尝试 {attempt["number"]} {label}（{attempt["status"]}）</a>')
        history_links = []
        for r in p["history"]:
            old_page = (r["result"].get("report") or {}).get("path")
            if not old_page and r["result"].get("audit_path"):
                old_page = str(Path(r["result"]["audit_path"]).parent / "review.html")
            target = Path(old_page) if old_page and Path(old_page).is_file() else Path(r["source"])
            history_links.append(f'<li><a href="{link(root,target)}">{html.escape(r["experiment"])} · {html.escape(r.get("model") or "模型未记录")}</a> · 临界标记：{arrival(r["result"])}</li>')
        metric_rows = [(key, old.get(key), new.get(key), p["metric_deltas"].get(key)) for key in METRICS]
        for key in ("planner_rounds", "elapsed_seconds", "plan_calls", "plan_commits", "incremental_commits", "final_revision"):
            before = (p["baseline_stats"] or {}).get(key)
            after = (p["current_stats"] or {}).get(key)
            delta = after - before if isinstance(before, (int,float)) and isinstance(after, (int,float)) else None
            metric_rows.append((key, before, after, delta))
        pair_table = '<table><tr><th>指标</th><th>旧基线</th><th>当前版本</th><th>差值</th></tr>' + "".join(
            "<tr>" + "".join("<td>" + html.escape(fmt(v)) + "</td>" for v in values) + "</tr>"
            for values in metric_rows) + "</table>"
        pair_table += '<p>第二对照：旧全量实验同案例</p><pre>' + html.escape(json.dumps({
            "metrics": p["secondary_baseline"].get("metrics"), "deltas": p["secondary_metric_deltas"],
            "near_boundary_stop": p["secondary_baseline"]["near_boundary_stop"],
            "diagnostics": p["secondary_stats"]}, ensure_ascii=False, indent=2)) + '</pre>'
        details.append(f'<details id="case{p["index"]}"><summary>{p["index"]}. {p["task"]["episode_id"]}：计划、token、历史尝试</summary>'
                       +" · ".join(links)+pair_table+"<ul>"+"".join(history_links)+"</ul><pre>"+html.escape(json.dumps({
                           "navigation_outcome": {k:(result or {}).get(k) for k in (
                               "status", "completed", "stopped_reason", "stop_source", "error")},
                           "current_stats":p["current_stats"],"baseline_stats":p["baseline_stats"],
                           "history":[{"source":r["source"],"model":r.get("model"),"api_protocol":r.get("api_protocol"),
                                       "plugins":r.get("plugins"),"metrics":r["result"].get("metrics"),
                                       "status":r["result"].get("status"),"near_boundary_stop":r["result"]["near_boundary_stop"],
                                       "source_aliases":r["source_aliases"]} for r in p["history"]]
                       },ensure_ascii=False,indent=2))+"</pre></details>")
    md.extend(review_md)
    summary_rows = []
    md.extend(["", "## 同案例指标", "", "|指标|旧基线|当前版本|", "|---|---|---|"])
    for key, label in (("success", "SR"), ("oracle_success", "Oracle SR"), ("ndtw", "nDTW"),
                       ("sdtw", "sDTW"), ("distance_to_goal", "终点距离（米）"),
                       ("path_length", "路径长度（米）"), ("uav_action_steps", "飞行动作数")):
        values = [data[group][key] for group in ("matched_baseline", "current")]
        cells = [label] + [f'{fmt(v["mean"])}（覆盖 {v["count"]} 例）' for v in values]
        md.append("|" + "|".join(cells) + "|")
        summary_rows.append("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in cells) + "</tr>")
    transitions = data["success_transitions"]
    transition_text = (f'失败→成功 {transitions.get("failure_to_success", 0)} 例；'
                       f'成功→失败 {transitions.get("success_to_failure", 0)} 例；'
                       f'成功保持 {transitions.get("success_to_success", 0)} 例；'
                       f'失败保持 {transitions.get("failure_to_failure", 0)} 例；'
                       f'未评分或未完成 {transitions.get("unscored_or_pending", 0)} 例。')
    md.extend(["", transition_text, "", "## 计划、请求与耗时", "",
               "表中 completed 表示实验已结束，导航成功以 SR 为准；stopped 为模型报告阻塞等导航终止。", "",
               "耗时为每例受监督进程的实测墙钟时间，包含启动及清理；各尝试分别保留。token 是原始 API usage 的观测总量；旧实验未捕获的 usage 为未知，不能推断减少。", "",
               "```json", json.dumps({k:data[k] for k in ("current_diagnostics", "baseline_diagnostics")},ensure_ascii=False,indent=2), "```"])
    md.extend(["", "## 汇总", "", "```json", json.dumps({k:data[k] for k in (
        "current","matched_baseline","ten_case_baseline")},ensure_ascii=False,indent=2),"```",
        "", "完整历史分组、计划更新、token 与尝试记录见 comparison.json / comparison.html。"])
    background = data["primary_background"]
    history_note = (f'全量基线背景：val_seen 完成 {background["cases"]} 例，成功 {fmt(background["success"]["sum"])} 例。'
                    '历史分组按独立尝试统计，包含重试和基础设施中断；不同数据集、协议、模型或插件分别展示，不能将尝试汇总当作官方单次全量成绩。')
    md.extend(["", history_note])
    md.extend(["", "## 历史实验临界标记", "", "|实验|模型|临界到达（全部独立尝试）|同案例临界到达|",
               "|---|---|---|---|"])
    for group in data["historical_groups"]:
        md.append("|" + "|".join(str(v).replace("|", "\\|") for v in (
            group["experiment"], group.get("model") or "未知", arrival_count(group["summary"]),
            arrival_count(group["overlap_summary"]))) + "|")
    md.extend(["", "逐次历史尝试的临界标签和原始证据链接见 comparison.html；全部标记保存在 comparison.json。"])
    md.extend(["", "|案例序号|历史实验|原始证据|临界标记|", "|---|---|---|---|"])
    for pair in data["pairs"]:
        for record in pair["history"]:
            experiment = record["experiment"].replace("|", "\\|")
            md.append(f'|{pair["index"]}|{experiment}|[原始记录]({link(root,record["source"])})|{arrival(record["result"])}|')
    if (root / "teaching_review.html").is_file():
        md.extend(["", "[知识与动作反馈审阅](teaching_review.html)。"])
    admission = data.get("api_admission") or {}
    if admission.get("plan_probe_status") == "failed":
        warning = "计划专项准入未通过；原始参数及拒绝诊断已保留。导航不会启动。"
        md.extend(["", "## 计划工具准入问题", "", warning])
    atomic_write_text(root/"comparison.md","\n".join(md)+"\n")
    document=f'<!doctype html><meta charset="utf-8"><title>{n} 例 AerialVLN 对比</title><style>body{{font:15px system-ui;margin:32px}}td,th{{padding:8px;border:1px solid #ddd}}table{{border-collapse:collapse}}pre{{white-space:pre-wrap}}details{{margin:20px 0}}</style>'
    document+=f'<h1>{n} 例真实 AerialVLN 对比</h1><p>状态：{data["status"]}；终态 {data["terminal_cases"]}/{n}。总体对照：{html.escape(data["primary_baseline_root"])}</p><p>{html.escape(cohort_description + scope_notice)}缺失 token 标记为未知。</p>'
    document += cohort_html
    document += record_review_html
    document += "".join("<p>" + html.escape(finding) + "</p>" for finding in findings)
    document+='<p><a href="comparison.json">完整比较 JSON</a> · <a href="comparison.md">中文报告</a> · <a href="place_memory_audit.html">Place memory audit</a> · <a href="context_modules.html">上下文模块与诊断审计</a> · <a href="history_inventory.json">全部历史运行索引</a> · <a href="manifest.json">冻结配置及源码哈希</a> · <a href="environment.json">GPU 环境</a></p><table><tr>'+"".join(f"<th>{v}</th>" for v in ("序号","案例","状态","旧 SR","新 SR","旧 nDTW","新 nDTW","Δ距离(m)","旧临界到达","新临界到达","第二对照临界到达","终点距离(m)","距20米差值(m)","临界失败"))+"</tr>"+"".join(rows)+"</table>"+"".join(details)
    if (root / "teaching_review.html").is_file():
        document += '<p><a href="teaching_review.html">知识与动作反馈审阅（原运行证据）</a></p>'
    if (root / "knowledge_behavior/report.json").is_file():
        document += '<p><a href="knowledge_behavior/report.json">四类真实上下文的知识行为检查（离线请求，不执行飞行）</a></p>'
    document += '<h2>同案例指标</h2><table><tr><th>指标</th><th>旧基线</th><th>当前版本</th></tr>' + "".join(summary_rows) + '</table><p>' + html.escape(transition_text) + '</p>'
    document += '<h2>计划、请求与耗时</h2><p>耗时包含进程启动和清理；token 只统计原始 API usage。captured_requests 是有记录的请求数，旧实验未捕获的实际请求数和 token 为未知。</p><pre>' + html.escape(json.dumps({k:data[k] for k in ("current_diagnostics", "baseline_diagnostics")},ensure_ascii=False,indent=2)) + '</pre>'
    if admission.get("plan_probe_status") == "failed":
        document += '<h2>计划工具准入问题</h2><p>'+html.escape(warning)+'</p><p><a href="admission.json">完整准入记录</a></p>'
    document+="<h2>历史实验分组</h2><p>"+html.escape(history_note)+"</p><pre>"+html.escape(json.dumps(data["historical_groups"],ensure_ascii=False,indent=2))+"</pre>"
    atomic_write_text(root/"comparison.html",document)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","admit","run","report","child"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    parser.add_argument("--previous-root",type=Path,default=DEFAULT_PREVIOUS)
    parser.add_argument("--case-ids-file",type=Path,default=WORKSPACE/"tests/manual/aerial_vln_simple_instruction10.json")
    parser.add_argument("--plugins-config", type=Path, help="Plugin profile copied only into the new snapshot")
    parser.add_argument("--hard-case-ids-file", type=Path, help="Append the locked ten high cases")
    parser.add_argument("--hard-only", action="store_true",
                        help="Run only --hard-case-ids-file; match its cohort inside the previous experiment")
    parser.add_argument("--subset-case-ids-file", type=Path,
                        help="Select an ordered non-empty subset after validating the complete selected cohorts")
    parser.add_argument("--plan-mode", choices=PLAN_MODES, default="optional",
                        help="Optional planning by default; off overlays only the new frozen snapshot")
    parser.add_argument("--reference-root", type=Path, action="append", default=[],
                        help="Additional complete historical experiment; paired by exact case identity")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--refresh-history",action="store_true")
    parser.add_argument("--index",type=int)
    parser.add_argument("--attempt",type=int)
    args=parser.parse_args();root=args.root.resolve()
    frozen_entry = root / "source/tests/manual/aerial_vln_plan_context_probe.py"
    if args.command in {"admit", "run", "child"} and frozen_entry.is_file() and Path(__file__).resolve() != frozen_entry.resolve():
        # Re-exec rather than blending a current controller with frozen classes.
        result = subprocess.run([sys.executable, "-u", str(frozen_entry), *sys.argv[1:]],
                                cwd=root / "source")
        raise SystemExit(result.returncode)
    if args.command != "prepare" and (root / "source").exists():
        os.chdir(root / "source")
    if args.command=="prepare":
        root=prepare(root,args.baseline.resolve(),args.case_ids_file.resolve(),args.previous_root.resolve(),
            args.plugins_config.resolve() if args.plugins_config else None,
            args.hard_case_ids_file.resolve() if args.hard_case_ids_file else None,
            hard_only=args.hard_only, plan_mode=args.plan_mode,
            reference_roots=args.reference_root,
            subset_case_ids_file=args.subset_case_ids_file.resolve() if args.subset_case_ids_file else None); value={"prepared":str(root)}
    elif args.command=="admit": value={"admission":admit(root)["status"]}
    elif args.command=="run":
        result=run(root,args.resume);value={"status":result["status"],"terminal_cases":result["terminal_cases"]}
    elif args.command=="child": child(root,args.index,args.attempt);value={"child":"complete"}
    else:
        result=report(root,args.refresh_history);value={"status":result["status"],"terminal_cases":result["terminal_cases"]}
    print(json.dumps(value,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
