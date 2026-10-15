"""Prepare a fixed paired cohort and dispatch it through the supervised lanes.

Sampling/provenance and configuration freezing are evaluation-owned. Workers
use production benchmark runners with a fresh Runtime per attempt. No scores,
historical outcomes or sampling strata enter planner inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BUDGETS = (10, 20, 30)
SEED = 42


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def quotas(counts, total, lower=None):
    """Hamilton proportional quotas, respecting already selected anchors."""
    lower = lower or {}
    population = sum(counts.values())
    if total > population or sum(lower.values()) > total:
        raise ValueError("Infeasible sample quota")
    result = {k: max(lower.get(k, 0), v * total // population) for k, v in counts.items()}
    # Allocate by deficit from the proportional target; anchors consume quota.
    while sum(result.values()) != total:
        if sum(result.values()) < total:
            pool = [k for k in counts if result[k] < counts[k]]
            key = min(pool, key=lambda k: (result[k] - counts[k] * total / population, str(k)))
            result[key] += 1
        else:
            pool = [k for k in counts if result[k] > lower.get(k, 0)]
            key = max(pool, key=lambda k: (result[k] - counts[k] * total / population, str(k)))
            result[key] -= 1
    return result


def sample_cell(pool, total, anchors, rng, *, by_scene=False):
    """Random fill from canonical sorted identities, without replacing anchors."""
    selected = [t for t in pool if (t["dataset"], t["split"], t["episode_id"]) in anchors]
    if not by_scene:
        remaining = sorted([t for t in pool if t not in selected], key=lambda t: t["episode_id"])
        return selected + rng.sample(remaining, total - len(selected))
    groups = defaultdict(list)
    for t in pool:
        groups[t["scene_id"]].append(t)
    allocation = quotas({k: len(v) for k, v in groups.items()}, total, Counter(t["scene_id"] for t in selected))
    return [t for scene in sorted(groups) for t in sample_cell(groups[scene], allocation[scene], anchors, rng)]


def historical_successes(population, roots):
    """Verify original instructions and truth; deduplicate exported run copies."""
    by_id = {t["episode_id"]: t for t in population}
    accepted, rejected = defaultdict(list), []
    inspected = 0
    for root in roots:
        for directory, folders, files in os.walk(root):
            folders[:] = [f for f in folders if f not in {
                "source", "source_snapshot", "controller_source", "__pycache__", ".git",
                "provider_media", "artifacts", "temp", "contexts", "frames", "depth"}]
            if "case_result.json" not in files:
                continue
            path = Path(directory) / "case_result.json"
            try:
                result = read(path)
                task = by_id.get(result.get("episode_id"))
                if not task or not (result.get("metrics") or {}).get("success"):
                    continue
                inspected += 1
                if result.get("infrastructure_error") or result.get("metric_error") or not result.get("completed"):
                    raise ValueError("invalid historical score")
                if result.get("dataset_split") not in {None, task["split"]}:
                    raise ValueError("split mismatch")
                transcript = Path(result.get("run_dir", "")) / "memory/agent_transcript.jsonl"
                if transcript.is_file():
                    events = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
                else:
                    audit = read(path.with_name("audit.json"))
                    events = audit.get("agent_transcript", [])
                    if isinstance(events, dict):
                        events = events.get("events", [])
                instructions = [e["payload"].get("instruction") for e in events if e.get("kind") == "user_instruction"]
                if instructions != [task["_case"].instruction]:
                    raise ValueError("original instruction unavailable or changed")
                truth_path = path.with_name("evaluation_truth.json")
                truth = read(truth_path)
                expected = task["_case"].evaluation_truth()
                if any(digest(truth.get(k)) != digest(v) for k, v in expected.items()):
                    raise ValueError("original goal/reference/target mismatch")
                key = (task["dataset"], task["split"], task["episode_id"])
                source = {"result_path": str(path.resolve()), "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "run_dir": result.get("run_dir"), "instruction_sha256": hashlib.sha256(instructions[0].encode()).hexdigest(),
                          "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest()}
                if not any(s["run_dir"] == source["run_dir"] for s in accepted[key]):
                    accepted[key].append(source)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejected.append({"path": str(path), "reason": str(exc)})
    return accepted, {"candidate_success_files": inspected, "unique_verified_cases": len(accepted), "rejected": rejected}


def select_samples(population, successes):
    """Exact agreed split/difficulty quotas; past successes only anchor selection."""
    rng = random.Random(SEED)
    aerial = [t for t in population if t["dataset"] == "aerial_vln"]
    eligible = sorted([t for t in aerial if t["difficulty"] in {"high", "medium"}
                       and (t["dataset"], t["split"], t["episode_id"]) in successes],
                      key=lambda t: (t["difficulty"] != "high", t["episode_id"]))
    if len(eligible) < 8:
        raise ValueError(f"Need eight verified medium/high Aerial anchors; found {len(eligible)}")
    anchors = {(t["dataset"], t["split"], t["episode_id"]) for t in eligible[:8]}
    chosen = []
    for split in ("val_seen", "val_unseen"):
        for difficulty, total in (("high", 35), ("medium", 15)):
            pool = [t for t in aerial if t["split"] == split and t["difficulty"] == difficulty]
            chosen.extend(sample_cell(pool, total, anchors, rng, by_scene=True))
    travel = [t for t in population if t["dataset"] == "travel_uav"]
    known = {(t["dataset"], t["split"], t["episode_id"]) for t in travel
             if (t["dataset"], t["split"], t["episode_id"]) in successes}
    if len(known) != 6:
        raise ValueError(f"Expected six verified TravelUAV anchors; found {len(known)}; audit before changing cohort")
    anchors |= known
    split_allocation = quotas(Counter(t["split"] for t in travel), 100, Counter(k[1] for k in known))
    for split, total in sorted(split_allocation.items()):
        pool = [t for t in travel if t["split"] == split]
        allocation = quotas(Counter(t["difficulty"] for t in pool), total,
                            Counter(t["difficulty"] for t in pool if (t["dataset"], split, t["episode_id"]) in known))
        for difficulty, n in sorted(allocation.items()):
            chosen.extend(sample_cell([t for t in pool if t["difficulty"] == difficulty], n, anchors, rng))
    for task in chosen:
        task["historical_success_anchor"] = (task["dataset"], task["split"], task["episode_id"]) in anchors
    return chosen


def freeze_source(output):
    """Use the common freezer with this experiment's explicit plugin profile."""
    from evaluation.benchmarks.core.campaign.support import freeze_source as freeze
    return freeze(output, root=ROOT, plugin_config=ROOT / "configs/plugins.pixel_map.json")


def prepare(output, travel_maps):
    from evaluation.benchmarks.core.campaign.support import environment_diagnostics
    from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
    from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
    from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue
    from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
    from evaluation.benchmarks.travel_uav.dataset import load_travel_uav_index
    from evaluation.benchmarks.core.report_io import write_json
    from src.agent.config import load_model_config
    from src.agent.config.model import read_env_file

    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise ValueError("Already prepared; use existing frozen experiment")
    for k, v in read_env_file(ROOT / ".env").items():
        if k.startswith("FLYAGENT_"):
            os.environ.setdefault(k, v)
    diagnostics = environment_diagnostics()
    if not diagnostics["cuda_available"]:
        raise RuntimeError("flyagent CUDA unavailable; no experiment started")
    population, configs, validations = [], {}, {}
    for split in ("val_seen", "val_unseen"):
        config = AerialVlnEvalConfig.from_file(ROOT / f"evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_{split}_full.example.json")
        config = replace(config, dataset_root=str((ROOT / config.dataset_root).resolve()),
                         max_cases=None, episode_ids=None, limit_per_difficulty=None)
        prepared = AerialVlnBenchmarkRunner(config).prepare_evaluation(require_execution_ready=False)
        configs[split] = config.to_dict()
        configs[split]["expected_split_sha256"] = prepared.provenance.split_sha256
        validations[split] = prepared.validation_dict()
        for case in prepared.cases:
            a = prepared.selection.assignment_by_episode[case.episode_id]
            population.append({"dataset": "aerial_vln", "split": split, "episode_id": case.episode_id,
                               "scene_id": case.scene_id, "difficulty": a.difficulty,
                               "reference_path_length_m": a.reference_path_length_m, "_case": case})
    travel_base = TravelUavEvalConfig.from_file(ROOT / "evaluation/benchmarks/travel_uav/configs/travel_uav_test_seen_quick.example.json")
    index = (ROOT / travel_base.index_path).resolve()
    for split in ("test_seen", "test_unseen_map", "test_unseen_object"):
        cases, provenance = load_travel_uav_index(index, split=split)
        configs[split] = replace(travel_base, split=split, index_path=str(index),
                                expected_index_sha256=provenance.index_sha256, limit_per_difficulty=None).to_dict()
        validations[split] = provenance.to_dict()
        population.extend({"dataset": "travel_uav", "split": split, "episode_id": c.episode_id,
                           "scene_id": c.map_name, "difficulty": c.difficulty,
                           "reference_path_length_m": c.reference_path_length_m, "_case": c} for c in cases)
    roots = [ROOT / "runs", Path("H:/FlyAgent-Eval"), ROOT / "evaluation/benchmarks/aerial_vln/outputs", ROOT / "evaluation/benchmarks/travel_uav/outputs"]
    successes, audit = historical_successes(population, roots)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "historical_selection_audit.json", audit)
    selected = select_samples(population, successes)
    samples = [{k: v for k, v in t.items() if k != "_case"} | {
        "instruction_sha256": hashlib.sha256(t["_case"].instruction.encode()).hexdigest(),
        "truth_digest": digest(t["_case"].evaluation_truth()),
        "historical_sources": successes.get((t["dataset"], t["split"], t["episode_id"]), [])
    } for t in selected]
    source = freeze_source(output)
    maps = read(ROOT / "configs/maps/aerial_vln_s_remote.example.json")
    travel_catalog = read(travel_maps)
    maps["maps"].extend(travel_catalog["maps"])
    if len({m["map_id"] for m in maps["maps"]}) != len(maps["maps"]):
        raise ValueError("Duplicate map IDs")
    map_path = source / "configs/maps/image_budget_mixed.json"
    write_json(map_path, maps)
    context = read(ROOT / "configs/context.json")
    context["history_projection"]["compression"]["enabled"] = False
    context_paths = {}
    for budget in BUDGETS:
        context["history_projection"]["media"]["max_request_images"] = budget
        path = source / f"configs/context.images_{budget}.json"
        write_json(path, context)
        context_paths[str(budget)] = str(path)
    for split, data in configs.items():
        data.update(env_path=str(ROOT / ".env"), map_config_path=str(map_path),
                    context_config_path=context_paths["20"], save_artifacts=True,
                    enable_action_planner_debug=True, generate_html_review=True)
        # Aerial's lightweight profile explicitly suppresses provider capture.
        # Full exchanges are required to prove actual media choices, calibrated
        # peaks and successful-attempt timing. All groups use the same audit
        # setting; it changes persistence, not model-visible inputs.
        data["report_profile"] = "audit"
        if split.startswith("test_"):
            # Derive routing from the actual local map catalog, including unseen maps.
            data["scene_map"] = {m["scene_id"]: m["map_id"] for m in travel_catalog["maps"]}
        ids = [t["episode_id"] for t in samples if t["split"] == split]
        data["case_ids" if split.startswith("test_") else "episode_ids"] = ids
    # Both lanes finish one entire budget before advancing. The same seeded
    # case order is used in each budget; only dispatch order changes.
    tasks = []
    ordering = samples[:]
    random.Random(SEED).shuffle(ordering)
    for budget in BUDGETS:
        tasks.extend({**task, "image_budget": budget} for task in ordering)
    # Twelve formal executions: two cases per dataset, all three budgets.
    pilots = [next(t for t in samples if t["dataset"] == d and t["difficulty"] == difficulty)
              for d, difficulty in (("aerial_vln", "high"), ("aerial_vln", "medium"), ("travel_uav", "hard"), ("travel_uav", "easy"))]
    technical = [t for p in pilots for t in tasks if t["dataset"] == p["dataset"] and t["episode_id"] == p["episode_id"]]
    hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob("*") if p.is_file()}
    manifest = {"version": 1, "experiment": "image_budget", "seed": SEED, "image_budgets": list(BUDGETS),
                "configs": configs, "context_configs": context_paths, "tasks": tasks,
                "technical_trial": technical, "counts": dict(Counter(t["split"] for t in tasks)),
                "model": load_model_config(ROOT / ".env").redacted_summary(), "environment": diagnostics,
                "source_hashes": hashes, "strict_workers": 2, "dispatch_concurrency": 2,
                "scheduling": {"mode": "budget_sequential", "group_field": "image_budget", "group_order": list(BUDGETS), "technical_trial_scope": "each_budget"},
                "compression_enabled": False, "terminal_provider_errors": False,
                "policy": {"episode_attempts": 3, "episode_deadline_sec": 7200, "no_progress_sec": 900},
                "accounting": "first_valid_task_execution_and_final_successful_request_attempt",
                "sampling": "Aerial 50 seen/50 unseen, 70 high/30 medium, 8 anchors; Travel 6 anchors + 94 proportional random",
                "completion_status": "awaiting_image_budget_choice"}
    manifest["fingerprint"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "source_hashes.json", hashes)
    write_json(output / "selected_cases.json", {"seed": SEED, "cases": samples, "counts": dict(Counter(t["dataset"] for t in samples))})
    write_json(output / "dataset_validation.json", validations)
    write_json(output / "environment.json", diagnostics)
    ParallelRecoveryQueue(output / "progress.json", tasks, manifest["fingerprint"])
    from evaluation.experiments.image_budget_report import write_report
    write_report(output)
    return {"output": str(output), "samples": len(samples), "executions": len(tasks), "technical_trial": len(technical),
            "anchors": dict(Counter(t["dataset"] for t in samples if t["historical_success_anchor"]))}


def execute_case(output, manifest, request, environment):
    """Compatibility callback; dataset dispatch now belongs to shared adapters."""
    from evaluation.benchmarks.core.campaign.parallel import execute_thread
    return execute_thread(output, manifest, request, environment)


def refreeze_unstarted(output):
    """Refresh prepared code before any flight, preserving exact sample IDs.

    This is a preparation operation, never an in-flight hot patch. Both source
    and progress identity change together under the controller's OS lock.
    """
    from evaluation.benchmarks.core.campaign.support import campaign_lock
    from evaluation.benchmarks.core.report_io import write_json
    output = Path(output).resolve()
    with campaign_lock(output):
        progress = read(output / "progress.json")
        if any(v["attempts"] for v in progress["tasks"].values()):
            raise ValueError("A started experiment cannot be refrozen")
        source = freeze_source(output)
        manifest = read(output / "manifest.json")
        manifest["source_hashes"] = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        manifest.pop("fingerprint")
        manifest["fingerprint"] = digest(manifest)
        progress["fingerprint"] = manifest["fingerprint"]
        write_json(output / "manifest.json", manifest)
        write_json(output / "source_hashes.json", manifest["source_hashes"])
        write_json(output / "progress.json", progress)
        # A fresh overlay/preflight must verify the new code identity.
        (output / "parallel/execution_overlay.json").unlink(missing_ok=True)
    from evaluation.experiments.image_budget_report import write_report
    write_report(output)
    return {"refrozen": str(output), "fingerprint": manifest["fingerprint"]}


def validate_captured_requests(run_dir, budget):
    """Technical gate on captured inputs; navigation success is never a gate."""
    from evaluation.experiments.compression_cost import effective_planner_calls
    errors = []
    calls = effective_planner_calls(run_dir)
    if not calls:
        errors.append("missing_decision_audit")
    for call in calls:
        name = call["call_id"]
        if call.get("total_images") is None or not 1 <= call["total_images"] <= budget:
            errors.append(name + ":image_budget")
        if call.get("map_images") != 1:
            errors.append(name + ":latest_map")
        if call.get("compression_enabled") is not False or call.get("compaction_status") != "disabled":
            errors.append(name + ":compression_enabled")
        if call.get("estimated_input_tokens") is None:
            errors.append(name + ":missing_estimate")
        if call["effective"] and (call.get("decision_seconds") is None or call.get("input_tokens") is None or call.get("output_tokens") is None):
            errors.append(name + ":missing_cost_measurement")
    if list((Path(run_dir) / "debug/history_compaction_calls").glob("*.json")):
        errors.append("unexpected_summary_call")
    return errors


def repair_reporting(output):
    """Freeze repaired controller/readers while preserving started flight source.

    Call only after the old controller has stopped. The original manifest,
    progress records, scores and source hashes remain untouched. Resuming the
    controller is a separate action so cleanup/resource checks remain owned by
    its normal recovery path.
    """
    from evaluation.benchmarks.core.campaign import parallel as controller
    output = Path(output).resolve()
    manifest = read(output / "manifest.json")
    if manifest.get("experiment") != "image_budget":
        raise ValueError("Expected an image-budget campaign")
    controller.verify_frozen(output, manifest)
    for filename in ("dispatcher.json", "parallel/host_process.json"):
        owner = controller.read_json(output / filename)
        if owner and controller.matching_process(owner) is not None:
            raise ValueError("Stop the owned controller/host before repairing reports")
    with controller.campaign_lock(output):
        progress_before = (output / "progress.json").read_bytes()
        manifest_before = (output / "manifest.json").read_bytes()
        old = read(output / "parallel/execution_overlay.json")
        controller.verify_overlay(old, manifest)
        controller.write_json(output / "parallel/report_repairs" / (old["controller_revision"] + ".json"), old)
        overlay = controller.prepare_extension(output, manifest)
        controller.verify_overlay(overlay, manifest)
        controller.install_report_readers(overlay, manifest)
        from evaluation.experiments.image_budget_report import write_report
        report = write_report(output)
        if (output / "progress.json").read_bytes() != progress_before or (output / "manifest.json").read_bytes() != manifest_before:
            raise RuntimeError("Report repair modified authoritative campaign state")
        controller.verify_frozen(output, manifest)
        queue = controller.ParallelRecoveryQueue(output / "progress.json", manifest["tasks"], manifest["fingerprint"])
        controller.write_exit_status(output, queue, workers=2, gate=False, host_pid=None)
        receipt = {"navigation_fingerprint": manifest["fingerprint"], "navigation_source_verified_unchanged": True,
                   "progress_verified_unchanged": True, "valid_executions": report["valid_executions"],
                   "report_reader_hashes": report["report_reader_hashes"], "controller_revision": overlay["controller_revision"],
                   "previous_controller_revision": old["controller_revision"], "repaired_at": controller.utc_now()}
        receipt["logging_profile"] = overlay.get("logging_profile")
        receipt["logging_adapter_sha256"] = overlay["source_hashes"].get("compact_logging.py")
        controller.write_json(output / "parallel/report_repair.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "report", "preflight", "refreeze-unstarted", "repair-reporting"))
    parser.add_argument("--output", type=Path, default=Path("H:/FlyAgent-Eval/image_budget_200_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--travel-maps", type=Path, default=ROOT / "configs/maps/travel_uav_remote.local")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.output, args.travel_maps), ensure_ascii=False), flush=True)
    elif args.command == "refreeze-unstarted":
        print(json.dumps(refreeze_unstarted(args.output), ensure_ascii=False), flush=True)
    elif args.command == "repair-reporting":
        print(json.dumps(repair_reporting(args.output), ensure_ascii=False), flush=True)
    elif args.command == "report":
        from evaluation.experiments.image_budget_report import write_report
        write_report(args.output.resolve())
    else:
        from tests.manual.image_budget_preflight import preflight
        print(json.dumps(preflight(args.output.resolve()), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
