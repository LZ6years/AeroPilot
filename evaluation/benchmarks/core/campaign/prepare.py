"""Prepare standalone benchmark campaigns through their dataset adapters.

Preparation freezes selected identities, configs and source before dispatch.
Offline validation uses the same adapters without launching a simulator/model.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from evaluation.benchmarks.core.campaign.compat import adapter_for
from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue
from evaluation.benchmarks.core.campaign.support import ROOT, environment_diagnostics, freeze_source
from evaluation.benchmarks.core.report_io import write_json
from src.agent.config import load_model_config
from src.agent.config.model import read_env_file


def load_selections(config_paths, dataset, *, root=ROOT):
    """Resolve workspace paths explicitly; adapters retain schema validation."""
    selections = []
    adapter = adapter_for({"version": 2, "dataset": dataset})
    for path in config_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for field in ("dataset_root", "index_path", "map_config_path", "observation_profile_path", "env_path", "context_config_path"):
            if data.get(field):
                value = Path(data[field])
                data[field] = str((value if value.is_absolute() else Path(root) / value).resolve())
        data.setdefault("env_path", str(Path(root) / ".env"))
        for name, value in read_env_file(data["env_path"]).items():
            if name.startswith("FLYAGENT_"):
                os.environ.setdefault(name, value)
        selection = adapter.prepare(data, require_execution_ready=False)
        for field in ("dataset_root", "index_path", "map_config_path", "observation_profile_path", "env_path", "context_config_path"):
            if selection.config.get(field):
                value = Path(selection.config[field])
                selection.config[field] = str((value if value.is_absolute() else Path(root) / value).resolve())
        if not selection.tasks:
            raise ValueError("Campaign selection contains no tasks")
        selections.append(selection)
    if not selections:
        raise ValueError("Pass at least one --config")
    return selections


def validate(config_paths, dataset, *, root=ROOT):
    selections = load_selections(config_paths, dataset, root=root)
    return {"dataset": dataset, "tasks": sum(len(s.tasks) for s in selections),
            "splits": [s.config["split"] for s in selections], "validation": [s.validation for s in selections]}


def prepare(output, config_paths, dataset, *, root=ROOT, plugin_config=None):
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise ValueError("Campaign already prepared; run resumes its frozen selection")
    selections = load_selections(config_paths, dataset, root=root)
    info = environment_diagnostics()
    if not info["cuda_available"]:
        raise RuntimeError("CUDA unavailable in the required flyagent environment")
    for name, value in read_env_file(selections[0].config["env_path"]).items():
        if name.startswith("FLYAGENT_"):
            os.environ.setdefault(name, value)
    output.mkdir(parents=True, exist_ok=True)
    source = freeze_source(output, root=root, plugin_config=plugin_config)
    maps, policy, configs, tasks, validations = {}, None, {}, [], {}
    model = load_model_config(selections[0].config["env_path"]).redacted_summary()
    for selection in selections:
        config = dict(selection.config)
        if load_model_config(config["env_path"]).redacted_summary() != model:
            raise ValueError("A campaign must use the same decision model in every split")
        key = dataset + "/" + config["split"]
        if key in configs:
            raise ValueError("Duplicate dataset/split configuration")
        catalog = json.loads(Path(config["map_config_path"]).read_text(encoding="utf-8-sig"))
        current_policy = catalog.get("port_policy", {})
        if policy is not None and current_policy != policy:
            raise ValueError("Split map catalogs use different port policies")
        policy = current_policy
        for item in catalog["maps"]:
            if item["map_id"] in maps and maps[item["map_id"]] != item:
                raise ValueError("Conflicting map IDs in campaign catalogs")
            maps[item["map_id"]] = item
        config.update(output_dir=str(output / dataset / config["split"]), save_artifacts=True,
                      generate_html_review=False, report_profile="paper_process")
        for field in ("context_config_path", "observation_profile_path"):
            if config.get(field):
                original = Path(config[field]).resolve()
                if Path(root).resolve() in original.parents:
                    config[field] = str(source / original.relative_to(Path(root).resolve()))
                else:
                    copied = source / "configs/campaign_inputs" / (hashlib.sha256(original.read_bytes()).hexdigest() + original.suffix)
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    copied.write_bytes(original.read_bytes())
                    config[field] = str(copied)
        configs[key] = config
        validations[key] = selection.validation
        tasks.extend({**task, "config_key": key} for task in selection.tasks)
    catalog_path = source / "configs/maps/campaign_merged.json"
    write_json(catalog_path, {"port_policy": policy, "maps": list(maps.values())})
    for config in configs.values():
        config["map_config_path"] = str(catalog_path)
    keys = [ParallelRecoveryQueue.key(task) for task in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("Campaign task identities are duplicated")
    hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    manifest = {"version": 2, "dataset": dataset, "configs": configs, "primary_config_key": next(iter(configs)),
                "tasks": tasks, "stage_splits": list(dict.fromkeys(t["split"] for t in tasks)),
                "counts": dict(Counter(t["split"] for t in tasks)), "technical_trial": tasks[:4],
                "strict_workers": 2, "dispatch_concurrency": 2, "terminal_provider_errors": False,
                "model": model, "environment": info, "source_hashes": hashes, "completion_status": "campaign_complete",
                "policy": {"episode_attempts": 3, "episode_deadline_sec": 7200, "no_progress_sec": 900},
                "logging": {"profile": "paper_process_metrics_v1", "adapter": "compact_logging.py"},
                "extension_files": {"compact_logging.py": str(Path(root) / "evaluation/experiments/compact_logging.py")}}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    write_json(output / "manifest.json", manifest)
    write_json(output / "source_hashes.json", hashes)
    write_json(output / "dataset_validation.json", validations)
    queue = ParallelRecoveryQueue(output / "progress.json", tasks, manifest["fingerprint"])
    from evaluation.benchmarks.core.campaign.reporting import write_report
    write_report(output, manifest, queue.state)
    return {"output": str(output), "dataset": dataset, "tasks": len(tasks), "fingerprint": manifest["fingerprint"]}
