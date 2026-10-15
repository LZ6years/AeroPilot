"""Read legacy campaign conventions without imposing them on new campaigns.

Older Aerial manifests selected one seen split and older paired experiments
declared image budgets directly. These human/evaluation compatibility rules do
not rewrite manifests, task receipts or any model-facing navigation state.
"""
from __future__ import annotations

import importlib
from pathlib import Path


def dataset_name(manifest, task=None):
    value = (task or {}).get("dataset") or manifest.get("dataset")
    if value:
        return value
    if manifest.get("version", 1) == 1:
        return "aerial_vln"
    raise ValueError("New campaigns require an explicit dataset")


def adapter_for(manifest, task=None):
    dataset = dataset_name(manifest, task)
    module = manifest.get("adapter_modules", {}).get(dataset, f"evaluation.benchmarks.{dataset}.campaign_adapter")
    if not module.startswith("evaluation.benchmarks."):
        raise ValueError("Campaign adapters must be evaluation modules")
    return importlib.import_module(module).create_adapter()


def selected_tasks(manifest, *, tasks=None):
    splits = manifest.get("stage_splits")
    if splits is None and manifest.get("version", 1) == 1 and manifest.get("experiment") != "image_budget" and not manifest.get("dataset"):
        splits = ["val_seen"]
    return [task for task in (manifest.get("tasks") if tasks is None else tasks) if splits is None or task["split"] in splits]


def completion_status(manifest):
    if manifest.get("completion_status"):
        return manifest["completion_status"]
    if manifest.get("version", 1) == 1 and not manifest.get("dataset"):
        return "awaiting_image_budget_choice" if manifest.get("experiment") == "image_budget" else "seen_complete"
    return "campaign_complete"


def group_policy(manifest):
    policy = manifest.get("scheduling", {})
    field, order = policy.get("group_field"), policy.get("group_order")
    if field is None and manifest.get("experiment") == "image_budget":
        field, order = "image_budget", manifest["image_budgets"]
    return field, order


def report_readers(manifest):
    if "report_readers" in manifest:
        return manifest["report_readers"]
    if manifest.get("experiment") == "image_budget":
        return {"compression_cost.py": "evaluation.experiments.compression_cost",
                "image_budget_report.py": "evaluation.experiments.image_budget_report"}
    return {}


def extra_files(manifest, repo):
    names = dict(manifest.get("extension_files", {}))
    if manifest.get("experiment") == "image_budget":
        names.update({name: str(Path(repo) / "evaluation/experiments" / name)
                      for name in ("compression_cost.py", "image_budget_report.py", "compact_logging.py")})
    if manifest.get("request_audit_gate"):
        names["request_audit.py"] = str(Path(repo) / "tests/manual/aerial_vln_plan_context_acceptance.py")
    return names


def logging_settings(manifest):
    if manifest.get("logging"):
        return manifest["logging"]
    if manifest.get("experiment") == "image_budget":
        return {"profile": "paper_process_metrics_v1", "adapter": "compact_logging.py"}
    return {"profile": "full"}


def report_profile(manifest, config):
    if manifest.get("experiment") == "image_budget":
        return "paper_process"
    return config.get("report_profile", "paper_process")


def render_report(output, manifest, queue):
    module = manifest.get("reporter")
    if module is None and manifest.get("experiment") == "image_budget":
        module = "evaluation.experiments.image_budget_report"
    if module:
        return importlib.import_module(module).write_report(output, queue.state)
    from evaluation.benchmarks.core.campaign.reporting import write_report
    return write_report(output, manifest, queue.state)


def attach_review(path, result, manifest):
    if manifest.get("experiment") in {"pixel_place_memory", "image_budget"}:
        from evaluation.benchmarks.aerial_vln.pixel_map_campaign import attach_place_report
        attach_place_report(path, result)


def check_requests(run_path, task, manifest):
    if manifest.get("experiment") == "image_budget":
        from evaluation.experiments.image_budget import validate_captured_requests
        return validate_captured_requests(run_path, task["image_budget"])
    return []


def write_comparison(output, queue, manifest, *, pilot=False):
    if manifest.get("experiment") == "pixel_place_memory":
        from evaluation.benchmarks.aerial_vln.pixel_map_campaign import write_comparison
        return write_comparison(output, queue.state, pilot=pilot)


def requires_delivered_pilot(manifest):
    return manifest.get("experiment") == "pixel_place_memory"
