"""AerialVLN case selection, execution and metric metadata for shared campaigns."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from evaluation.benchmarks.core.campaign.contracts import PreparedCampaignSplit
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner


class AerialVlnCampaignAdapter:
    """Preserve the existing runner's navigation and scoring contracts."""

    dataset = "aerial_vln"
    metric_fields = ("success", "ndtw")

    def prepare(self, config, *, require_execution_ready=False):
        settings = AerialVlnEvalConfig.from_dict(config)
        config = settings.to_dict()
        runner = AerialVlnBenchmarkRunner(settings)
        prepared = runner.prepare_evaluation(require_execution_ready=require_execution_ready)
        config["expected_split_sha256"] = prepared.provenance.split_sha256
        tasks = []
        for case in prepared.cases:
            assignment = prepared.selection.assignment_by_episode[case.episode_id]
            tasks.append({"dataset": self.dataset, "split": config["split"], "episode_id": case.episode_id,
                          "scene_id": case.scene_id, "difficulty": assignment.difficulty,
                          "reference_path_length_m": assignment.reference_path_length_m,
                          "instruction_sha256": hashlib.sha256(case.instruction.encode("utf-8")).hexdigest(),
                          "truth_digest": _truth_digest(case.evaluation_truth())})
        return PreparedCampaignSplit(config, tuple(tasks), prepared.validation_dict())

    def execute(self, config, task, directory):
        settings = replace(AerialVlnEvalConfig.from_dict(config), episode_ids=(task["episode_id"],),
                           output_dir=str(directory / "evaluation"), run_root=str(directory / "runs"), generate_html_review=False)
        runner = AerialVlnBenchmarkRunner(settings)
        prepared = runner.prepare_evaluation(require_execution_ready=True)
        case = _verified_case(prepared, task)
        return runner.run_case(case, directory / "evaluation", provenance=prepared.provenance,
                               assignment=prepared.selection.assignment_by_episode[case.episode_id])


def _truth_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def create_adapter():
    return AerialVlnCampaignAdapter()


def _verified_case(prepared, task):
    if len(prepared.cases) != 1 or prepared.cases[0].episode_id != task["episode_id"]:
        raise ValueError("Frozen task identity changed")
    case = prepared.cases[0]
    if task.get("instruction_sha256") and hashlib.sha256(case.instruction.encode("utf-8")).hexdigest() != task["instruction_sha256"]:
        raise ValueError("Frozen task instruction changed")
    if task.get("truth_digest") and _truth_digest(case.evaluation_truth()) != task["truth_digest"]:
        raise ValueError("Frozen dataset truth changed")
    return case
