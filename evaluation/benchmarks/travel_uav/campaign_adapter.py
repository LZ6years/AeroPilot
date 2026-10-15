"""TravelUAV selection and execution adapter for the shared parallel engine."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from evaluation.benchmarks.core.campaign.contracts import PreparedCampaignSplit
from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner


class TravelUavCampaignAdapter:
    """Keep target placement, original instruction and 3D metrics runner-owned."""

    dataset = "travel_uav"
    metric_fields = ("success", "navigation_error")

    def prepare(self, config, *, require_execution_ready=False):
        settings = TravelUavEvalConfig.from_dict(config)
        config = settings.to_dict()
        prepared = TravelUavBenchmarkRunner(settings).prepare_evaluation(
            require_execution_ready=require_execution_ready)
        config["expected_index_sha256"] = prepared.provenance.index_sha256
        tasks = tuple({"dataset": self.dataset, "split": case.split, "episode_id": case.episode_id,
                       "scene_id": case.map_name, "difficulty": case.difficulty,
                       "reference_path_length_m": case.reference_path_length_m,
                       "instruction_sha256": hashlib.sha256(case.instruction.encode("utf-8")).hexdigest(),
                       "truth_digest": _truth_digest(case.evaluation_truth())} for case in prepared.cases)
        return PreparedCampaignSplit(config, tasks, prepared.validation_dict())

    def execute(self, config, task, directory):
        settings = replace(TravelUavEvalConfig.from_dict(config), case_ids=(task["episode_id"],),
                           output_dir=str(directory / "evaluation"), run_root=str(directory / "runs"), generate_html_review=False)
        runner = TravelUavBenchmarkRunner(settings)
        prepared = runner.prepare_evaluation(require_execution_ready=True)
        if len(prepared.cases) != 1 or prepared.cases[0].episode_id != task["episode_id"]:
            raise ValueError("Frozen task identity changed")
        case = prepared.cases[0]
        if task.get("instruction_sha256") and hashlib.sha256(case.instruction.encode("utf-8")).hexdigest() != task["instruction_sha256"]:
            raise ValueError("Frozen task instruction changed")
        if task.get("truth_digest") and _truth_digest(case.evaluation_truth()) != task["truth_digest"]:
            raise ValueError("Frozen dataset truth changed")
        return runner.run_case(case, directory / "evaluation", provenance=prepared.provenance)


def _truth_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def create_adapter():
    return TravelUavCampaignAdapter()
