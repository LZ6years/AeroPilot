"""Run an ordered explicit subset of a complete AerialVLN split.

Manual validation harness only: no production sampling or metrics change. Unlike
limit-per-difficulty=1, explicit IDs preserve case identity.
The manifest explicitly records this post-selection filter. Use --validate-only
to inspect the selected ID without Provider, SSH or simulator execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
from evaluation.benchmarks.aerial_vln.selection import AerialVlnCaseSelection


@dataclass(frozen=True)
class ExplicitProbeSelection(AerialVlnCaseSelection):
    manual_cases: tuple[dict, ...] = field(default_factory=tuple)
    dataset_sha256: str = ""

    def to_dict(self):
        result = super().to_dict()
        result["parent_sampling_method"] = result["method"]
        result["method"] = "explicit episode ID subset of complete split (manual simple-instruction probe)"
        result["manual_cases"] = list(self.manual_cases)
        result["dataset_sha256"] = self.dataset_sha256
        return result


class SelectedCaseProbe(AerialVlnBenchmarkRunner):
    """Reuse normal lifecycle and artifact generation, replacing only selection."""
    def __init__(self, config, episode_id=None, *, episode_ids=None, strict_case_gates=False):
        self.strict_case_gates = strict_case_gates
        if (episode_id is None) == (episode_ids is None):
            raise ValueError("Provide exactly one of episode_id or episode_ids")
        self.episode_ids = validate_episode_ids([episode_id] if episode_id is not None else episode_ids)
        # Classify the entire split before applying the explicit execution subset.
        super().__init__(replace(config, difficulties=("low", "medium", "high"),
                                 limit_per_difficulty=None,
                                 abort_on_infrastructure_error=True if strict_case_gates else config.abort_on_infrastructure_error))

    def run_case(self, case, eval_dir, **kwargs):
        result = super().run_case(case, eval_dir, **kwargs)
        if self.strict_case_gates:
            from tests.manual.aerial_vln_case_gate import audit_case
            from evaluation.benchmarks.core.review import BenchmarkReviewRenderer
            package = self.map_packages_by_id[self.config.scene_map[case.scene_id]]
            try:
                # Production builds reviews here but writes HTML at batch end.
                # Materialize this finalized case with the shared renderer
                # before checking images; never weaken the HTML gate.
                BenchmarkReviewRenderer().write_case(eval_dir, self._case_reviews[case.episode_id])
                gate = audit_case(Path(eval_dir) / "cases" / case.episode_id, result, package)
            except Exception as exc:
                gate = {"passed": False, "error": f"case review gate: {type(exc).__name__}: {exc}"}
                (Path(eval_dir) / "cases" / case.episode_id / "manual_gate.json").write_text(
                    json.dumps(gate, indent=2), encoding="utf-8")
            result["manual_gate"] = gate
            if not gate["passed"]:
                result["infrastructure_error"] = result.get("infrastructure_error") or {
                    "stage": "manual_audit_gate", "error": "Per-case protocol/trace/tracking/cleanup gate failed"}
            path = Path(eval_dir) / "cases" / case.episode_id / "case_result.json"
            durable = json.loads(path.read_text(encoding="utf-8"))
            durable["manual_gate_passed"] = gate["passed"]
            durable["infrastructure_error"] = result.get("infrastructure_error")
            path.write_text(json.dumps(durable, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def prepare_evaluation(self, *, require_execution_ready):
        prepared = super().prepare_evaluation(require_execution_ready=False)
        by_id = {c.episode_id: c for c in prepared.cases}
        missing = [i for i in self.episode_ids if i not in by_id]
        if missing:
            raise ValueError(f"Unknown episode IDs: {missing}")
        selected = tuple(by_id[i] for i in self.episode_ids)
        values = vars(prepared.selection).copy()
        assignments = prepared.selection.assignment_by_episode
        values["selected_assignments"] = tuple(assignments[i] for i in self.episode_ids)
        selection = ExplicitProbeSelection(
            **values, dataset_sha256=prepared.provenance.split_sha256,
            manual_cases=tuple({**assignments[c.episode_id].to_dict(),
                                "instruction": c.instruction} for c in selected))
        result = replace(prepared, cases=selected, selection=selection,
                         execution_readiness=self._execution_readiness(selection.selected_scene_ids))
        if require_execution_ready:
            self._validate_dataset_for_execution(result)
            if not result.execution_readiness["ready"]:
                raise ValueError(f"Selected scenes not ready: {result.execution_readiness}")
        return result


def validate_episode_ids(value):
    """Reject invalid manifests rather than silently dropping IDs or reordering."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Episode IDs must be a non-empty list")
    if any(not isinstance(i, str) or not i.strip() or i != i.strip() for i in value):
        raise ValueError("Episode IDs must be non-empty strings without surrounding whitespace")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate episode IDs")
    return tuple(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--episode-id")
    ids.add_argument("--episode-ids-file", type=Path)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--strict-case-gates", action="store_true",
                        help="Stop the existing batch loop on failed per-case audit/cleanup evidence.")
    args = parser.parse_args()
    config = AerialVlnEvalConfig.from_file(args.config)
    runner = (SelectedCaseProbe(config, episode_ids=json.loads(
        args.episode_ids_file.read_text(encoding="utf-8")), strict_case_gates=args.strict_case_gates) if args.episode_ids_file
        else SelectedCaseProbe(config, args.episode_id, strict_case_gates=args.strict_case_gates))
    if args.validate_only:
        prepared = runner.prepare_evaluation(require_execution_ready=True)
        print(json.dumps({"episode_ids": [c.episode_id for c in prepared.cases],
                          "selection": prepared.selection.to_dict(), "readiness": prepared.execution_readiness}, indent=2))
    else:
        result = runner.run(eval_id=args.eval_id)
        print(json.dumps({"eval_dir": result["eval_dir"], "case_count": len(result["results"])}))
