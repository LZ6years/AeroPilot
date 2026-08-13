"""CLI entry point for AerialVLN native-action evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import (
    AerialVlnBenchmarkRunner,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _config_from_args(args)
    runner = AerialVlnBenchmarkRunner(config)
    if args.validate_only:
        validation = runner.validate(eval_id=args.eval_id)
        print(
            json.dumps(
                validation,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    result = runner.run(eval_id=args.eval_id)
    print(
        json.dumps(
            {
                "eval_dir": result["eval_dir"],
                "summary_paths": result["summary_paths"],
                "case_count": len(result["results"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _config_from_args(args: argparse.Namespace) -> AerialVlnEvalConfig:
    base = AerialVlnEvalConfig.from_file(args.config)
    data = base.to_dict()
    if args.dataset_root:
        data["dataset_root"] = args.dataset_root
    if args.split:
        data["split"] = args.split
    if args.dataset_family:
        data["dataset_family"] = args.dataset_family
    if args.expected_split_sha256:
        data["expected_split_sha256"] = args.expected_split_sha256
    if args.difficulties:
        data["difficulties"] = _parse_difficulties(args.difficulties)
    if args.all_per_difficulty:
        data["limit_per_difficulty"] = None
    elif args.limit_per_difficulty is not None:
        data["limit_per_difficulty"] = args.limit_per_difficulty
    if args.execution_mode:
        data["execution_mode"] = args.execution_mode
    if args.max_cases is not None:
        data["max_cases"] = args.max_cases
    if args.scene:
        data["scene_filter"] = list(args.scene)
    if (
        data["execution_mode"] != "dry"
        and data.get("generate_html_review", True)
    ):
        data["save_artifacts"] = True
        data["enable_action_planner_debug"] = True
    return AerialVlnEvalConfig.from_dict(data)


def _parse_difficulties(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AerialVLN with UAV-Agent native continuous actions"
    )
    parser.add_argument(
        "--config",
        default=(
            "evaluation/benchmarks/aerial_vln/configs/"
            "aerial_vln_eval.example.json"
        ),
    )
    parser.add_argument("--eval-id", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Dataset split. AerialVLN-S supports val_seen and val_unseen; "
            "one invocation evaluates exactly one split."
        ),
    )
    parser.add_argument(
        "--dataset-family",
        choices=["aerialvln", "aerialvln-s"],
        default=None,
    )
    parser.add_argument(
        "--expected-split-sha256",
        default=None,
        help="Expected SHA256 for the split selected by --split/config.",
    )
    parser.add_argument(
        "--difficulties",
        default=None,
        help="Comma-separated low,medium,high difficulty buckets.",
    )
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument("--limit-per-difficulty", type=int)
    limit_group.add_argument(
        "--all-per-difficulty",
        action="store_true",
        help="Run every episode in each requested difficulty bucket.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["dry", "attach", "managed"],
        default=None,
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate and classify the local split without model, SSH, "
            "simulator, or Runtime execution."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
