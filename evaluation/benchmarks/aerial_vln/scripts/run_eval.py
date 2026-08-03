"""CLI entry point for AerialVLN native-action evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import (
    AerialVlnBenchmarkRunner,
)


def main() -> int:
    args = _parser().parse_args()
    base = AerialVlnEvalConfig.from_file(args.config)
    data = base.to_dict()
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
    result = AerialVlnBenchmarkRunner(
        AerialVlnEvalConfig.from_dict(data)
    ).run(eval_id=args.eval_id)
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
    parser.add_argument(
        "--execution-mode",
        choices=["dry", "attach", "managed"],
        default=None,
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--scene", action="append", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
