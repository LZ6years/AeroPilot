"""Run a class-filtered UAV-Flow benchmark batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.runner import UavFlowBenchmarkRunner


def main() -> int:
    args = _parser().parse_args()
    base = UavFlowEvalConfig.from_file(args.config)
    data = base.to_dict()
    if args.classes:
        data["classes"] = [item.strip() for item in args.classes.split(",") if item.strip()]
    if args.limit_per_class is not None:
        data["limit_per_class"] = args.limit_per_class
    if args.max_cases is not None:
        data["max_cases"] = args.max_cases
    data["real_unrealcv"] = args.real_unrealcv or base.real_unrealcv
    data["enable_sr_judge"] = args.enable_sr_judge or base.enable_sr_judge
    result = UavFlowBenchmarkRunner(UavFlowEvalConfig.from_dict(data)).run()
    print(json.dumps({"eval_dir": result["eval_dir"], "summary_paths": result["summary_paths"]}, indent=2, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UAV-Flow benchmark batch")
    parser.add_argument("--config", default="evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--real-unrealcv", action="store_true")
    parser.add_argument("--enable-sr-judge", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
