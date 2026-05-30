"""Run a small UAV-Flow benchmark smoke test through UAV-Agent runtime."""

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
    config = _config_from_args(args)
    result = UavFlowBenchmarkRunner(config).run()
    print(json.dumps({"eval_dir": result["eval_dir"], "summary_paths": result["summary_paths"]}, indent=2, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UAV-Flow benchmark smoke cases")
    parser.add_argument("--config", default="evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json")
    parser.add_argument("--classes", default=None, help="Comma-separated classes, for example Turn,Move,Approach")
    parser.add_argument("--limit-per-class", type=int, default=1)
    parser.add_argument("--real-unrealcv", action="store_true", help="Use real gym_unrealcv instead of dry-run scenario")
    parser.add_argument("--enable-sr-judge", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> UavFlowEvalConfig:
    base = UavFlowEvalConfig.from_file(args.config)
    data = base.to_dict()
    if args.classes:
        data["classes"] = [item.strip() for item in args.classes.split(",") if item.strip()]
    data["limit_per_class"] = args.limit_per_class
    data["real_unrealcv"] = args.real_unrealcv or base.real_unrealcv
    data["enable_sr_judge"] = args.enable_sr_judge or base.enable_sr_judge
    return UavFlowEvalConfig.from_dict(data)


if __name__ == "__main__":
    raise SystemExit(main())
