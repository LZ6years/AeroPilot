"""Run one UAV-Flow environment smoke case without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.env_smoke import run_env_smoke


def main() -> int:
    args = _parser().parse_args()
    config = UavFlowEvalConfig.from_file(args.config)
    result = run_env_smoke(
        config,
        class_name=args.class_name,
        output_root=args.output_root,
        real_unrealcv=args.real_unrealcv or config.real_unrealcv,
    )
    print(json.dumps({"case_dir": result["case_dir"]}, indent=2, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UAV-Flow env smoke without model calls")
    parser.add_argument("--config", default="evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json")
    parser.add_argument("--class", dest="class_name", default=None)
    parser.add_argument("--output-root", default="tests/manual_outputs/uav_flow_env_smoke")
    parser.add_argument("--real-unrealcv", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
