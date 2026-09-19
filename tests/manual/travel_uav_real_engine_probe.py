"""Run one real TravelUAV case through the production FlyAgent runner.

This is not a fixed-action smoke test.  It selects one compact-index case,
spawns its target, runs the configured TaskParser/ModelActionPlanner, and emits
the same case/evaluation HTML as a normal benchmark run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--eval-id", default="travel_uav_real_engine_probe")
    args = parser.parse_args()
    data = TravelUavEvalConfig.from_file(args.config).to_dict()
    data.update(
        {
            "execution_mode": "managed",
            "case_ids": [args.case_id],
            "save_artifacts": True,
            "generate_html_review": True,
            "enable_action_planner_debug": True,
        }
    )
    result = TravelUavBenchmarkRunner(TravelUavEvalConfig.from_dict(data)).run(
        eval_id=args.eval_id
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["summary"]["batch_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
