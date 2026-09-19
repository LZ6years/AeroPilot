"""Run or validate one TravelUAV FlyAgent transfer-evaluation config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-id")
    parser.add_argument(
        "--split",
        choices=("test_seen", "test_unseen_map", "test_unseen_object"),
    )
    parser.add_argument("--difficulties", help="Comma-separated easy,hard")
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--limit-per-difficulty", type=int)
    limits.add_argument("--all-per-difficulty", action="store_true")
    parser.add_argument("--case-ids", help="Comma-separated compact-index IDs")
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    data = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("TravelUAV config root must be an object")
    _apply_cli_overrides(data, args)
    config = TravelUavEvalConfig.from_dict(data)
    runner = TravelUavBenchmarkRunner(config)
    result = (
        runner.validate(eval_id=args.eval_id)
        if args.validate_only
        else runner.run(eval_id=args.eval_id)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def _apply_cli_overrides(data: dict[str, Any], args: Any) -> None:
    if args.split:
        data["split"] = args.split
    if args.difficulties:
        data["difficulties"] = _csv(args.difficulties)
    if args.limit_per_difficulty is not None:
        data["limit_per_difficulty"] = args.limit_per_difficulty
    if args.all_per_difficulty:
        data["limit_per_difficulty"] = None
    if args.case_ids:
        data["case_ids"] = _csv(args.case_ids)
    if args.expected_index_sha256:
        data["expected_index_sha256"] = args.expected_index_sha256


def _csv(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Comma-separated option cannot be empty")
    return parsed


if __name__ == "__main__":
    main()
