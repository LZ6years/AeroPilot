"""Generate UAV-Flow HTML reports from an existing evaluation directory.

This command is safe to use after an interrupted batch: it performs no model,
Runtime, simulator, process, or network operations.  It only validates
finalized ``case.json``/``audit.json`` pairs and rewrites report artifacts in
the selected evaluation directory.

Example
-------

    python evaluation/benchmarks/uav_flow/scripts/render_existing_eval.py \
      --eval-dir outputs/uav_flow_partial \
      --run-status interrupted \
      --expected-case-count 273 \
      --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.postprocess import (
    recover_existing_evaluation,
)


def main() -> int:
    """Parse CLI input, rebuild reports, and print a compact UTF-8 summary."""

    args = _parser().parse_args()
    result = recover_existing_evaluation(
        args.eval_dir,
        run_status=args.run_status,
        expected_case_count=args.expected_case_count,
        strict=args.strict,
    )
    print(
        json.dumps(
            {
                "eval_dir": str(Path(args.eval_dir).resolve()),
                "run_status": result["run_status"],
                "expected_case_count": result["expected_case_count"],
                "finalized_case_count": result["finalized_case_count"],
                "incomplete_case_ids": result["incomplete_case_ids"],
                "excluded_cases": result["excluded_cases"],
                "warnings": result["warnings"],
                "summary_paths": result["summary_paths"],
                "report_manifest": result["report_manifest"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild UAV-Flow JSON, Markdown, and HTML reports without "
            "running UE or model services."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Existing evaluation directory containing cases/<case-id>/.",
    )
    parser.add_argument(
        "--run-status",
        choices=["running", "completed", "interrupted", "failed"],
        default="interrupted",
        help="Lifecycle state shown prominently in the recovered report.",
    )
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=None,
        help=(
            "Original selected case count. Omit only when finalized cases "
            "represent the entire intended batch."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail before writing when a finalized case is invalid or a "
            "declared frame asset is missing. Incomplete directories without "
            "case.json remain allowed."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
