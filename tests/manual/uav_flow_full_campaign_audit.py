"""Build a read-only, cross-segment audit for a UAV-Flow full campaign.

The command may be run while a segment is active to obtain an ordered list of
missing cases, or with ``--strict`` after all continuations have finished.  It
does not contact the model or UnrealCV and does not alter any source segment;
only the explicitly selected output directory is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.campaign import (  # noqa: E402
    audit_campaign,
    write_campaign_outputs,
)
from tests.manual.native_tool_loop_real_audit import (  # noqa: E402
    audit_evaluation,
)


def main() -> int:
    args = _parser().parse_args()
    report = audit_campaign(
        args.config,
        args.eval_dir,
        expected_loop_guard_windows=args.expected_loop_guard_windows,
        segment_auditor=audit_evaluation,
        replace_infrastructure_errors=args.replace_infrastructure_errors,
    )
    paths = write_campaign_outputs(report, args.output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "coverage_complete": report["coverage_complete"],
                "protocol_valid": report["protocol_valid"],
                "infrastructure_clean": report["infrastructure_clean"],
                "finalized_case_count": report["finalized_case_count"],
                "expected_case_count": report["expected_case_count"],
                "missing_case_count": len(report["missing_case_ids"]),
                "replacement_count": len(report["case_replacements"]),
                "violation_count": len(report["violations"]),
                "outputs": paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.strict and report["status"] != "ok":
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit main and continuation UAV-Flow evaluation directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Full UAV-Flow config used to locate the expected dataset.")
    parser.add_argument("--eval-dir", action="append", required=True, help="Evaluation segment directory; repeat in execution order.")
    parser.add_argument("--expected-loop-guard-windows", type=int, required=True, help="Effective per-phase window limit used by every segment.")
    parser.add_argument("--output-dir", required=True, help="Directory for campaign reports and remaining-case selectors.")
    parser.add_argument(
        "--replace-infrastructure-errors",
        action="store_true",
        help=(
            "Allow a later compatible, clean retry to replace an earlier "
            "finalized infrastructure error. Duplicate successful benchmark "
            "outcomes remain invalid."
        ),
    )
    parser.add_argument("--strict", action="store_true", help="Exit 2 unless coverage, protocol, and infrastructure checks all pass.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
