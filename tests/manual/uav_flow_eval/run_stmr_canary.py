"""Run configurable UAV-Flow online-STMR canaries.

This is evaluation scaffolding, not runtime behavior. Each selected STMR mode
writes the normal UAV-Flow batch and case reviews. The script deliberately does
not build a second comparison report; STMR rendering is inspected from the
native case ``review.html`` panel.

Complete Approach example (normal 100-window emergency guard):

    F:\\conda_env\\flyagent\\python.exe tests/manual/uav_flow_eval/run_stmr_canary.py ^
      --canary-id stmr_approach_full_1x --classes Approach --modes planner

This entry point requires real UnrealCV. Offline orchestration tests use
explicit test-only dependency injection. Local model directories are used by
default when present; explicit CLI paths always take precedence.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.runner import UavFlowBenchmarkRunner
from plugins.research.stmr_map.perception import StmrPerceptionSession
from plugins.research.stmr_map.perception.grounding_dino_backend import DEFAULT_GROUNDING_DINO_MODEL
from plugins.research.stmr_map.perception.sam_backend import DEFAULT_SAM_MODEL


DEFAULT_CONFIG = "evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json"
DEFAULT_CLASSES = ["Turn", "Move", "Shift", "Approach", "Pass"]
VALID_MODES = ("off", "shadow", "planner")
LOCAL_GROUNDING_MODEL = Path(r"F:\hf_models\grounding-dino-tiny")
LOCAL_SAM_MODEL = Path(r"F:\hf_models\sam-vit-base")


def main() -> int:
    args = _parser().parse_args()
    classes = _csv_values(args.classes)
    modes = _csv_values(args.modes, allowed=set(VALID_MODES))
    base = UavFlowEvalConfig.from_file(args.config)
    canary_id = args.canary_id or f"stmr_canary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    common = replace(
        base,
        execution_mode=args.execution_mode,
        enable_sr_judge=not args.no_sr_judge,
        enable_visual_diagnostics=not args.no_visual_diagnostics,
        enable_action_planner_debug=True,
        classes=classes,
        limit_per_class=1,
        max_cases=None,
        runtime_loop_guard_windows=args.runtime_loop_guard_windows,
        stmr_prior=args.stmr_prior,
        stmr_max_queries=args.stmr_max_queries,
        stmr_device=args.stmr_device,
    )
    session = None
    if any(mode != "off" for mode in modes):
        session = StmrPerceptionSession(
            grounding_model=args.grounding_model,
            sam_model=args.sam_model,
            device=common.stmr_device,
        )

    runs: dict[str, dict[str, Any]] = {}
    for mode in modes:
        result = _run(
            replace(common, stmr_mode=mode),
            f"{canary_id}_{mode}",
            None if mode == "off" else session,
        )
        runs[mode] = _summarize_run(result)

    print(
        json.dumps(
            {
                "canary_id": canary_id,
                "classes": classes,
                "modes": modes,
                "runtime_loop_guard_windows": args.runtime_loop_guard_windows,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configurable online-STMR UAV-Flow canaries.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--canary-id", default=None)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--modes", default="planner", help="Comma-separated: off,shadow,planner")
    parser.add_argument(
        "--execution-mode",
        choices=["attach", "managed"],
        default="attach",
    )
    parser.add_argument("--no-sr-judge", action="store_true")
    parser.add_argument("--no-visual-diagnostics", action="store_true")
    parser.add_argument("--stmr-prior", default="default_uav_prior")
    parser.add_argument("--stmr-max-queries", type=int, default=16)
    parser.add_argument("--stmr-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--grounding-model", default=_available_model(LOCAL_GROUNDING_MODEL, DEFAULT_GROUNDING_DINO_MODEL))
    parser.add_argument("--sam-model", default=_available_model(LOCAL_SAM_MODEL, DEFAULT_SAM_MODEL))
    parser.add_argument(
        "--runtime-loop-guard-windows",
        type=int,
        default=100,
        help="Emergency loop guard; normal completion and safety budgets may stop earlier.",
    )
    return parser


def _available_model(local_path: Path, fallback: str) -> str:
    return str(local_path) if local_path.is_dir() else fallback


def _csv_values(value: str, *, allowed: set[str] | None = None) -> list[str]:
    values = list(dict.fromkeys(item.strip() for item in str(value).split(",") if item.strip()))
    if not values:
        raise ValueError("At least one comma-separated value is required")
    invalid = [item for item in values if allowed is not None and item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported values: {', '.join(invalid)}")
    return values


def _run(config: UavFlowEvalConfig, eval_id: str, session: Any | None) -> dict[str, Any]:
    return UavFlowBenchmarkRunner(config, stmr_session=session).run(eval_id=eval_id)


def _summarize_run(result: dict[str, Any]) -> dict[str, Any]:
    """Return only native UAV-Flow review paths for terminal handoff."""

    eval_dir = Path(str(result.get("eval_dir") or ""))
    summary_paths = result.get("summary_paths") if isinstance(result.get("summary_paths"), dict) else {}
    case_reviews = []
    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue
        case_dir = Path(str(item.get("case_dir") or ""))
        case_reviews.append(
            {
                "case_id": item.get("case_id"),
                "class_name": item.get("class_name"),
                "status": item.get("status"),
                "completed": item.get("completed"),
                "review_html": str(case_dir / "review.html"),
            }
        )
    return {
        "eval_dir": str(eval_dir),
        "review_html": summary_paths.get("html_review") or str(eval_dir / "review.html"),
        "summary_json": summary_paths.get("summary_json") or str(eval_dir / "summary.json"),
        "case_reviews": case_reviews,
    }


if __name__ == "__main__":
    raise SystemExit(main())
