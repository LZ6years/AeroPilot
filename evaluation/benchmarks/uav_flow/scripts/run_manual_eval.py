"""Run real UAV-Flow evaluation using explicitly configured attach/managed scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.runner import UavFlowBenchmarkRunner


DEFAULT_CONFIG = "evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json"


def main() -> int:
    args = _parser().parse_args()
    config = _config_from_args(args)
    summary = _config_summary(config, args.eval_id)
    print(json.dumps({"uav_flow_manual_eval_config": summary}, indent=2, ensure_ascii=False))
    preflight = _unrealcv_preflight(config)
    print(json.dumps({"unrealcv_preflight": preflight}, indent=2, ensure_ascii=False))
    if preflight["status"] != "ok":
        return 2
    if args.preflight_only:
        return 0
    print(
        "Reminder: this run will launch real UnrealCV/UE. "
        "Check for old python/Collection/UE4/Unreal processes before running long batches."
    )
    result = UavFlowBenchmarkRunner(config).run(eval_id=args.eval_id)
    print(json.dumps(_result_summary(result), indent=2, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run UAV-Flow manual benchmark batches with debug-friendly defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Base UAV-Flow eval config JSON.")
    parser.add_argument("--eval-id", default=None, help="Output eval id. Omit to use a timestamped eval directory.")
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument(
        "--case-ids",
        default=None,
        help=(
            "Comma-separated dataset case stems. Exact selection overrides "
            "--classes, --limit-per-class, and --max-cases."
        ),
    )
    selectors.add_argument(
        "--all-cases",
        action="store_true",
        help=(
            "Select every dataset case by clearing case_ids, classes, "
            "limit_per_class, and max_cases from the base config."
        ),
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional comma-separated class override; omitted preserves the config.",
    )
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional number of cases per class.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional total case cap after class filtering.")
    parser.add_argument(
        "--execution-mode",
        choices=["attach", "managed"],
        default=None,
        help="Real simulator ownership mode.",
    )
    parser.add_argument(
        "--sr-judge",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="enable_sr_judge",
        help="Override the config's optional post-episode SR judge.",
    )
    parser.add_argument(
        "--visual-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="enable_visual_diagnostics",
        help="Override the config's optional frame-level visual diagnostics.",
    )
    parser.add_argument(
        "--action-planner-debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="enable_action_planner_debug",
        help="Override action-planner debug artifact generation.",
    )
    parser.add_argument(
        "--runtime-loop-guard-windows",
        type=int,
        default=None,
        help="Optional per-semantic-phase runtime window limit.",
    )
    parser.add_argument("--env-path", default=None, help="Optional .env path for model config.")
    parser.add_argument(
        "--planner-depth-input",
        choices=["off", "now"],
        default=None,
        help="Optional pixel-aligned NOW RGB + metric-depth overlay for the planner.",
    )
    parser.add_argument("--stmr-mode", choices=["off", "shadow", "planner"], default=None, help="Optional online STMR experiment mode.")
    parser.add_argument("--stmr-prior", default=None, help="Named STMR detector-query prior.")
    parser.add_argument("--stmr-max-queries", type=int, default=None, help="Maximum task-plus-prior detector queries.")
    parser.add_argument("--stmr-device", choices=["auto", "cuda", "cpu"], default=None, help="Device for GroundingDINO and SAM.")
    parser.add_argument("--preflight-only", action="store_true", help="Check real UnrealCV imports and exit before launching UE.")
    return parser


def _config_from_args(args: argparse.Namespace) -> UavFlowEvalConfig:
    base = UavFlowEvalConfig.from_file(args.config)
    data = base.to_dict()
    case_ids = _parse_case_ids(args.case_ids)

    # Selection flags are intentionally explicit. In particular, omitted CLI
    # options must not turn a carefully reviewed full-run config back into the
    # script's historical Turn/Approach one-case smoke selection.
    if args.all_cases:
        data.update(
            {
                "case_ids": None,
                "classes": None,
                "limit_per_class": None,
                "max_cases": None,
            }
        )
    elif case_ids is not None:
        data.update(
            {
                "case_ids": case_ids,
                "classes": None,
                "limit_per_class": None,
                "max_cases": None,
            }
        )
    else:
        if args.classes is not None:
            data["case_ids"] = None
            data["classes"] = _parse_classes(args.classes)
        if args.limit_per_class is not None:
            data["limit_per_class"] = args.limit_per_class
        if args.max_cases is not None:
            data["max_cases"] = args.max_cases

    optional_overrides = {
        "execution_mode": args.execution_mode,
        "enable_sr_judge": args.enable_sr_judge,
        "enable_visual_diagnostics": args.enable_visual_diagnostics,
        "enable_action_planner_debug": args.enable_action_planner_debug,
        "runtime_loop_guard_windows": args.runtime_loop_guard_windows,
    }
    for field_name, value in optional_overrides.items():
        if value is not None:
            data[field_name] = value
    if args.env_path:
        data["env_path"] = args.env_path
    if args.planner_depth_input:
        data["planner_depth_input"] = args.planner_depth_input
    if args.stmr_mode:
        data["stmr_mode"] = args.stmr_mode
    if args.stmr_prior:
        data["stmr_prior"] = args.stmr_prior
    if args.stmr_max_queries is not None:
        data["stmr_max_queries"] = args.stmr_max_queries
    if args.stmr_device:
        data["stmr_device"] = args.stmr_device
    if data.get("generate_html_review", True):
        data["save_artifacts"] = True
        data["enable_action_planner_debug"] = True
    return UavFlowEvalConfig.from_dict(data)


def _parse_classes(value: str) -> list[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise ValueError("--classes must contain at least one class name; use --all-cases for the full dataset")
    return classes


def _parse_case_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _config_summary(config: UavFlowEvalConfig, eval_id: str | None) -> dict[str, Any]:
    return {
        "eval_id": eval_id,
        "uav_flow_repo_path": config.uav_flow_repo_path,
        "output_dir": config.output_dir,
        "case_ids": list(config.case_ids) if config.case_ids else None,
        "classes": config.classes,
        "limit_per_class": config.limit_per_class,
        "max_cases": config.max_cases,
        "map_config_path": config.map_config_path,
        "map_id": config.map_id,
        "execution_mode": config.execution_mode,
        "observation_profile_path": config.observation_profile_path,
        "enable_sr_judge": config.enable_sr_judge,
        "enable_visual_diagnostics": config.enable_visual_diagnostics,
        "enable_action_planner_debug": config.enable_action_planner_debug,
        "generate_html_review": config.generate_html_review,
        "save_artifacts": config.save_artifacts,
        "runtime_loop_guard_windows": config.runtime_loop_guard_windows,
        "env_path": config.env_path,
        "planner_depth_input": config.planner_depth_input,
        "stmr_mode": config.stmr_mode,
        "stmr_prior": config.stmr_prior,
        "stmr_max_queries": config.stmr_max_queries,
        "stmr_device": config.stmr_device,
    }


def _unrealcv_preflight(config: UavFlowEvalConfig) -> dict[str, Any]:
    map_config = Path(config.map_config_path)
    result: dict[str, Any] = {
        "status": "ok",
        "python_executable": sys.executable,
        "map_config_path": str(map_config),
        "map_config_exists": map_config.exists(),
        "execution_mode": config.execution_mode,
    }
    if not map_config.exists():
        result["status"] = "failed"
        result["reason"] = "map_config_path_missing"
        return result
    try:
        from unrealcv import Client  # noqa: F401
    except Exception as exc:
        result["status"] = "failed"
        result["reason"] = "unrealcv_import_failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    eval_dir = Path(str(result.get("eval_dir", "")))
    summary_paths = dict(result.get("summary_paths", {}))
    return {
        "eval_dir": str(eval_dir),
        "review_html": summary_paths.get("html_review") or str(eval_dir / "review.html"),
        "summary_json": summary_paths.get("summary_json") or str(eval_dir / "summary.json"),
        "summary_md": summary_paths.get("summary_md") or str(eval_dir / "summary.md"),
        "case_count": len(result.get("results", [])) if isinstance(result.get("results"), list) else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
