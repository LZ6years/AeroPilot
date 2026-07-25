"""Run UAV-Flow benchmark cases from one small manual command.

Why this script exists
----------------------
UAV-Flow real evaluation usually needs many runtime switches: real UnrealCV,
UE process reuse, SR judge, visual diagnostics, action-planner debug context,
parser mode, and loop-guard settings. This script packages the debug-friendly
defaults in one place so you can launch a run without pasting an inline Python
program into PowerShell.

Default behavior
----------------
By default this script runs a small real-UnrealCV debug batch:

- classes: Turn,Approach
- limit_per_class: 1
- real_unrealcv: True
- reuse_real_unrealcv_env: True
- enable_sr_judge: True
- enable_visual_diagnostics: True
- enable_action_planner_debug: True
- runtime_loop_guard_windows: 40
- parser_mode: model

Common commands
---------------
Run the default 1x real debug smoke:

    python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py --eval-id uav_flow_manual_real_1x

Run a 3x Turn/Approach real batch:

    python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py --classes Turn,Approach --limit-per-class 3 --eval-id uav_flow_manual_real_3x

Run a dry benchmark smoke without launching UE:

    python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py --dry-run --classes Turn --limit-per-class 1 --eval-id uav_flow_manual_dry

Run a fast dry script check without model/SR/visual diagnostic calls:

    python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py --dry-run --classes Turn --limit-per-class 1 --eval-id uav_flow_manual_dry_fast --parser-mode rule_based --no-sr-judge --no-visual-diagnostics --no-action-planner-debug

Run one Move-only debug batch:

    python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py --classes Move --limit-per-class 1 --eval-id uav_flow_manual_move_1x

Check real UnrealCV imports without launching UE:

    python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py --preflight-only

Before real UE runs
-------------------
Check whether old UnrealCV/UE processes are still alive:

    Get-Process | Where-Object {
      $_.ProcessName -match "python|Collection|UE4|Unreal"
    } | Select-Object Id, ProcessName, Path

This script does not kill processes automatically. That is intentional: a
manual benchmark launcher should not risk stopping unrelated Python or Unreal
work.

Outputs to inspect
------------------
After a run, this script prints the eval directory and the key report paths.
Open:

    evaluation/benchmarks/uav_flow/outputs/<eval_id>/review.html

Then inspect per-case review pages, RGB frames, coupled planner debug cards,
latest planner context, runtime phase transition events, nDTW, SR judge, and
trajectory plots.
"""

from __future__ import annotations

import argparse
import importlib.util
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
DEFAULT_CLASSES = ["Turn", "Approach"]


def main() -> int:
    args = _parser().parse_args()
    config = _config_from_args(args)
    summary = _config_summary(config, args.eval_id)
    print(json.dumps({"uav_flow_manual_eval_config": summary}, indent=2, ensure_ascii=False))
    if config.real_unrealcv:
        preflight = _real_unrealcv_preflight(config)
        print(json.dumps({"real_unrealcv_preflight": preflight}, indent=2, ensure_ascii=False))
        if preflight["status"] != "ok":
            return 2
        if args.preflight_only:
            return 0
        print(
            "Reminder: this run will launch real UnrealCV/UE. "
            "Check for old python/Collection/UE4/Unreal processes before running long batches."
        )
    elif args.preflight_only:
        print("preflight_only skipped because --dry-run disables real UnrealCV.")
        return 0
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
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated class names.")
    parser.add_argument("--limit-per-class", type=int, default=1, help="Number of cases per class.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional total case cap after class filtering.")
    parser.add_argument("--dry-run", action="store_true", help="Use dry UnrealCV mode instead of launching real UE.")
    parser.add_argument("--no-reuse-real-env", action="store_true", help="Disable real UE reuse across cases.")
    parser.add_argument("--no-sr-judge", action="store_true", help="Disable SR judge.")
    parser.add_argument("--no-visual-diagnostics", action="store_true", help="Disable visual diagnostics.")
    parser.add_argument("--no-action-planner-debug", action="store_true", help="Disable action planner debug artifacts.")
    parser.add_argument("--runtime-loop-guard-windows", type=int, default=40, help="Runtime loop guard window count.")
    parser.add_argument("--parser-mode", default="model", help="Task parser mode.")
    parser.add_argument("--env-path", default=None, help="Optional .env path for model config.")
    parser.add_argument("--stmr-mode", choices=["off", "shadow", "planner"], default=None, help="Optional online STMR experiment mode.")
    parser.add_argument("--stmr-prior", default=None, help="Named STMR detector-query prior.")
    parser.add_argument("--stmr-max-queries", type=int, default=None, help="Maximum task-plus-prior detector queries.")
    parser.add_argument("--stmr-device", choices=["auto", "cuda", "cpu"], default=None, help="Device for GroundingDINO and SAM.")
    parser.add_argument("--preflight-only", action="store_true", help="Check real UnrealCV imports and exit before launching UE.")
    return parser


def _config_from_args(args: argparse.Namespace) -> UavFlowEvalConfig:
    base = UavFlowEvalConfig.from_file(args.config)
    data = base.to_dict()
    data.update(
        {
            "real_unrealcv": not args.dry_run,
            "reuse_real_unrealcv_env": not args.no_reuse_real_env,
            "classes": _parse_classes(args.classes),
            "limit_per_class": args.limit_per_class,
            "enable_sr_judge": not args.no_sr_judge,
            "enable_visual_diagnostics": not args.no_visual_diagnostics,
            "enable_action_planner_debug": not args.no_action_planner_debug,
            "runtime_loop_guard_windows": args.runtime_loop_guard_windows,
            "parser_mode": args.parser_mode,
        }
    )
    if args.max_cases is not None:
        data["max_cases"] = args.max_cases
    if args.env_path:
        data["env_path"] = args.env_path
    if args.stmr_mode:
        data["stmr_mode"] = args.stmr_mode
    if args.stmr_prior:
        data["stmr_prior"] = args.stmr_prior
    if args.stmr_max_queries is not None:
        data["stmr_max_queries"] = args.stmr_max_queries
    if args.stmr_device:
        data["stmr_device"] = args.stmr_device
    return UavFlowEvalConfig.from_dict(data)


def _parse_classes(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_CLASSES)
    return [item.strip() for item in value.split(",") if item.strip()]


def _config_summary(config: UavFlowEvalConfig, eval_id: str | None) -> dict[str, Any]:
    return {
        "eval_id": eval_id,
        "uav_flow_repo_path": config.uav_flow_repo_path,
        "output_dir": config.output_dir,
        "classes": config.classes,
        "limit_per_class": config.limit_per_class,
        "max_cases": config.max_cases,
        "real_unrealcv": config.real_unrealcv,
        "reuse_real_unrealcv_env": config.reuse_real_unrealcv_env,
        "enable_sr_judge": config.enable_sr_judge,
        "enable_visual_diagnostics": config.enable_visual_diagnostics,
        "enable_action_planner_debug": config.enable_action_planner_debug,
        "runtime_loop_guard_windows": config.runtime_loop_guard_windows,
        "parser_mode": config.parser_mode,
        "env_path": config.env_path,
        "stmr_mode": config.stmr_mode,
        "stmr_prior": config.stmr_prior,
        "stmr_max_queries": config.stmr_max_queries,
        "stmr_device": config.stmr_device,
    }


def _real_unrealcv_preflight(config: UavFlowEvalConfig) -> dict[str, Any]:
    repo = Path(config.uav_flow_repo_path)
    result: dict[str, Any] = {
        "status": "ok",
        "python_executable": sys.executable,
        "uav_flow_repo_path": str(repo),
        "repo_exists": repo.exists(),
        "imports": {},
    }
    if not repo.exists():
        result["status"] = "failed"
        result["reason"] = "uav_flow_repo_path_missing"
        return result
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    required_modules = ["gym", "gym_unrealcv", "unrealcv"]
    if config.stmr_mode != "off":
        required_modules.extend(["torch", "transformers"])
    for module_name in required_modules:
        spec = importlib.util.find_spec(module_name)
        result["imports"][module_name] = {
            "found": spec is not None,
            "origin": getattr(spec, "origin", None) if spec else None,
        }
    missing = [name for name, item in result["imports"].items() if not item["found"]]
    if missing:
        result["status"] = "failed"
        result["reason"] = "missing_python_modules"
        result["missing_modules"] = missing
        return result
    try:
        from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation  # noqa: F401
    except Exception as exc:
        result["status"] = "failed"
        result["reason"] = "gym_unrealcv_wrapper_import_failed"
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
