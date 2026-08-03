"""Run a real AerialVLN batch stratified by reference-path length.

This is an evaluation probe, not benchmark production logic. It keeps path
length and difficulty labels outside Runtime/model context, selects evenly
spaced cases from low/medium/high tertiles, and writes audit files alongside
the normal AerialVLN report.

Example:

    $env:FLYAGENT_SSH_KEY = "C:\\Users\\me\\.ssh\\id_rsa"
    python tests/manual/aerial_vln_remote_difficulty_batch.py \
      --dataset-root G:/datasets/aerialvln-v8 \
      --map-config configs/maps/remote_ssh.example.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.aerial_vln.cases import AerialVlnCase
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner


DIFFICULTIES = ("low", "medium", "high")
AGGREGATE_METRICS = (
    "distance_to_goal",
    "success",
    "oracle_success",
    "ndtw",
    "sdtw",
    "path_length",
    "steps_taken",
    "uav_action_steps",
)


def main() -> int:
    args = _parser().parse_args()
    config = AerialVlnEvalConfig.from_dict(
        {
            "dataset_root": args.dataset_root,
            "split": args.split,
            "output_dir": args.output_dir,
            "run_root": args.run_root,
            "scene_filter": [args.scene_id],
            "map_config_path": args.map_config,
            "scene_map": {args.scene_id: args.map_id},
            "observation_profile_path": args.observation_profile,
            "execution_mode": "managed",
            "headless": args.headless,
            "save_artifacts": True,
            "enable_action_planner_debug": True,
            "env_path": args.env_path,
            "parser_mode": "model",
            "shutdown_policy": "hover",
            "runtime_loop_guard_windows": args.loop_guard_windows,
            "initialization_min_settle_sec": args.initialization_settle_sec,
            "dataset_version": args.dataset_version,
            "upstream_reference_commit": args.upstream_reference_commit,
        }
    )
    runner = AerialVlnBenchmarkRunner(config)
    all_cases = runner.load_cases()
    selected, selection = select_difficulty_cases(
        all_cases, cases_per_difficulty=args.cases_per_difficulty
    )
    selection["execution"] = {
        "protocol_variant": (
            "aerial_vln.uav_agent_native_actions.v1"
        ),
        "execution_mode": config.execution_mode,
        "runtime_loop_guard_windows": config.runtime_loop_guard_windows,
        "initialization_min_settle_sec": (
            config.initialization_min_settle_sec
        ),
        "observation_profile": runner.observation_profile.to_dict(),
        "map_config_path": config.map_config_path,
        "map_id": args.map_id,
    }
    print(json.dumps(selection, indent=2, ensure_ascii=False), flush=True)

    # The production runner remains unchanged. This manual probe supplies only
    # the selected cases and wraps per-case execution to expose progress.
    runner.load_cases = lambda: selected  # type: ignore[method-assign]
    original_run_case = runner.run_case

    def reporting_run_case(
        case: AerialVlnCase, eval_dir: str | Path
    ) -> dict[str, Any]:
        label = selection["cases_by_episode"][case.episode_id]["difficulty"]
        length = selection["cases_by_episode"][case.episode_id][
            "reference_path_length_m"
        ]
        print(
            f"START difficulty={label} episode={case.episode_id} "
            f"reference_path_m={length:.3f}",
            flush=True,
        )
        result = original_run_case(case, eval_dir)
        print(
            f"DONE difficulty={label} episode={case.episode_id} "
            f"status={result.get('status')} "
            f"distance_to_goal="
            f"{result.get('metrics', {}).get('distance_to_goal')}",
            flush=True,
        )
        return result

    runner.run_case = reporting_run_case  # type: ignore[method-assign]
    result = runner.run(eval_id=args.eval_id)
    eval_dir = Path(result["eval_dir"])
    readiness = _first_frame_readiness(result["results"])
    difficulty_summary = _difficulty_summary(
        result["results"], selection, readiness
    )
    _write_json(eval_dir / "difficulty_selection.json", selection)
    _write_json(eval_dir / "first_frame_readiness.json", readiness)
    _write_json(eval_dir / "difficulty_summary.json", difficulty_summary)
    (eval_dir / "difficulty_summary.md").write_text(
        _difficulty_summary_markdown(difficulty_summary),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "eval_dir": str(eval_dir),
                "difficulty_summary": difficulty_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    invalid_frames = [
        item
        for item in readiness["cases"]
        if not item["rgb_non_degenerate"]
    ]
    cleanup_failures = [
        item
        for item in result["results"]
        if item.get("cleanup_errors")
    ]
    return 2 if invalid_frames or cleanup_failures else 0


def select_difficulty_cases(
    cases: Iterable[AerialVlnCase], *, cases_per_difficulty: int
) -> tuple[list[AerialVlnCase], dict[str, Any]]:
    """Split by path-length tertiles and sample evenly within each bucket."""

    if cases_per_difficulty <= 0:
        raise ValueError("cases_per_difficulty must be positive")
    rows = sorted(
        (
            {
                "case": case,
                "reference_path_length_m": reference_path_length_m(
                    case.reference_path
                ),
            }
            for case in cases
        ),
        key=lambda item: (
            item["reference_path_length_m"],
            item["case"].episode_id,
        ),
    )
    if len(rows) < cases_per_difficulty * len(DIFFICULTIES):
        raise ValueError(
            "Not enough cases for the requested difficulty sampling: "
            f"available={len(rows)} required="
            f"{cases_per_difficulty * len(DIFFICULTIES)}"
        )
    lengths = [float(item["reference_path_length_m"]) for item in rows]
    q33 = _linear_quantile(lengths, 1.0 / 3.0)
    q67 = _linear_quantile(lengths, 2.0 / 3.0)
    buckets: dict[str, list[dict[str, Any]]] = {
        "low": [],
        "medium": [],
        "high": [],
    }
    for row in rows:
        length = float(row["reference_path_length_m"])
        label = "low" if length <= q33 else "medium" if length <= q67 else "high"
        buckets[label].append(row)

    selected: list[AerialVlnCase] = []
    cases_by_episode: dict[str, dict[str, Any]] = {}
    selected_by_difficulty: dict[str, list[dict[str, Any]]] = {}
    for label in DIFFICULTIES:
        bucket = buckets[label]
        if len(bucket) < cases_per_difficulty:
            raise ValueError(
                f"Difficulty bucket {label} contains only {len(bucket)} cases"
            )
        indices = [
            min(
                len(bucket) - 1,
                int((index + 0.5) * len(bucket) / cases_per_difficulty),
            )
            for index in range(cases_per_difficulty)
        ]
        chosen = [bucket[index] for index in indices]
        selected_by_difficulty[label] = []
        for row in chosen:
            case = row["case"]
            metadata = {
                "difficulty": label,
                "episode_id": case.episode_id,
                "trajectory_id": case.trajectory_id,
                "scene_id": case.scene_id,
                "reference_path_length_m": round(
                    float(row["reference_path_length_m"]), 6
                ),
            }
            selected.append(case)
            selected_by_difficulty[label].append(metadata)
            cases_by_episode[case.episode_id] = metadata
    return selected, {
        "method": (
            "remove consecutive duplicate XYZ points, sum Euclidean "
            "segments, split at linear 1/3 and 2/3 quantiles, sample evenly "
            "within each bucket"
        ),
        "scene_case_count": len(rows),
        "thresholds_m": {"q33": q33, "q67": q67},
        "bucket_counts": {
            label: len(bucket) for label, bucket in buckets.items()
        },
        "selected_by_difficulty": selected_by_difficulty,
        "cases_by_episode": cases_by_episode,
    }


def reference_path_length_m(
    points: Iterable[Iterable[float]],
) -> float:
    """Return XYZ length after dropping consecutive duplicate positions."""

    deduplicated: list[tuple[float, float, float]] = []
    for point in points:
        values = tuple(float(value) for value in point)
        xyz = values[:3]
        if len(xyz) != 3:
            raise ValueError("reference path rows require XYZ")
        if not deduplicated or xyz != deduplicated[-1]:
            deduplicated.append(xyz)
    return sum(
        math.dist(previous, current)
        for previous, current in zip(
            deduplicated, deduplicated[1:], strict=False
        )
    )


def _linear_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _first_frame_readiness(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        run_dir = Path(str(result["run_dir"]))
        frames_path = run_dir / "memory" / "frames.jsonl"
        frames = [
            json.loads(line)
            for line in frames_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rgb = next(
            (
                item
                for item in frames
                if item.get("observation_id") == "obs_0001"
                and item.get("modality") == "rgb"
            ),
            None,
        )
        artifact_path = (
            rgb.get("metadata", {}).get("artifact_path") if rgb else None
        )
        if artifact_path and Path(artifact_path).is_file():
            image = np.asarray(
                Image.open(artifact_path).convert("RGB"), dtype=np.uint8
            )
            std = float(image.std())
            unique_colors = int(
                np.unique(image.reshape(-1, 3), axis=0).shape[0]
            )
        else:
            std = 0.0
            unique_colors = 0
        rows.append(
            {
                "episode_id": result["episode_id"],
                "rgb_path": artifact_path,
                "rgb_std": std,
                "rgb_unique_colors": unique_colors,
                "rgb_non_degenerate": std > 1.0 and unique_colors > 100,
            }
        )
    return {
        "policy": "rgb_std > 1.0 and unique_colors > 100",
        "cases": rows,
    }


def _difficulty_summary(
    results: list[dict[str, Any]],
    selection: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    metadata = selection["cases_by_episode"]
    ready_by_id = {
        row["episode_id"]: row for row in readiness["cases"]
    }
    summary: dict[str, Any] = {
        "thresholds_m": selection["thresholds_m"],
        "cases_per_difficulty": {},
    }
    for label in DIFFICULTIES:
        rows = [
            result
            for result in results
            if metadata[result["episode_id"]]["difficulty"] == label
        ]
        metric_rows = [
            row["metrics"] for row in rows if row.get("metrics")
        ]
        summary["cases_per_difficulty"][label] = {
            "case_count": len(rows),
            "status_counts": _counts(
                str(row.get("status")) for row in rows
            ),
            "valid_first_frame_count": sum(
                bool(ready_by_id[row["episode_id"]]["rgb_non_degenerate"])
                for row in rows
            ),
            "mean_metrics": {
                key: (
                    sum(float(row[key]) for row in metric_rows)
                    / len(metric_rows)
                    if metric_rows
                    else None
                )
                for key in AGGREGATE_METRICS
            },
            "cases": [
                {
                    **metadata[row["episode_id"]],
                    "status": row.get("status"),
                    "completed": row.get("completed"),
                    "stopped_reason": row.get("stopped_reason"),
                    "metrics": {
                        key: row.get("metrics", {}).get(key)
                        for key in AGGREGATE_METRICS
                    },
                    "first_frame": ready_by_id[row["episode_id"]],
                }
                for row in rows
            ],
        }
    return summary


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _difficulty_summary_markdown(summary: dict[str, Any]) -> str:
    thresholds = summary["thresholds_m"]
    lines = [
        "# AerialVLN Path-Length Difficulty Summary",
        "",
        f"- Low/medium threshold: {thresholds['q33']:.3f} m",
        f"- Medium/high threshold: {thresholds['q67']:.3f} m",
        "",
        "| Difficulty | Cases | Valid RGB | Distance to goal | nDTW | Path length | Actions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in DIFFICULTIES:
        row = summary["cases_per_difficulty"][label]
        metrics = row["mean_metrics"]
        lines.append(
            f"| {label} | {row['case_count']} | "
            f"{row['valid_first_frame_count']} | "
            f"{_format(metrics['distance_to_goal'])} | "
            f"{_format(metrics['ndtw'])} | "
            f"{_format(metrics['path_length'])} | "
            f"{_format(metrics['uav_action_steps'])} |"
        )
    return "\n".join(lines) + "\n"


def _format(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene-id", default="1")
    parser.add_argument("--map-config", required=True)
    parser.add_argument("--map-id", default="airsim_remote_env_1")
    parser.add_argument(
        "--output-dir",
        default="evaluation/benchmarks/aerial_vln/outputs",
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--observation-profile")
    parser.add_argument("--cases-per-difficulty", type=int, default=3)
    parser.add_argument("--loop-guard-windows", type=int, default=6)
    parser.add_argument(
        "--initialization-settle-sec", type=float, default=6.0
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dataset-version", default="AerialVLN-v8")
    parser.add_argument(
        "--upstream-reference-commit",
        default="local_airvln_reference",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
