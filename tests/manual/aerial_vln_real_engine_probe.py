"""Real airvln_env_1 probe for initial pose, front RGB/depth, and cleanup."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.aerial_vln.adapter import AerialVlnCaseAdapter
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.metrics import AerialVlnMetricSuite
from evaluation.benchmarks.aerial_vln.runner import (
    AerialVlnBenchmarkRunner,
)
from evaluation.benchmarks.core import (
    BenchmarkEpisodeExecutor,
    NoopScenarioController,
)
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from tests.manual.benchmark_probe_support import (
    CompleteProbePlanner,
    StaticProbeTaskParser,
)


def main() -> int:
    args = _parser().parse_args()
    base = AerialVlnEvalConfig.from_file(args.config)
    data = base.to_dict()
    data.update(
        {
            "execution_mode": args.execution_mode,
            "save_artifacts": True,
            "enable_action_planner_debug": True,
            "max_cases": 1,
            "scene_filter": ["1"],
        }
    )
    config = AerialVlnEvalConfig.from_dict(data)
    runner = AerialVlnBenchmarkRunner(config)
    cases = runner.load_cases()
    if not cases:
        raise ValueError("No scene_id=1 AerialVLN case is available")
    case = cases[0]
    probe_dir = Path(args.output_dir) / case.episode_id
    probe_dir.mkdir(parents=True, exist_ok=False)
    run_dir = RunStore(probe_dir / "runs").create_run_dir("runtime")
    recorder = AsyncRunLogger(run_dir)
    try:
        runtime = runner._make_runtime(
            run_dir,
            recorder,
            task_parser=StaticProbeTaskParser(),
            planner=CompleteProbePlanner(),
        )
        episode = AerialVlnCaseAdapter(
            scene_map=config.scene_map,
            observation_profile=runner.observation_profile,
        ).prepare_case(case)
        execution = BenchmarkEpisodeExecutor().execute(
            episode,
            runtime=runtime,
            scenario_controller=NoopScenarioController(),
            metric_suite=AerialVlnMetricSuite(),
            run_dir=run_dir,
        )
    finally:
        recorder.close()

    trajectory = execution.metrics.get("trajectory_ned", [])
    first_ned = trajectory[0] if trajectory else None
    roundtrip_error = (
        math.sqrt(
            sum(
                (float(first_ned[index]) - case.start_position[index])
                ** 2
                for index in range(3)
            )
        )
        if first_ned is not None
        else math.inf
    )
    frames = _read_jsonl(run_dir / "memory" / "frames.jsonl")
    modalities = {
        item.get("modality")
        for item in frames
        if item.get("observation_id") == "obs_0001"
    }
    checks = {
        "runtime_status_ok": execution.outcome.status == "ok",
        "cleanup_errors_empty": not execution.outcome.cleanup_errors,
        "initial_ned_roundtrip_lte_0_05m": roundtrip_error <= 0.05,
        "front_rgb_depth_present": {"rgb", "depth"}.issubset(modalities),
        "multirotor_takeoff_capability": bool(
            execution.scene_capabilities
            and execution.scene_capabilities.get("requires_takeoff")
        ),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "episode_id": case.episode_id,
        "run_dir": str(run_dir),
        "protocol_variant": episode.protocol_variant,
        "checks": checks,
        "initial_ned_roundtrip_error_m": roundtrip_error,
        "dataset_start_position_ned": list(case.start_position),
        "first_runtime_position_ned": first_ned,
        "observation_profile": runner.observation_profile.to_dict(),
        "scene_capabilities": execution.scene_capabilities,
        "metrics": execution.metrics,
        "metric_error": execution.metric_error,
    }
    (probe_dir / "probe_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "evaluation/benchmarks/aerial_vln/configs/"
            "aerial_vln_eval.example.json"
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=["attach", "managed"],
        default="attach",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/manual_outputs/aerial_vln_real_engine_probe",
    )
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
