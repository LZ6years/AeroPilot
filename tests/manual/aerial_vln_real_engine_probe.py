"""Real AerialVLN scene probe for pose, front RGB/depth, and cleanup.

This is a low-level MapPackage/Backend gate, not a benchmark result. Dataset
difficulty selection and full evaluation remain owned by the production runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.aerial_vln.adapter import AerialVlnCaseAdapter
from evaluation.benchmarks.aerial_vln.cases import load_aerial_vln_dataset
from evaluation.benchmarks.aerial_vln.config import (
    AerialVlnEvalConfig,
    aerial_vln_split_label,
)
from evaluation.benchmarks.aerial_vln.metrics import AerialVlnMetricSuite
from evaluation.benchmarks.aerial_vln.runner import (
    AerialVlnBenchmarkRunner,
)
from evaluation.benchmarks.core import (
    BenchmarkAuditReader,
    BenchmarkCaseReview,
    BenchmarkEpisodeExecutor,
    BenchmarkReviewRenderer,
    BenchmarkSummaryReview,
    FrameAssetResolver,
    NoopScenarioController,
    ReviewSection,
    write_audit_json,
)
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from tests.manual.benchmark_probe_support import (
    CompleteProbePlanner,
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
        }
    )
    config = AerialVlnEvalConfig.from_dict(data)
    runner = AerialVlnBenchmarkRunner(config)
    cases, provenance = load_aerial_vln_dataset(
        config.dataset_root,
        config.split,
        dataset_family=config.dataset_family,
        dataset_version=config.dataset_version,
    )
    if (
        config.expected_split_sha256 is not None
        and provenance.split_sha256 != config.expected_split_sha256
    ):
        raise ValueError("AerialVLN split SHA256 mismatch")
    if config.dataset_family == "aerialvln-s":
        if config.expected_split_sha256 is None:
            raise ValueError(
                "Real AerialVLN-S probe requires expected_split_sha256"
            )
        dataset_version = config.dataset_version.strip()
        if not dataset_version or dataset_version.lower() == "unspecified":
            raise ValueError(
                "Real AerialVLN-S probe requires dataset_version"
            )
    candidates = [
        case
        for case in cases
        if case.scene_id == args.scene_id
        and (
            args.episode_id is None
            or case.episode_id == args.episode_id
        )
    ]
    if not candidates:
        raise ValueError(
            f"No AerialVLN case is available for scene_id={args.scene_id!r}"
        )
    case = candidates[0]
    map_id = config.scene_map.get(case.scene_id)
    if map_id is None or map_id not in runner.map_packages_by_id:
        raise ValueError(
            f"scene_id={case.scene_id!r} is not execution-ready"
        )
    map_package = runner.map_packages_by_id[map_id]
    original_timeout = float(map_package.launch.get("ready_timeout_sec", 120))
    if args.ready_timeout_sec is not None:
        if args.ready_timeout_sec <= 0:
            raise ValueError("ready timeout must be positive")
        # This runner owns its package objects. Never rewrite source map JSON
        # merely to run a diagnostic with a longer observation ceiling.
        map_package.launch["ready_timeout_sec"] = args.ready_timeout_sec
        map_package.launch["diagnostic_independent_tunnel"] = True
        # Production may reserve just one serial slot. The diagnostic needs a
        # second local forwarding port, not a second simulator or remote slot.
        runner.port_policy = replace(runner.port_policy,
                                     max_scene_num=max(2, runner.port_policy.max_scene_num))
    if str(map_package.scene_id) != case.scene_id:
        raise ValueError(
            f"scene_id={case.scene_id!r} routes to map_id={map_id!r}, "
            f"whose MapPackage declares scene_id={map_package.scene_id!r}"
        )
    probe_dir = Path(args.output_dir) / case.episode_id
    probe_dir.mkdir(parents=True, exist_ok=False)
    run_dir = RunStore(probe_dir / "runs").create_run_dir("runtime")
    recorder = AsyncRunLogger(run_dir)
    try:
        runtime = runner._make_runtime(
            run_dir,
            recorder,
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
    capabilities = execution.scene_capabilities or {}
    lifecycle = [record for path in (run_dir / "scenes").glob("*/lifecycle.jsonl")
                 for record in _read_jsonl(path)]
    server = runtime.env_client.service
    remote = server.remote_scene_control
    scene_ids = list(server.lifecycle)
    cleanup_checks = []
    for scene_id in scene_ids:
        try:
            status = remote.status(map_package, scene_id)
            host, port = server._remote_endpoint(map_package)
            endpoint = remote.probe_prepared_port(map_package, scene_id, host, port)
            cleanup_checks.append({"scene_id": scene_id, "process": status,
                                   "port": endpoint,
                                   "verified": status.get("status") == "stopped"
                                   and endpoint.get("status") == "unreachable"})
        except Exception as exc:
            cleanup_checks.append({"scene_id": scene_id, "verified": False, "error": str(exc)})
    from PIL import Image
    import numpy as np
    dimensions = []
    for frame in frames:
        path = Path(str(frame.get("path") or frame.get("metadata", {}).get("artifact_path", "")))
        if not path.is_file():
            continue
        if frame.get("modality") == "rgb":
            with Image.open(path) as image:
                dimensions.append(list(image.size))
        elif frame.get("modality") == "depth" and path.suffix == ".npy":
            dimensions.append(list(np.load(path).shape[:2][::-1]))
    expected_control_mode = str(
        map_package.airsim.get("control_strategy", "pose_kinematic")
    )
    checks = {
        "runtime_status_ok": execution.outcome.status == "ok",
        "cleanup_errors_empty": not execution.outcome.cleanup_errors,
        "initial_ned_roundtrip_lte_0_05m": roundtrip_error <= 0.05,
        "front_rgb_depth_present": {"rgb", "depth"}.issubset(modalities),
        "owned_process_and_remote_port_released": bool(cleanup_checks) and all(item["verified"] for item in cleanup_checks),
        "captured_dimensions_256": bool(dimensions) and all(size == [256, 256] for size in dimensions),
        "local_tunnel_port_released": server.port_manager._is_available(server.port_policy.control_port_start),
        **capability_contract_checks(expected_control_mode, capabilities),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "episode_id": case.episode_id,
        "dataset": provenance.to_dict(),
        "dataset_split_label": aerial_vln_split_label(config.split),
        "run_dir": str(run_dir),
        "protocol_variant": episode.protocol_variant,
        "checks": checks,
        "initial_ned_roundtrip_error_m": roundtrip_error,
        "dataset_start_position_ned": list(case.start_position),
        "first_runtime_position_ned": first_ned,
        "observation_profile": runner.observation_profile.to_dict(),
        "scene_capabilities": capabilities,
        "metrics": execution.metrics,
        "metric_error": execution.metric_error,
        "outcome": execution.outcome.to_dict(),
        "lifecycle": lifecycle,
        "cleanup_verification": cleanup_checks,
        "original_ready_timeout_sec": original_timeout,
        "diagnostic_ready_timeout_sec": args.ready_timeout_sec,
        "rpc_exceeded_original_timeout": any(
            item["stage"] == "rpc_ready" and item["elapsed_sec"] - next(
                (start["elapsed_sec"] for start in lifecycle if start["stage"] == "process_started"), 0
            ) > original_timeout for item in lifecycle),
    }
    audit = BenchmarkAuditReader().read(execution.audit_cursor)
    case_dir = probe_dir / "cases" / case.episode_id
    frame_assets = FrameAssetResolver().materialize(audit, case_dir)
    summary["audit_path"] = write_audit_json(
        case_dir / "audit.json", audit, frame_assets
    )
    review_paths = BenchmarkReviewRenderer().write(
        probe_dir,
        BenchmarkSummaryReview(
            benchmark_name="AerialVLN Infrastructure Probe",
            protocol_variant=episode.protocol_variant,
            metrics={
                "status": summary["status"],
                "dataset_split": config.split,
                "dataset_split_label": aerial_vln_split_label(config.split),
                "initial_ned_roundtrip_error_m": roundtrip_error,
            },
            cases=(
                {
                    "case_id": case.episode_id,
                    "status": summary["status"],
                    "metrics": checks,
                },
            ),
        ),
        (
            BenchmarkCaseReview(
                case_id=case.episode_id,
                benchmark_name="AerialVLN Infrastructure Probe",
                protocol_variant=episode.protocol_variant,
                instruction=case.instruction,
                outcome=execution.outcome.to_dict(),
                metrics={
                    **checks,
                    "initial_ned_roundtrip_error_m": roundtrip_error,
                },
                audit=audit,
                frame_assets=frame_assets,
                sections=(
                    ReviewSection(
                        "Probe Contract",
                        "key_value",
                        {
                            "benchmark_result": False,
                            "dataset_split": config.split,
                            "dataset_split_label": aerial_vln_split_label(
                                config.split
                            ),
                            "scene_id": case.scene_id,
                            "map_id": map_id,
                            "expected_control_mode": expected_control_mode,
                        },
                    ),
                    ReviewSection(
                        "Coordinate Roundtrip",
                        "key_value",
                        {
                            "dataset_start_position_ned": list(
                                case.start_position
                            ),
                            "first_runtime_position_ned": first_ned,
                            "error_m": roundtrip_error,
                        },
                    ),
                ),
            ),
        ),
    )
    summary["review_paths"] = review_paths
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
        default="managed",
    )
    parser.add_argument("--scene-id", default="1")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--ready-timeout-sec", type=float, default=None,
                        help="Diagnostic override only; source map JSON is unchanged.")
    parser.add_argument(
        "--output-dir",
        default=(
            "runs/experiments/aerial_vln_real_engine_probe/"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
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


def capability_contract_checks(
    expected_control_mode: str,
    capabilities: dict,
) -> dict[str, bool]:
    """Validate no-takeoff semantics against the configured strategy."""

    return {
        "control_mode_matches_map_package": (
            capabilities.get("control_mode") == expected_control_mode
        ),
        "takeoff_requirement_matches_control_mode": (
            capabilities.get("requires_takeoff")
            is False
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
