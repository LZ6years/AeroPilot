"""Run AerialVLN episodes through native continuous UAV-Agent primitives."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmarks.aerial_vln.adapter import (
    AERIAL_VLN_PROTOCOL,
    AerialVlnCaseAdapter,
)
from evaluation.benchmarks.aerial_vln.cases import (
    AerialVlnCase,
    load_aerial_vln_cases,
)
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.metrics import AerialVlnMetricSuite
from evaluation.benchmarks.core import (
    BenchmarkAuditReader,
    BenchmarkCaseReview,
    BenchmarkEpisodeExecutor,
    BenchmarkReviewRenderer,
    BenchmarkSummaryReview,
    EpisodeAuditCursor,
    FrameAssetResolver,
    NoopScenarioController,
    ReviewSection,
    write_audit_json,
)
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.agent.semantic.task_parser import TaskParser
from src.env_client import (
    ObservationProfile,
    SimulatorEnvClient,
    load_observation_profile,
)
from src.runtime.configs import load_map_config
from src.runtime.configs.action_config import load_uav_action_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.configs.safety_config import load_uav_safety_config
from src.runtime.configs.stabilization_config import (
    UavStabilizationConfig,
)
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.server import SimulationServer


class AerialVlnBenchmarkRunner:
    """AerialVLN benchmark composition over the shared executor."""

    def __init__(self, config: AerialVlnEvalConfig) -> None:
        self.config = config
        self.model_config = load_model_config(config.env_path)
        self.action_config = _action_config(config)
        self.safety_config = _safety_config(config)
        self.map_packages, self.port_policy = load_map_config(
            config.map_config_path
        )
        self.map_packages_by_id = {
            package.map_id: package for package in self.map_packages
        }
        self.observation_profile = (
            load_observation_profile(config.observation_profile_path)
            if config.observation_profile_path
            else ObservationProfile()
        )
        self._case_reviews: dict[str, BenchmarkCaseReview] = {}
        self._validate_review_config()
        if self.observation_profile.modalities != (
            "rgb",
            "depth",
            "pose",
        ):
            raise ValueError(
                "AerialVLN v1 requires observation modalities "
                "rgb, depth, pose"
            )
        self._validate_scene_map()

    def load_cases(self) -> list[AerialVlnCase]:
        cases = load_aerial_vln_cases(
            self.config.dataset_root,
            self.config.split,
            max_cases=self.config.max_cases,
            scene_filter=self.config.scene_filter,
        )
        unknown = sorted(
            {
                case.scene_id
                for case in cases
                if case.scene_id not in self.config.scene_map
            }
        )
        if unknown:
            raise ValueError(
                "AerialVLN scenes have no scene_map entry: "
                + ", ".join(unknown)
            )
        return cases

    def run(self, *, eval_id: str | None = None) -> dict[str, Any]:
        cases = self.load_cases()
        eval_dir = _create_eval_dir(self.config.output_dir, eval_id)
        results = [self.run_case(case, eval_dir) for case in cases]
        summary = _summary(results)
        _write_json(eval_dir / "summary.json", summary)
        (eval_dir / "summary.md").write_text(
            _summary_markdown(summary), encoding="utf-8"
        )
        manifest = {
            "benchmark": "aerial_vln",
            "protocol_variant": AERIAL_VLN_PROTOCOL,
            "dataset_split": self.config.split,
            "dataset_version": self.config.dataset_version,
            "upstream_reference_commit": (
                self.config.upstream_reference_commit
            ),
            "scene_map": dict(self.config.scene_map),
            "observation_profile": self.observation_profile.to_dict(),
            "runtime_loop_guard_windows": (
                self.config.runtime_loop_guard_windows
            ),
            "initialization_min_settle_sec": (
                self.config.initialization_min_settle_sec
            ),
            "case_count": len(results),
        }
        _write_json(eval_dir / "manifest.json", manifest)
        review_paths: dict[str, str] = {}
        if self.config.generate_html_review:
            review_paths = BenchmarkReviewRenderer().write(
                eval_dir,
                BenchmarkSummaryReview(
                    benchmark_name="AerialVLN",
                    protocol_variant=AERIAL_VLN_PROTOCOL,
                    metrics=dict(summary.get("metrics") or {}),
                    cases=tuple(
                        {
                            "case_id": item.get("episode_id"),
                            "status": item.get("status"),
                            "metrics": item.get("metrics") or {},
                        }
                        for item in results
                    ),
                ),
                tuple(
                    self._case_reviews[item["episode_id"]]
                    for item in results
                    if item.get("episode_id") in self._case_reviews
                ),
            )
        return {
            "eval_dir": str(eval_dir),
            "summary": summary,
            "results": results,
            "summary_paths": {
                "summary_json": str(eval_dir / "summary.json"),
                "summary_md": str(eval_dir / "summary.md"),
                "manifest": str(eval_dir / "manifest.json"),
                **review_paths,
            },
        }

    def run_case(
        self, case: AerialVlnCase, eval_dir: str | Path
    ) -> dict[str, Any]:
        case_dir = Path(eval_dir) / "cases" / case.episode_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "evaluation_truth.json", case.evaluation_truth())
        run_dir = RunStore(self.config.run_root).create_run_dir(
            f"aerial_vln_{case.episode_id}"
        )
        recorder = AsyncRunLogger(run_dir)
        execution = None
        setup_error = None
        try:
            runtime = self._make_runtime(run_dir, recorder)
            episode = AerialVlnCaseAdapter(
                scene_map=self.config.scene_map,
                observation_profile=self.observation_profile,
            ).prepare_case(case)
            execution = BenchmarkEpisodeExecutor().execute(
                episode,
                runtime=runtime,
                scenario_controller=NoopScenarioController(),
                metric_suite=AerialVlnMetricSuite(),
                run_dir=run_dir,
            )
        except Exception as exc:
            setup_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            recorder.event("aerial_vln_case_failed", setup_error)
        finally:
            recorder.close()

        audit_cursor = (
            execution.audit_cursor
            if execution is not None
            else EpisodeAuditCursor.freeze(run_dir)
        )
        audit = BenchmarkAuditReader().read(audit_cursor)
        frame_assets = FrameAssetResolver().materialize(audit, case_dir)
        audit_path = write_audit_json(
            case_dir / "audit.json", audit, frame_assets
        )

        if execution is None:
            outcome = {
                "status": "error",
                "completed": False,
                "stopped": False,
                "stopped_reason": None,
                "stop_source": None,
                "error": setup_error,
                "cleanup_errors": [],
            }
            metrics: dict[str, Any] = {}
            trace = {
                "samples": [],
                "action_count": 0,
                "coordinate_frames": [],
            }
            capabilities = None
            metric_error = None
        else:
            outcome = execution.outcome.to_dict()
            metrics = dict(execution.metrics)
            trace = execution.trace.to_dict()
            capabilities = execution.scene_capabilities
            metric_error = execution.metric_error
            if metric_error is not None and outcome["status"] == "ok":
                outcome["status"] = "error"
                outcome["error"] = metric_error

        trajectory = {
            "coordinate_frame": "airsim_ned_m",
            "samples": metrics.get("trajectory_ned", []),
            "uav_action_steps": metrics.get(
                "uav_action_steps", trace["action_count"]
            ),
        }
        case_result = {
            "episode_id": case.episode_id,
            "trajectory_id": case.trajectory_id,
            "scene_id": case.scene_id,
            "protocol_variant": AERIAL_VLN_PROTOCOL,
            "run_dir": str(run_dir),
            **outcome,
            "metrics": metrics,
            "metric_error": metric_error,
            "scene_capabilities": capabilities,
            "observation_profile": self.observation_profile.to_dict(),
            "audit_path": audit_path,
        }
        _write_json(case_dir / "case_result.json", case_result)
        _write_json(case_dir / "trajectory.json", trajectory)
        _write_json(case_dir / "trace.json", trace)
        _write_json(case_dir / "metrics.json", metrics)
        _write_json(
            case_dir / "manifest.json",
            {
                "dataset_split": self.config.split,
                "dataset_version": self.config.dataset_version,
                "upstream_reference_commit": (
                    self.config.upstream_reference_commit
                ),
                "protocol_variant": AERIAL_VLN_PROTOCOL,
                "scene_capabilities": capabilities,
                "observation_profile": self.observation_profile.to_dict(),
                "runtime_loop_guard_windows": (
                    self.config.runtime_loop_guard_windows
                ),
                "initialization_min_settle_sec": (
                    self.config.initialization_min_settle_sec
                ),
            },
        )
        self._case_reviews[case.episode_id] = BenchmarkCaseReview(
            case_id=case.episode_id,
            benchmark_name="AerialVLN",
            protocol_variant=AERIAL_VLN_PROTOCOL,
            instruction=case.instruction,
            outcome={
                key: outcome.get(key)
                for key in (
                    "status",
                    "completed",
                    "stopped",
                    "stopped_reason",
                    "stop_source",
                    "error",
                    "cleanup_errors",
                )
            },
            metrics={
                key: value
                for key, value in metrics.items()
                if key not in {"trajectory_ned", "reference_path_ned"}
            },
            audit=audit,
            frame_assets=frame_assets,
            sections=(
                ReviewSection(
                    "Trajectory",
                    "key_value",
                    {
                        "coordinate_frame": "airsim_ned_m",
                        "executed_samples": len(
                            metrics.get("trajectory_ned", [])
                        ),
                        "reference_samples": len(
                            metrics.get("reference_path_ned", [])
                        ),
                        "path_length": metrics.get("path_length"),
                    },
                ),
                ReviewSection(
                    "Protocol Notes",
                    "text",
                    (
                        "AerialVLN data and metrics with UAV-Agent native "
                        "continuous actions; coordinates are reported in "
                        "AirSim NED meters."
                    ),
                ),
            ),
            metadata={
                "trajectory_id": case.trajectory_id,
                "scene_id": case.scene_id,
            },
        )
        return {
            **case_result,
            "case_dir": str(case_dir),
        }

    def _make_runtime(
        self,
        run_dir: Path,
        recorder: AsyncRunLogger,
        *,
        task_parser: Any | None = None,
        planner: Any | None = None,
    ) -> TaskRuntime:
        backend_registry = None
        if self.config.execution_mode == "dry":
            backend_registry = _dry_airsim_registry()
            server_mode = SimulationExecutionMode.ATTACH
        else:
            server_mode = SimulationExecutionMode.parse(
                self.config.execution_mode
            )
        server = SimulationServer(
            self.map_packages,
            port_policy=self.port_policy,
            run_dir=run_dir,
            execution_mode=server_mode,
            show_display=not self.config.headless,
            action_config=self.action_config,
            save_artifacts=self.config.save_artifacts,
            backend_registry=backend_registry,
        )
        return TaskRuntime(
            SimulatorEnvClient(
                server, observation_profile=self.observation_profile
            ),
            recorder,
            task_parser
            or TaskParser(
                mode=self.config.parser_mode,
                model_config=self.model_config,
            ),
            planner=planner or ModelActionPlanner(self.model_config),
            safety=SafetySupervisor(self.safety_config),
            shutdown_policy=self.config.shutdown_policy,
            action_config=self.action_config,
            stabilization_config=UavStabilizationConfig(
                initialization_min_settle_sec=(
                    self.config.initialization_min_settle_sec
                )
            ),
            runtime_loop_guard_windows=self.config.runtime_loop_guard_windows,
            enable_action_planner_debug=(
                self.config.enable_action_planner_debug
            ),
        )

    def _validate_scene_map(self) -> None:
        missing = sorted(
            {
                map_id
                for map_id in self.config.scene_map.values()
                if map_id not in self.map_packages_by_id
            }
        )
        if missing:
            raise ValueError(
                "AerialVLN scene_map references missing MapPackages: "
                + ", ".join(missing)
            )
        wrong_scene = sorted(
            scene_id
            for scene_id, map_id in self.config.scene_map.items()
            if str(self.map_packages_by_id[map_id].scene_id)
            != str(scene_id)
        )
        if wrong_scene:
            raise ValueError(
                "AerialVLN scene_map keys must match MapPackage scene_id: "
                + ", ".join(wrong_scene)
            )

    def _validate_review_config(self) -> None:
        """Require persistent evidence for real, portable HTML reviews."""

        if (
            self.config.execution_mode == "dry"
            or not self.config.generate_html_review
        ):
            return
        missing = []
        if not self.config.save_artifacts:
            missing.append("save_artifacts=true")
        if not self.config.enable_action_planner_debug:
            missing.append("enable_action_planner_debug=true")
        if missing:
            raise ValueError(
                "Real AerialVLN HTML review requires "
                + " and ".join(missing)
            )


def _dry_airsim_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: AirSimBackend(
            use_real_uav=False,
            action_config=context.action_config,
            artifact_dir=context.artifact_dir,
            save_artifacts=context.save_artifacts,
        ),
    )
    return registry


def _action_config(config: AerialVlnEvalConfig) -> Any:
    loaded = load_uav_action_config()
    data = loaded.to_dict()
    data.update(config.action_config_overrides)
    return loaded.__class__.from_dict(data)


def _safety_config(config: AerialVlnEvalConfig) -> Any:
    loaded = load_uav_safety_config()
    data = loaded.to_dict()
    data.update(config.safety_overrides)
    return loaded.__class__.from_dict(data)


def _create_eval_dir(root: str | Path, eval_id: str | None) -> Path:
    identifier = eval_id or datetime.now(timezone.utc).strftime(
        "aerial_vln_%Y%m%d_%H%M%S"
    )
    path = Path(root) / identifier
    path.mkdir(parents=True, exist_ok=False)
    return path


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [
        item["metrics"]
        for item in results
        if isinstance(item.get("metrics"), dict) and item["metrics"]
    ]
    keys = (
        "distance_to_goal",
        "success",
        "oracle_success",
        "ndtw",
        "sdtw",
        "path_length",
        "steps_taken",
        "uav_action_steps",
    )
    aggregates = {
        key: (
            sum(float(row[key]) for row in metric_rows) / len(metric_rows)
            if metric_rows
            else None
        )
        for key in keys
    }
    return {
        "benchmark": "aerial_vln",
        "protocol_variant": AERIAL_VLN_PROTOCOL,
        "case_count": len(results),
        "evaluated_count": len(metric_rows),
        "metrics": aggregates,
        "cases": [
            {
                "episode_id": item.get("episode_id"),
                "status": item.get("status"),
                "metrics": item.get("metrics"),
            }
            for item in results
        ],
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AerialVLN Native-Action Summary",
        "",
        f"- Protocol: `{summary['protocol_variant']}`",
        f"- Cases: {summary['case_count']}",
        f"- Evaluated: {summary['evaluated_count']}",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    for key, value in summary["metrics"].items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
