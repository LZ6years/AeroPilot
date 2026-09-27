"""Compose TravelUAV episodes over FlyAgent and Benchmark Core.

This runner owns dataset selection, TravelUAV target placement, metrics, and
reports.  Scene lifecycle, native actions, safety, observations, and remote SSH
remain owned by the existing Runtime and SimulationServer stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation.benchmarks.core import (
    BenchmarkAuditReader,
    BenchmarkCaseReview,
    BenchmarkEpisodeExecutor,
    BenchmarkReviewRenderer,
    BenchmarkSummaryReview,
    EpisodeAuditCursor,
    FrameAssetResolver,
    ReviewSection,
    write_audit_json,
)
from evaluation.benchmarks.travel_uav.adapter import TravelUavCaseAdapter
from evaluation.benchmarks.travel_uav.cases import (
    TravelUavCase,
    TravelUavDatasetProvenance,
)
from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.dataset import load_travel_uav_index
from evaluation.benchmarks.travel_uav.metrics import TravelUavMetricSuite
from evaluation.benchmarks.travel_uav.protocol import (
    TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT,
    TRAVEL_UAV_PROTOCOL,
)
from evaluation.benchmarks.travel_uav.scenario import TravelUavScenarioController
from evaluation.benchmarks.travel_uav.selection import (
    TravelUavCaseSelection,
    select_travel_uav_cases,
)
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.env_client import ObservationProfile, SimulatorEnvClient, load_observation_profile
from src.runtime.configs import load_map_config
from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.configs.safety_config import UavSafetyConfig, load_uav_safety_config
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.server import SimulationServer


@dataclass(frozen=True)
class PreparedTravelUavEvaluation:
    """Selected cases plus provenance and local scene-routing readiness."""

    cases: tuple[TravelUavCase, ...]
    provenance: TravelUavDatasetProvenance
    selection: TravelUavCaseSelection
    execution_readiness: dict[str, Any]

    def validation_dict(self) -> dict[str, Any]:
        return {
            "dataset_provenance": self.provenance.to_dict(),
            "case_selection": self.selection.to_dict(),
            "execution_readiness": dict(self.execution_readiness),
        }


class TravelUavBenchmarkRunner:
    """Run TravelUAV data in the front-view no-assistant FlyAgent protocol."""

    def __init__(
        self,
        config: TravelUavEvalConfig,
        *,
        simulation_server_factory: Any = None,
        planner_factory: Callable[[Any], Any] | None = None,
        scenario_controller_factory: Callable[[], Any] | None = None,
        read_only_execution_config=None,
    ) -> None:
        self.config = config
        self.read_only_execution_config = read_only_execution_config
        self._simulation_server_factory = simulation_server_factory or SimulationServer
        self.map_packages, self.port_policy = load_map_config(config.map_config_path)
        self.map_packages_by_id = {package.map_id: package for package in self.map_packages}
        self.observation_profile = (
            load_observation_profile(config.observation_profile_path)
            if config.observation_profile_path
            else ObservationProfile()
        )
        missing = sorted(
            {"rgb", "depth", "pose"}.difference(self.observation_profile.modalities)
        )
        if missing:
            raise ValueError(
                "TravelUAV front-view v1 requires rgb, depth, pose; missing "
                + ", ".join(missing)
            )
        self.action_config = _action_config(config)
        self.safety_config = _safety_config(config)
        self._model_config_value: Any | None = None
        self._planner_factory = planner_factory
        self._scenario_controller_factory = scenario_controller_factory
        self._case_reviews: dict[str, BenchmarkCaseReview] = {}

    def prepare_evaluation(
        self, *, require_execution_ready: bool
    ) -> PreparedTravelUavEvaluation:
        population, provenance = load_travel_uav_index(
            self.config.index_path, split=self.config.split
        )
        expected = self.config.expected_index_sha256
        if expected is not None and provenance.index_sha256 != expected:
            raise ValueError(
                "TravelUAV index SHA256 mismatch: expected "
                f"{expected}, got {provenance.index_sha256}"
            )
        cases, selection = select_travel_uav_cases(
            population,
            split=self.config.split,
            difficulties=self.config.difficulties,
            limit_per_difficulty=self.config.limit_per_difficulty,
            case_ids=self.config.case_ids,
        )
        if not cases:
            raise ValueError("TravelUAV selection produced no cases")
        readiness = self._execution_readiness(selection.selected_map_names)
        prepared = PreparedTravelUavEvaluation(
            cases=tuple(cases),
            provenance=provenance,
            selection=selection,
            execution_readiness=readiness,
        )
        if require_execution_ready:
            self._validate_real_execution(prepared)
            if not readiness["ready"]:
                raise ValueError(
                    "TravelUAV selected maps are not execution-ready: "
                    + json.dumps(readiness, ensure_ascii=False, sort_keys=True)
                )
        return prepared

    def validate(self, *, eval_id: str | None = None) -> dict[str, Any]:
        """Write selection/readiness artifacts without model or simulator use."""

        prepared = self.prepare_evaluation(require_execution_ready=False)
        eval_dir = _create_eval_dir(self.config.output_dir, eval_id)
        paths = _write_preparation_artifacts(eval_dir, prepared)
        return {**prepared.validation_dict(), "eval_dir": str(eval_dir), "validation_paths": paths}

    def run(self, *, eval_id: str | None = None) -> dict[str, Any]:
        self._validate_review_config()
        prepared = self.prepare_evaluation(require_execution_ready=True)
        eval_dir = _create_eval_dir(self.config.output_dir, eval_id)
        preparation_paths = _write_preparation_artifacts(eval_dir, prepared)
        results: list[dict[str, Any]] = []
        abort_reason = None
        for case in prepared.cases:
            result = self.run_case(case, eval_dir, provenance=prepared.provenance)
            results.append(result)
            if self.config.abort_on_infrastructure_error and result.get("infrastructure_error"):
                abort_reason = {
                    "episode_id": case.episode_id,
                    **dict(result["infrastructure_error"]),
                }
                break
        summary = _summary(results)
        summary.update(
            {
                "batch_status": "aborted" if abort_reason else "completed",
                "abort_reason": abort_reason,
                "requested_case_count": len(prepared.cases),
                "completed_case_count": len(results),
                "dataset": prepared.provenance.to_dict(),
                "case_selection": prepared.selection.to_dict(),
                "execution_readiness": prepared.execution_readiness,
            }
        )
        _write_json(eval_dir / "summary.json", summary)
        _write_json(
            eval_dir / "manifest.json",
            {
                "benchmark": "travel_uav",
                "protocol_variant": TRAVEL_UAV_PROTOCOL,
                "transfer_evaluation": True,
                "official_assistant_protocol": False,
                "front_view_only": True,
                "dataset": prepared.provenance.to_dict(),
                "case_selection": prepared.selection.to_dict(),
                "execution_readiness": prepared.execution_readiness,
                "scene_map": dict(self.config.scene_map),
                "airsim_control_strategies": {
                    package.map_id: package.airsim.get("control_strategy")
                    for package in self.map_packages
                    if package.map_id in set(self.config.scene_map.values())
                },
                "observation_profile": self.observation_profile.to_dict(),
                "upstream_reference_commit": self.config.upstream_reference_commit,
                "batch_status": summary["batch_status"],
                "abort_reason": abort_reason,
            },
        )
        review_paths: dict[str, str] = {}
        if self.config.generate_html_review:
            review_paths = BenchmarkReviewRenderer().write(
                eval_dir,
                BenchmarkSummaryReview(
                    benchmark_name="TravelUAV Transfer Evaluation",
                    protocol_variant=TRAVEL_UAV_PROTOCOL,
                    metrics={
                        **dict(summary.get("metrics") or {}),
                        "split": self.config.split,
                        "index_sha256": prepared.provenance.index_sha256,
                        "batch_status": summary["batch_status"],
                        "requested_case_count": len(prepared.cases),
                        "completed_case_count": len(results),
                        "control_strategies": ", ".join(
                            sorted(
                                {
                                    str(package.airsim["control_strategy"])
                                    for package in self.map_packages
                                    if package.map_id
                                    in set(self.config.scene_map.values())
                                }
                            )
                        ),
                        **{
                            f"population_{key}": value
                            for key, value in prepared.selection.population_counts.items()
                        },
                        **{
                            f"selected_{key}": value
                            for key, value in prepared.selection.selected_counts.items()
                        },
                    },
                    cases=tuple(
                        {
                            "case_id": result["episode_id"],
                            "status": result["status"],
                            "metrics": result.get("metrics", {}),
                        }
                        for result in results
                    ),
                    metadata={"run_status": summary["batch_status"]},
                ),
                tuple(
                    self._case_reviews[result["episode_id"]]
                    for result in results
                    if result["episode_id"] in self._case_reviews
                ),
            )
        return {
            "eval_dir": str(eval_dir),
            "summary": summary,
            "results": results,
            "summary_paths": {
                "summary_json": str(eval_dir / "summary.json"),
                "manifest": str(eval_dir / "manifest.json"),
                **preparation_paths,
                **review_paths,
            },
        }

    def run_case(
        self,
        case: TravelUavCase,
        eval_dir: str | Path,
        *,
        provenance: TravelUavDatasetProvenance | None = None,
    ) -> dict[str, Any]:
        case_dir = Path(eval_dir) / "cases" / case.episode_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "evaluation_truth.json", case.evaluation_truth())
        run_dir = RunStore(self.config.run_root).create_run_dir(
            f"travel_uav_{case.episode_id}"
        )
        recorder = AsyncRunLogger(run_dir)
        execution = None
        setup_error = None
        try:
            runtime = self._make_runtime(run_dir, recorder)
            episode = TravelUavCaseAdapter(
                scene_map=self.config.scene_map,
                observation_profile=self.observation_profile,
            ).prepare_case(case)
            execution = BenchmarkEpisodeExecutor().execute(
                episode,
                runtime=runtime,
                scenario_controller=self._scenario_controller(),
                metric_suite=TravelUavMetricSuite(),
                run_dir=run_dir,
            )
        except Exception as exc:
            setup_error = {"error_type": type(exc).__name__, "error": str(exc)}
            recorder.event("travel_uav_case_failed", setup_error)
        finally:
            recorder.close()

        audit_cursor = execution.audit_cursor if execution else EpisodeAuditCursor.freeze(run_dir)
        audit = BenchmarkAuditReader().read(audit_cursor)
        frame_assets = FrameAssetResolver().materialize(audit, case_dir)
        audit_path = write_audit_json(case_dir / "audit.json", audit, frame_assets)
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
            trace = {"samples": [], "action_count": 0, "coordinate_frames": []}
            capabilities = None
            metric_error = None
        else:
            outcome = execution.outcome.to_dict()
            metrics = dict(execution.metrics)
            trace = execution.trace.to_dict()
            capabilities = execution.scene_capabilities
            metric_error = execution.metric_error
            if metric_error and outcome["status"] == "ok":
                outcome["status"] = "error"
                outcome["error"] = metric_error
        infrastructure_error = _infrastructure_error(
            execution=execution,
            setup_error=setup_error,
            capabilities=capabilities,
            cleanup_errors=outcome.get("cleanup_errors", []),
        )
        trajectory = {
            "coordinate_frame": "airsim_ned_m",
            "samples": metrics.get("trajectory_ned", []),
            "uav_action_steps": metrics.get("uav_action_steps", trace["action_count"]),
        }
        if infrastructure_error and (outcome.get("error") or {}).get("error_type") == "SceneRecoveryRequired":
            metrics = {key: value for key, value in metrics.items()
                       if key in {"trajectory_ned", "reference_path_ned", "uav_action_steps"}}
            outcome["final_score_available"] = False
        case_result = {
            "episode_id": case.episode_id,
            "sequence_id": case.sequence_id,
            "map_name": case.map_name,
            "dataset_split": case.split,
            "difficulty": case.difficulty,
            "reference_path_length_m": case.reference_path_length_m,
            "protocol_variant": TRAVEL_UAV_PROTOCOL,
            "run_dir": str(run_dir),
            **outcome,
            "metrics": metrics,
            "metric_error": metric_error,
            "infrastructure_error": infrastructure_error,
            "scene_capabilities": capabilities,
            "control_strategy": (
                capabilities.get("control_mode")
                if capabilities is not None
                else None
            ),
            "observation_profile": self.observation_profile.to_dict(),
            "audit_path": audit_path,
        }
        _write_json(case_dir / "case_result.json", case_result)
        _write_json(case_dir / "trajectory.json", trajectory)
        _write_json(case_dir / "metrics.json", metrics)
        self._case_reviews[case.episode_id] = BenchmarkCaseReview(
            case_id=case.episode_id,
            benchmark_name="TravelUAV Transfer Evaluation",
            protocol_variant=TRAVEL_UAV_PROTOCOL,
            instruction=case.instruction,
            outcome={key: outcome.get(key) for key in outcome},
            metrics={
                key: value
                for key, value in metrics.items()
                if key not in {"trajectory_ned", "reference_path_ned"}
            }
            | {
                "difficulty": case.difficulty,
                "reference_path_length_m": case.reference_path_length_m,
            },
            audit=audit,
            frame_assets=frame_assets,
            sections=(
                ReviewSection(
                    "Dataset",
                    "key_value",
                    {
                        "split": case.split,
                        "map_name": case.map_name,
                        "difficulty": case.difficulty,
                        "reference_path_length_m": case.reference_path_length_m,
                        "index_sha256": provenance.index_sha256 if provenance else None,
                    },
                ),
                ReviewSection(
                    "Trajectory",
                    "key_value",
                    {
                        "coordinate_frame": "airsim_ned_m",
                        "executed_samples": len(metrics.get("trajectory_ned", [])),
                        "reference_samples": len(metrics.get("reference_path_ned", [])),
                        "path_length": metrics.get("path_length"),
                    },
                ),
                ReviewSection(
                    "AirSim Control",
                    "key_value",
                    {
                        "control_strategy": (
                            capabilities.get("control_mode")
                            if capabilities is not None
                            else None
                        ),
                        "requires_takeoff": (
                            capabilities.get("requires_takeoff")
                            if capabilities is not None
                            else None
                        ),
                    },
                ),
                ReviewSection(
                    "Protocol Notes",
                    "text",
                    "FlyAgent native actions with one front RGB/depth camera and no TravelUAV L1/L2/L3 assistant. This is a transfer evaluation, not the official five-view leaderboard protocol.",
                ),
            ),
            metadata={"sequence_id": case.sequence_id, "map_name": case.map_name},
        )
        return {**case_result, "case_dir": str(case_dir)}

    def _make_runtime(self, run_dir: Path, recorder: AsyncRunLogger) -> TaskRuntime:
        model_config = self._model_config()
        planner = (
            self._planner_factory(model_config)
            if self._planner_factory
            else ModelActionPlanner(model_config)
        )
        server_mode = SimulationExecutionMode.parse(self.config.execution_mode)
        server = self._simulation_server_factory(
            self.map_packages,
            port_policy=self.port_policy,
            run_dir=run_dir,
            execution_mode=server_mode,
            show_display=not self.config.headless,
            action_config=self.action_config,
            save_artifacts=self.config.save_artifacts,
        )
        return TaskRuntime(
            SimulatorEnvClient(server, observation_profile=self.observation_profile),
            recorder,
            planner=planner,
            safety=SafetySupervisor(self.safety_config),
            shutdown_policy=self.config.shutdown_policy,
            action_config=self.action_config,
            stabilization_config=UavStabilizationConfig(
                initialization_min_settle_sec=self.config.initialization_min_settle_sec
            ),
            runtime_loop_guard_windows=self.config.runtime_loop_guard_windows,
            enable_action_planner_debug=self.config.enable_action_planner_debug,
            model_config=model_config,
            read_only_execution_config=self.read_only_execution_config,
        )

    def _scenario_controller(self) -> Any:
        if self._scenario_controller_factory:
            return self._scenario_controller_factory()
        return TravelUavScenarioController(cleanup_strategy="process_exit")

    def _model_config(self) -> Any:
        if self._model_config_value is None:
            self._model_config_value = load_model_config(self.config.env_path)
        return self._model_config_value

    def _execution_readiness(self, map_names: tuple[str, ...]) -> dict[str, Any]:
        missing_scene_map = []
        missing_map_packages = []
        mismatched_packages = []
        ready = []
        for map_name in map_names:
            map_id = self.config.scene_map.get(map_name)
            if map_id is None:
                missing_scene_map.append(map_name)
                continue
            package = self.map_packages_by_id.get(map_id)
            if package is None:
                missing_map_packages.append({"map_name": map_name, "map_id": map_id})
                continue
            if str(package.scene_id) != map_name:
                mismatched_packages.append(
                    {
                        "map_name": map_name,
                        "map_id": map_id,
                        "map_package_scene_id": str(package.scene_id),
                    }
                )
                continue
            ready.append(map_name)
        missing = [name for name in map_names if name not in ready]
        return {
            "ready": not missing,
            "selected_map_names": list(map_names),
            "ready_map_names": ready,
            "missing_maps": missing,
            "missing_scene_map": missing_scene_map,
            "missing_map_packages": missing_map_packages,
            "mismatched_map_packages": mismatched_packages,
        }

    def _validate_real_execution(self, prepared: PreparedTravelUavEvaluation) -> None:
        if self.config.execution_mode != "managed":
            return
        if self.config.expected_index_sha256 is None:
            raise ValueError(
                "Real TravelUAV evaluation requires expected_index_sha256; "
                "run --validate-only first"
            )
        if prepared.provenance.index_sha256 != self.config.expected_index_sha256:
            raise ValueError("TravelUAV index SHA256 mismatch")

    def _validate_review_config(self) -> None:
        if (
            self.config.execution_mode == "managed"
            and self.config.generate_html_review
            and (not self.config.save_artifacts or not self.config.enable_action_planner_debug)
        ):
            raise ValueError(
                "Real TravelUAV HTML review requires save_artifacts=true and "
                "enable_action_planner_debug=true"
            )



def _action_config(config: TravelUavEvalConfig) -> UavActionConfig:
    values = load_uav_action_config().to_dict()
    values.update(config.action_config_overrides)
    return UavActionConfig.from_dict(values)


def _safety_config(config: TravelUavEvalConfig) -> UavSafetyConfig:
    values = load_uav_safety_config().to_dict()
    values.update(config.safety_overrides)
    return UavSafetyConfig.from_dict(values)


def _create_eval_dir(root: str | Path, eval_id: str | None) -> Path:
    name = eval_id or datetime.now(timezone.utc).strftime("travel_uav_%Y%m%dT%H%M%SZ")
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_preparation_artifacts(
    eval_dir: Path, prepared: PreparedTravelUavEvaluation
) -> dict[str, str]:
    paths = {
        "dataset_provenance": eval_dir / "dataset_provenance.json",
        "case_selection": eval_dir / "case_selection.json",
    }
    _write_json(paths["dataset_provenance"], prepared.provenance.to_dict())
    selection = prepared.selection.to_dict()
    selection["execution_readiness"] = prepared.execution_readiness
    _write_json(paths["case_selection"], selection)
    return {key: str(value) for key, value in paths.items()}


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [result.get("metrics") or {} for result in results]
    aggregate: dict[str, Any] = {"case_count": len(results)}
    for key in ("navigation_error", "success", "oracle_success", "spl", "path_length"):
        values = [float(item[key]) for item in metrics if item.get(key) is not None]
        aggregate[key] = sum(values) / len(values) if values else None
    return {
        "case_count": len(results),
        "status_counts": {
            status: sum(result.get("status") == status for result in results)
            for status in ("ok", "stopped", "error")
        },
        "metrics": aggregate,
    }


def _infrastructure_error(
    *,
    execution: Any,
    setup_error: dict[str, str] | None,
    capabilities: dict[str, Any] | None,
    cleanup_errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if execution is None:
        return setup_error or {"error_type": "SetupError", "error": "case setup failed"}
    runtime_error = getattr(execution.outcome, "error", None)
    if runtime_error and runtime_error.get("error_type") == "SceneRecoveryRequired":
        return dict(runtime_error)
    if capabilities is None:
        scene_error = getattr(execution.outcome, "error", None)
        if scene_error:
            return dict(scene_error)
        return {"error_type": "SceneCapabilityError", "error": "scene capabilities unavailable"}
    if cleanup_errors:
        return {"error_type": "CleanupError", "error": "scene cleanup failed", "details": cleanup_errors}
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
