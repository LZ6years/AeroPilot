"""Run AerialVLN episodes through configured UAV-Agent AirSim primitives."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

from evaluation.benchmarks.aerial_vln.adapter import AerialVlnCaseAdapter
from evaluation.benchmarks.aerial_vln.cases import (
    AerialVlnCase,
    AerialVlnDatasetProvenance,
    load_aerial_vln_dataset,
)
from evaluation.benchmarks.aerial_vln.config import (
    AerialVlnEvalConfig,
    aerial_vln_split_label,
)
from evaluation.benchmarks.aerial_vln.metrics import AerialVlnMetricSuite
from evaluation.benchmarks.aerial_vln.protocol import (
    AERIAL_VLN_PARSER_PROTOCOL_CONTEXT,
    AERIAL_VLN_PROTOCOL,
)
from evaluation.benchmarks.aerial_vln.selection import (
    AerialVlnCaseSelection,
    AerialVlnDifficultyAssignment,
    classify_and_select_aerial_vln_cases,
)
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
from plugins.research.stmr_map.context_modules.map import (
    PluginContextModule as StmrMapContextModule,
)
from plugins.research.stmr_map.perception import StmrPerceptionSession
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.env_client import (
    ObservationProfile,
    SimulatorEnvClient,
    load_observation_profile,
)
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.modules.image import DefaultImageContextModule
from src.memory.context.builder import ModelContextBuilder
from src.memory.context.contracts import TaskContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryRecordReader
from src.runtime.configs import load_map_config
from src.agent.config import load_model_config
from src.runtime.configs.stabilization_config import (
    UavStabilizationConfig,
)
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.factory import RuntimeFactory
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.server import SimulationServer
from src.uav.config import load_uav_action_config, load_uav_safety_config


@dataclass(frozen=True)
class PreparedAerialVlnEvaluation:
    """Dataset classification plus simulator-readiness diagnostics."""

    cases: tuple[AerialVlnCase, ...]
    provenance: AerialVlnDatasetProvenance
    selection: AerialVlnCaseSelection
    execution_readiness: dict[str, Any]

    def validation_dict(self) -> dict[str, Any]:
        difficulty_assignment, case_selection = (
            _dataset_bound_selection_artifacts(self)
        )
        return {
            "dataset_provenance": self.provenance.to_dict(),
            "difficulty_assignment": difficulty_assignment,
            "case_selection": case_selection,
            "execution_readiness": dict(self.execution_readiness),
        }


class AerialVlnBenchmarkRunner:
    """AerialVLN benchmark composition over the shared executor."""

    def __init__(
        self,
        config: AerialVlnEvalConfig,
        *,
        simulation_server_factory: Any = None,
        stmr_session: Any | None = None,
        read_only_execution_config=None,
    ) -> None:
        self.config = config
        self.read_only_execution_config = read_only_execution_config
        self._simulation_server_factory = simulation_server_factory or SimulationServer
        # Dataset-only validation must not read model credentials or construct
        # model clients. Runtime creation resolves this lazily.
        self._loaded_model_config: Any | None = None
        self.action_config = _action_config(config)
        self.safety_config = _safety_config(config)
        self.runtime_factory = RuntimeFactory()
        self.plugin_registry = self.runtime_factory.plugin_registry
        self._stmr_session = stmr_session
        self._stmr_warmed = False
        self._stmr_warmup_error: dict[str, str] | None = None
        self._stmr_modules: dict[str, StmrMapContextModule] = {}
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
        missing_modalities = sorted(
            {"rgb", "depth", "pose"}.difference(
                self.observation_profile.modalities
            )
        )
        if missing_modalities:
            raise ValueError(
                "AerialVLN requires observation modalities rgb, depth, pose; "
                "missing " + ", ".join(missing_modalities)
            )
        if (
            self.config.stmr_mode != "off"
            and not self.config.save_artifacts
        ):
            raise ValueError(
                "Real AerialVLN STMR requires save_artifacts=true so RGB "
                "and depth payloads remain available to the map builder"
            )

    def load_cases(self) -> list[AerialVlnCase]:
        """Return the selected, execution-ready evaluation cases."""

        return list(
            self.prepare_evaluation(
                require_execution_ready=True
            ).cases
        )

    def prepare_evaluation(
        self, *, require_execution_ready: bool
    ) -> PreparedAerialVlnEvaluation:
        """Load the full split, classify it, and inspect scene routing."""

        population, provenance = load_aerial_vln_dataset(
            self.config.dataset_root,
            self.config.split,
            dataset_family=self.config.dataset_family,
            dataset_version=self.config.dataset_version,
        )
        expected_sha256 = self.config.expected_split_sha256
        if (
            expected_sha256 is not None
            and provenance.split_sha256 != expected_sha256
        ):
            raise ValueError(
                "AerialVLN split SHA256 mismatch: expected "
                f"{expected_sha256}, got {provenance.split_sha256}"
            )

        # Legacy AerialVLN configs retain their historical filters. AerialVLN-S
        # config validation forbids both so global thresholds always see the
        # complete population of exactly one selected split.
        selected_population = population
        if self.config.scene_filter:
            allowed = set(self.config.scene_filter)
            selected_population = [
                case
                for case in selected_population
                if case.scene_id in allowed
            ]
        if self.config.max_cases is not None:
            selected_population = selected_population[
                : max(0, self.config.max_cases)
            ]

        cases, selection = classify_and_select_aerial_vln_cases(
            selected_population,
            difficulties=self.config.difficulties,
            limit_per_difficulty=(
                self.config.limit_per_difficulty
            ),
        )
        if self.config.episode_ids is not None:
            by_id = {case.episode_id: case for case in cases}
            missing = set(self.config.episode_ids) - by_id.keys()
            if missing:
                raise ValueError(f"episode_ids unavailable after split classification: {sorted(missing)}")
            cases = [by_id[key] for key in self.config.episode_ids]
            selection = replace(selection, selected_assignments=tuple(
                selection.assignment_by_episode[key] for key in self.config.episode_ids
            ))
        if not cases:
            raise ValueError(
                "AerialVLN difficulty selection produced no cases"
            )
        readiness = self._execution_readiness(selection.selected_scene_ids)
        prepared = PreparedAerialVlnEvaluation(
            cases=tuple(cases),
            provenance=provenance,
            selection=selection,
            execution_readiness=readiness,
        )
        if require_execution_ready:
            self._validate_dataset_for_execution(prepared)
            if not readiness["ready"]:
                raise ValueError(
                    "AerialVLN selected scenes are not execution-ready: "
                    + json.dumps(
                        readiness,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
        return prepared

    def _model_config(self) -> Any:
        if self._loaded_model_config is None:
            self._loaded_model_config = load_model_config(
                self.config.env_path
            )
        return self._loaded_model_config

    def validate(self, *, eval_id: str | None = None) -> dict[str, Any]:
        """Persist offline classification without constructing Runtime."""

        prepared = self.prepare_evaluation(
            require_execution_ready=False
        )
        eval_dir = _create_eval_dir(self.config.output_dir, eval_id)
        paths = _write_preparation_artifacts(eval_dir, prepared)
        return {
            **prepared.validation_dict(),
            "eval_dir": str(eval_dir),
            "validation_paths": paths,
        }

    def run(self, *, eval_id: str | None = None) -> dict[str, Any]:
        self._validate_review_config()
        prepared = self.prepare_evaluation(
            require_execution_ready=True
        )
        self._prepare_stmr()
        split_label = aerial_vln_split_label(self.config.split)
        eval_dir = _create_eval_dir(self.config.output_dir, eval_id)
        preparation_paths = _write_preparation_artifacts(
            eval_dir, prepared
        )
        assignments = prepared.selection.assignment_by_episode
        results: list[dict[str, Any]] = []
        abort_reason: dict[str, Any] | None = None
        for case in prepared.cases:
            result = self.run_case(
                case,
                eval_dir,
                assignment=assignments[case.episode_id],
                provenance=prepared.provenance,
            )
            results.append(result)
            if (
                self.config.abort_on_infrastructure_error
                and result.get("infrastructure_error")
            ):
                abort_reason = {
                    "episode_id": case.episode_id,
                    **dict(result["infrastructure_error"]),
                }
                break
        summary = _summary(results)
        summary.update(
            {
                "batch_status": (
                    "aborted" if abort_reason is not None else "completed"
                ),
                "abort_reason": abort_reason,
                "requested_case_count": len(prepared.cases),
                "completed_case_count": len(results),
                "dataset": prepared.provenance.to_dict(),
                "dataset_split_label": split_label,
                "difficulty_selection": (
                    prepared.selection.to_dict()
                ),
                "execution_readiness": (
                    prepared.execution_readiness
                ),
            }
        )
        _write_json(eval_dir / "summary.json", summary)
        (eval_dir / "summary.md").write_text(
            _summary_markdown(summary), encoding="utf-8"
        )
        manifest = {
            "benchmark": "aerial_vln",
            "protocol_variant": AERIAL_VLN_PROTOCOL,
            "dataset_split": self.config.split,
            "dataset_split_label": split_label,
            "dataset_version": self.config.dataset_version,
            "dataset": prepared.provenance.to_dict(),
            "difficulty_selection": prepared.selection.to_dict(),
            "execution_readiness": prepared.execution_readiness,
            "upstream_reference_commit": (
                self.config.upstream_reference_commit
            ),
            "scene_map": dict(self.config.scene_map),
            "airsim_control_strategies": {
                package.map_id: package.airsim.get("control_strategy")
                for package in self.map_packages
                if package.map_id in set(self.config.scene_map.values())
            },
            "observation_profile": self.observation_profile.to_dict(),
            "runtime_loop_guard_windows": (
                self.config.runtime_loop_guard_windows
            ),
            "initialization_min_settle_sec": (
                self.config.initialization_min_settle_sec
            ),
            "stmr": _stmr_config_summary(self.config),
            "batch_status": summary["batch_status"],
            "abort_reason": summary["abort_reason"],
            "requested_case_count": summary["requested_case_count"],
            "completed_case_count": summary["completed_case_count"],
            "case_count": len(results),
        }
        _write_json(eval_dir / "manifest.json", manifest)
        review_paths: dict[str, str] = {}
        if self.config.generate_html_review and self.config.report_profile == "audit":
            review_paths = BenchmarkReviewRenderer().write(
                eval_dir,
                BenchmarkSummaryReview(
                    benchmark_name="AerialVLN",
                    protocol_variant=AERIAL_VLN_PROTOCOL,
                    metrics={
                        **dict(summary.get("metrics") or {}),
                        "dataset_family": (
                            prepared.provenance.dataset_family
                        ),
                        "dataset_split": prepared.provenance.split,
                        "dataset_split_label": split_label,
                        "split_sha256": (
                            prepared.provenance.split_sha256
                        ),
                        "difficulty_q33_m": prepared.selection.q33_m,
                        "difficulty_q67_m": prepared.selection.q67_m,
                        "population_scene_count": len(
                            prepared.selection.population_scene_ids
                        ),
                        "selected_scene_count": len(
                            prepared.selection.selected_scene_ids
                        ),
                        "ready_scene_count": len(
                            prepared.execution_readiness[
                                "ready_scene_ids"
                            ]
                        ),
                        "missing_scenes": ", ".join(
                            prepared.execution_readiness[
                                "missing_scenes"
                            ]
                        )
                        or "none",
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
                        "batch_status": summary["batch_status"],
                        "requested_case_count": summary[
                            "requested_case_count"
                        ],
                        "completed_case_count": summary[
                            "completed_case_count"
                        ],
                        **{
                            f"population_{label}": count
                            for label, count in (
                                prepared.selection.bucket_counts.items()
                            )
                        },
                        **{
                            f"selected_{label}": count
                            for label, count in (
                                prepared.selection.selected_counts.items()
                            )
                        },
                    },
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
                **preparation_paths,
                **review_paths,
            },
        }

    def run_case(
        self,
        case: AerialVlnCase,
        eval_dir: str | Path,
        *,
        assignment: AerialVlnDifficultyAssignment | None = None,
        provenance: AerialVlnDatasetProvenance | None = None,
    ) -> dict[str, Any]:
        case_dir = Path(eval_dir) / "cases" / case.episode_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "evaluation_truth.json", case.evaluation_truth())
        run_dir = RunStore(self.config.run_root).create_run_dir(
            f"aerial_vln_{case.episode_id}"
        )
        recorder = AsyncRunLogger(run_dir, capture_provider_debug=self.config.report_profile == "audit")
        execution = None
        setup_error = None
        stmr_module: StmrMapContextModule | None = None
        stmr_postprocess_error: dict[str, str] | None = None
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

        if self.config.stmr_mode == "planner":
            stmr_module = self._stmr_modules.pop(str(run_dir), None)
        elif self.config.stmr_mode == "shadow":
            stmr_module, stmr_postprocess_error = self._build_shadow_stmr(
                run_dir,
                instruction=case.instruction,
            )
        stmr_summary = _stmr_summary(
            stmr_module,
            mode=self.config.stmr_mode,
            warmup_error=self._stmr_warmup_error,
            postprocess_error=stmr_postprocess_error,
        )
        stmr_assets = _materialize_stmr_assets(
            case_dir,
            stmr_summary,
        )

        audit_cursor = (
            execution.audit_cursor
            if execution is not None
            else EpisodeAuditCursor.freeze(run_dir)
        )
        audit = BenchmarkAuditReader().read(audit_cursor)
        if self.config.report_profile == "audit":
            frame_assets = FrameAssetResolver().materialize(audit, case_dir)
            audit_path = write_audit_json(case_dir / "audit.json", audit, frame_assets)
        else:
            frame_assets = ()
            _write_json(case_dir / "audit_cursor.json", audit_cursor.to_dict())
            audit_path = str(case_dir / "audit_cursor.json")

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

        infrastructure_error = _infrastructure_error(
            execution=execution,
            setup_error=setup_error,
            outcome=outcome,
            capabilities=capabilities,
        )

        trajectory = {
            "coordinate_frame": "airsim_ned_m",
            "samples": metrics.get("trajectory_ned", []),
            "uav_action_steps": metrics.get(
                "uav_action_steps", trace["action_count"]
            ),
        }
        if infrastructure_error:
            metrics = {key: value for key, value in metrics.items()
                       if key in {"trajectory_ned", "reference_path_ned", "uav_action_steps"}}
            outcome["final_score_available"] = False
        case_result = {
            "episode_id": case.episode_id,
            "trajectory_id": case.trajectory_id,
            "scene_id": case.scene_id,
            "dataset_split": self.config.split,
            "dataset_split_label": aerial_vln_split_label(
                self.config.split
            ),
            "difficulty": (
                assignment.difficulty if assignment is not None else None
            ),
            "reference_path_length_m": (
                assignment.reference_path_length_m
                if assignment is not None
                else None
            ),
            "protocol_variant": AERIAL_VLN_PROTOCOL,
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
            "stmr": stmr_summary,
        }
        task_plan_diagnostics = _task_plan_diagnostics(
            run_dir, outcome
        )
        case_result["task_plan_diagnostics"] = (
            task_plan_diagnostics
        )
        _write_json(case_dir / "case_result.json", case_result)
        _write_json(case_dir / "trajectory.json", trajectory)
        _write_json(case_dir / "trace.json", trace)
        _write_json(case_dir / "metrics.json", metrics)
        if self.config.report_profile != "audit" and self.config.generate_html_review:
            from evaluation.benchmarks.core.process_review import write_process_review
            try:
                case_result["report"] = write_process_review(case_dir, case_result, case.instruction,
                                                             profile=self.config.report_profile)
            except Exception as exc:
                case_result["report_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
            _write_json(case_dir / "case_result.json", case_result)
        _write_json(case_dir / "stmr_summary.json", stmr_summary)
        _write_json(
            case_dir / "manifest.json",
            {
                "dataset_split": self.config.split,
                "dataset_split_label": aerial_vln_split_label(
                    self.config.split
                ),
                "dataset_version": self.config.dataset_version,
                "dataset": (
                    provenance.to_dict()
                    if provenance is not None
                    else None
                ),
                "difficulty": (
                    assignment.to_dict()
                    if assignment is not None
                    else None
                ),
                "upstream_reference_commit": (
                    self.config.upstream_reference_commit
                ),
                "protocol_variant": AERIAL_VLN_PROTOCOL,
                "task_plan_diagnostics": task_plan_diagnostics,
                "scene_capabilities": capabilities,
                "control_strategy": (
                    capabilities.get("control_mode")
                    if capabilities is not None
                    else None
                ),
                "observation_profile": self.observation_profile.to_dict(),
                "runtime_loop_guard_windows": (
                    self.config.runtime_loop_guard_windows
                ),
                "initialization_min_settle_sec": (
                    self.config.initialization_min_settle_sec
                ),
                "stmr": stmr_summary,
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
            }
            | {
                "difficulty": (
                    assignment.difficulty
                    if assignment is not None
                    else None
                ),
                "reference_path_length_m": (
                    assignment.reference_path_length_m
                    if assignment is not None
                    else None
                ),
            },
            audit=audit,
            frame_assets=frame_assets,
            sections=(
                ReviewSection(
                    "Dataset And Difficulty",
                    "key_value",
                    {
                        "dataset_family": (
                            provenance.dataset_family
                            if provenance is not None
                            else self.config.dataset_family
                        ),
                        "dataset_split": self.config.split,
                        "dataset_split_label": aerial_vln_split_label(
                            self.config.split
                        ),
                        "difficulty": (
                            assignment.difficulty
                            if assignment is not None
                            else None
                        ),
                        "reference_path_length_m": (
                            assignment.reference_path_length_m
                            if assignment is not None
                            else None
                        ),
                    },
                ),
                ReviewSection(
                    "Trajectory",
                    "trajectory_ned",
                    {
                        "actual": metrics.get("trajectory_ned", []),
                        "reference": metrics.get("reference_path_ned", []),
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
                    "Task Plan Diagnostics",
                    "key_value",
                    task_plan_diagnostics,
                ),
                ReviewSection(
                    "STMR Spatial Memory",
                    "key_value",
                    _compact_stmr_summary(stmr_summary),
                ),
                *(
                    (
                        ReviewSection(
                            "STMR Map",
                            "image_group",
                            stmr_assets,
                        ),
                    )
                    if stmr_assets
                    else ()
                ),
                ReviewSection(
                    "Protocol Notes",
                    "text",
                    (
                        "AerialVLN data and metrics with UAV-Agent native "
                        "tools under the declared AirSim control strategy; "
                        "coordinates are reported in "
                        "AirSim NED meters. Landing language denotes reaching "
                        "the requested target region and ending the episode; "
                        "it does not require physical ground contact."
                    ),
                ),
            ),
            metadata={
                "trajectory_id": case.trajectory_id,
                "scene_id": case.scene_id,
                "dataset_split": self.config.split,
                "dataset_split_label": aerial_vln_split_label(
                    self.config.split
                ),
                "difficulty": (
                    assignment.difficulty
                    if assignment is not None
                    else None
                ),
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
        planner: Any | None = None,
    ) -> TaskRuntime:
        model_config = self._model_config()
        active_planner = planner or ModelActionPlanner(model_config)
        context_builder, stmr_module = self._planner_context_builder(run_dir)
        if stmr_module is not None:
            self._stmr_modules[str(run_dir)] = stmr_module
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
        return self.runtime_factory.create_task_runtime(
            SimulatorEnvClient(
                server, observation_profile=self.observation_profile
            ),
            recorder,
            planner=active_planner,
            safety=SafetySupervisor(self.safety_config),
            shutdown_policy=self.config.shutdown_policy,
            stabilization_config=UavStabilizationConfig(
                initialization_min_settle_sec=(
                    self.config.initialization_min_settle_sec
                )
            ),
            runtime_loop_guard_windows=self.config.runtime_loop_guard_windows,
            context_builder=context_builder,
            model_config=model_config,
            read_only_execution_config=self.read_only_execution_config,
        )

    def _planner_context_builder(
        self,
        run_dir: Path,
    ) -> tuple[
        ModelContextBuilder | None,
        StmrMapContextModule | None,
    ]:
        """Install STMR only when it is allowed to affect model context."""

        if self.config.stmr_mode != "planner":
            return None, None
        default_registry = self.runtime_factory.context_registry(run_dir)
        modules = dict(default_registry.modules)
        image_module = DefaultImageContextModule()
        stmr_module = StmrMapContextModule(
            mode="planner",
            session=self._shared_stmr_session(),
            prior=self.config.stmr_prior,
            max_queries=self.config.stmr_max_queries,
            device=self.config.stmr_device,
        )
        modules["image"] = image_module
        modules["map"] = stmr_module
        module_order = list(default_registry.module_order)
        registry = ContextModuleRegistry(
            modules,
            disabled=default_registry.disabled,
            module_order=module_order,
            history_policies=default_registry.history_policies,
        )
        return (
            ModelContextBuilder(
                safety_config=self.safety_config,
                registry=registry,
                knowledge_catalog=[
                    {"name": skill.name, "description": skill.description}
                    for skill in self.plugin_registry.skill_catalog.skills
                    if skill.capability is None
                ],
            ),
            stmr_module,
        )

    def _build_shadow_stmr(
        self,
        run_dir: Path,
        *,
        instruction: str,
    ) -> tuple[StmrMapContextModule, dict[str, str] | None]:
        """Build report-only STMR after Runtime logging has closed.

        Keeping shadow outside model-context assembly guarantees that neither task
        parsing nor Planner requests change while the spatial map is evaluated.
        """

        module = StmrMapContextModule(
            mode="shadow",
            session=self._shared_stmr_session(),
            prior=self.config.stmr_prior,
            max_queries=self.config.stmr_max_queries,
            device=self.config.stmr_device,
        )
        memory = FlightMemory(
            run_id=run_dir.name,
            map_id="aerial_vln_shadow",
            scene_instance_id="completed_episode",
            agent_id="uav",
            active_instruction=instruction,
            memory_reader=MemoryRecordReader(run_dir),
        )
        try:
            module.online_builder.build(
                memory,
                TaskContextInput(active_instruction=instruction),
            )
        except Exception as exc:
            # debug_summary preserves the last successful state when possible;
            # report assembly remains fail-open by design.
            return module, {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        return module, None

    def _shared_stmr_session(self) -> Any:
        if self._stmr_session is None:
            self._stmr_session = StmrPerceptionSession(
                device=self.config.stmr_device
            )
        return self._stmr_session

    def _prepare_stmr(self) -> None:
        if self.config.stmr_mode == "off" or self._stmr_warmed:
            return
        session = self._shared_stmr_session()
        warmup = getattr(session, "warmup", None)
        try:
            if callable(warmup):
                warmup()
        except Exception as exc:
            self._stmr_warmup_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            self._stmr_warmed = True

    def _execution_readiness(
        self, selected_scene_ids: tuple[str, ...]
    ) -> dict[str, Any]:
        missing_scene_map = []
        missing_map_packages = []
        mismatched_map_packages = []
        ready_scene_ids = []
        for scene_id in selected_scene_ids:
            map_id = self.config.scene_map.get(scene_id)
            if map_id is None:
                missing_scene_map.append(scene_id)
                continue
            package = self.map_packages_by_id.get(map_id)
            if package is None:
                missing_map_packages.append(
                    {"scene_id": scene_id, "map_id": map_id}
                )
                continue
            if str(package.scene_id) != str(scene_id):
                mismatched_map_packages.append(
                    {
                        "scene_id": scene_id,
                        "map_id": map_id,
                        "map_package_scene_id": str(package.scene_id),
                    }
                )
                continue
            ready_scene_ids.append(scene_id)
        missing_scenes = [
            scene_id
            for scene_id in selected_scene_ids
            if scene_id not in ready_scene_ids
        ]
        return {
            "ready": not missing_scenes,
            "selected_scene_ids": list(selected_scene_ids),
            "ready_scene_ids": ready_scene_ids,
            "missing_scenes": missing_scenes,
            "missing_scene_map": missing_scene_map,
            "missing_map_packages": missing_map_packages,
            "mismatched_map_packages": mismatched_map_packages,
        }

    def _validate_dataset_for_execution(
        self, prepared: PreparedAerialVlnEvaluation
    ) -> None:
        if self.config.dataset_family != "aerialvln-s":
            return
        if self.config.execution_mode in {"attach", "managed"}:
            dataset_version = self.config.dataset_version.strip()
            if (
                not dataset_version
                or dataset_version.lower() == "unspecified"
            ):
                raise ValueError(
                    "Real AerialVLN-S evaluation requires dataset_version"
                )
            if self.config.expected_split_sha256 is None:
                raise ValueError(
                    "Real AerialVLN-S evaluation requires "
                    "expected_split_sha256; run --validate-only first"
                )
        if (
            self.config.expected_split_sha256 is not None
            and prepared.provenance.split_sha256
            != self.config.expected_split_sha256
        ):
            # Kept here as a defensive invariant even though preparation checks
            # before selection as well.
            raise ValueError("AerialVLN-S split SHA256 mismatch")

    def _validate_review_config(self) -> None:
        """Require persistent evidence for real, portable HTML reviews."""

        if (
            not self.config.generate_html_review
        ):
            return
        missing = []
        if not self.config.save_artifacts:
            missing.append("save_artifacts=true")
        if self.config.report_profile == "audit" and not self.config.enable_action_planner_debug:
            missing.append("enable_action_planner_debug=true")
        if missing:
            raise ValueError(
                "Real AerialVLN HTML review requires "
                + " and ".join(missing)
            )


def _stmr_config_summary(config: AerialVlnEvalConfig) -> dict[str, Any]:
    return {
        "mode": config.stmr_mode,
        "prior": config.stmr_prior,
        "max_queries": config.stmr_max_queries,
        "device": config.stmr_device,
    }


def _stmr_summary(
    module: StmrMapContextModule | None,
    *,
    mode: str,
    warmup_error: dict[str, str] | None,
    postprocess_error: dict[str, str] | None,
) -> dict[str, Any]:
    """Return human-facing STMR diagnostics without rebuilding the map."""

    if module is None:
        summary: dict[str, Any] = {
            "mode": mode,
            "status": "disabled" if mode == "off" else "unavailable",
        }
    else:
        try:
            summary = {"mode": mode, **module.debug_summary()}
            summary["latency"] = _stmr_latency_summary(
                summary.get("profiling")
            )
        except Exception as exc:
            summary = {
                "mode": mode,
                "status": "summary_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    if warmup_error is not None:
        summary["warmup_error"] = dict(warmup_error)
    if postprocess_error is not None:
        summary["postprocess_error"] = dict(postprocess_error)
        if summary.get("status") in {None, "waiting_for_complete_observation"}:
            summary["status"] = "postprocess_failed"
    return summary


def _compact_stmr_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep the case page readable while full debug stays in JSON."""

    keys = (
        "mode",
        "status",
        "stale",
        "processed_observation_count",
        "attempted_observation_count",
        "failed_observation_count",
        "point_count",
        "voxel_count",
        "latency",
        "png_path",
        "json_path",
        "warnings",
        "warmup_error",
        "postprocess_error",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def _stmr_latency_summary(profiling: Any) -> dict[str, Any]:
    if not isinstance(profiling, dict):
        return {}
    stages = profiling.get("stages")
    if not isinstance(stages, dict):
        return {}
    frame = stages.get("stmr_online_frame_total")
    total = stages.get("stmr_online_perception_map_total")
    frame = frame if isinstance(frame, dict) else {}
    total = total if isinstance(total, dict) else {}
    return {
        "frame_sample_count": frame.get("sample_count"),
        "frame_mean_ms": frame.get("mean_ms"),
        "frame_p95_ms": frame.get("p95_ms"),
        "total_ms": total.get("total_ms"),
    }


def _materialize_stmr_assets(
    case_dir: Path,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Copy the latest report-only STMR image beside portable case HTML."""

    raw_path = summary.get("png_path")
    source = Path(str(raw_path)) if raw_path else None
    if source is None or not source.is_file():
        return []
    destination = case_dir / "review_assets" / "stmr" / "latest_stmr_map.png"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    except OSError as exc:
        summary.setdefault("asset_errors", []).append(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source": str(source),
            }
        )
        return []
    return [
        {
            "path": str(destination),
            "label": "Latest STMR semantic history",
            "description": (
                "UAV-centered accumulated semantics and actual trajectory; "
                "blank cells are unknown, not verified free space."
            ),
        }
    ]



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


def _write_preparation_artifacts(
    eval_dir: Path,
    prepared: PreparedAerialVlnEvaluation,
) -> dict[str, str]:
    difficulty_assignment, case_selection = (
        _dataset_bound_selection_artifacts(prepared)
    )
    artifacts = {
        "dataset_provenance": (
            eval_dir / "dataset_provenance.json",
            prepared.provenance.to_dict(),
        ),
        "difficulty_assignment": (
            eval_dir / "difficulty_assignment.json",
            difficulty_assignment,
        ),
        "case_selection": (
            eval_dir / "case_selection.json",
            case_selection,
        ),
        "execution_readiness": (
            eval_dir / "execution_readiness.json",
            prepared.execution_readiness,
        ),
    }
    for _, (path, payload) in artifacts.items():
        _write_json(path, payload)
    return {
        name: str(path)
        for name, (path, _) in artifacts.items()
    }


def _dataset_bound_selection_artifacts(
    prepared: PreparedAerialVlnEvaluation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind split identity to evaluation-only classification artifacts."""

    dataset_identity = {
        "dataset": prepared.provenance.to_dict(),
        "dataset_split_label": aerial_vln_split_label(
            prepared.provenance.split
        ),
    }
    return (
        {
            **dataset_identity,
            **prepared.selection.assignments_dict(),
        },
        {
            **dataset_identity,
            **prepared.selection.to_dict(),
        },
    )


def _infrastructure_error(
    *,
    execution: Any | None,
    setup_error: dict[str, Any] | None,
    outcome: dict[str, Any],
    capabilities: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Identify failures that make continuing a remote batch unsafe.

    Planner and metric failures can occur after a valid simulator scene is
    available and remain benchmark results. Exhausted provider transport
    failures are different: later episodes would use the same unhealthy model
    endpoint, so the batch must stop even though the simulator is still valid.
    Failures before capabilities are established, and all cleanup failures,
    likewise make continuing unsafe.
    """

    cleanup_errors = list(outcome.get("cleanup_errors") or [])
    if cleanup_errors:
        return {"stage": "cleanup", "errors": cleanup_errors}
    outcome_error = outcome.get("error")
    if isinstance(outcome_error, dict) and outcome_error.get("error_type") == "SceneRecoveryRequired":
        return {"stage": "scene_control", "error": dict(outcome_error)}
    if (
        isinstance(outcome_error, dict)
        and outcome_error.get("error_type") == "ProviderRequestError"
    ):
        return {"stage": "provider", "error": dict(outcome_error)}
    if execution is None:
        return {
            "stage": "case_setup",
            "error": setup_error or outcome.get("error"),
        }
    if outcome.get("error") and capabilities is None:
        return {
            "stage": "scene_setup",
            "error": outcome.get("error"),
        }
    return None


def _task_plan_diagnostics(
    run_dir: Path, outcome: dict[str, Any]
) -> dict[str, Any]:
    """Report optional task planning without inventing missing phases."""
    latest_state = _read_json_object(run_dir / "state" / "latest_state.json")
    plan = latest_state.get("task_plan")
    return {
        "loop_schema": "unified_tool_loop_v2",
        "plan_created": isinstance(plan, dict),
        "task_plan": plan,
        "stopped_reason": outcome.get("stopped_reason"),
        "stop_source": outcome.get("stop_source"),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [
        item["metrics"]
        for item in results
        if isinstance(item.get("metrics"), dict) and item["metrics"]
        and item.get("final_score_available", not bool(item.get("infrastructure_error")))
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
    # Failed episodes can still carry partial metrics (for example nDTW=0
    # while distance_to_goal is unavailable because no pose was recorded).
    # Aggregate each metric over its own valid observations so reporting a
    # genuine failure never masks the case result with a summary exception.
    aggregates = {key: _mean_available_metric(metric_rows, key) for key in keys}
    return {
        "benchmark": "aerial_vln",
        "protocol_variant": AERIAL_VLN_PROTOCOL,
        "case_count": len(results),
        "evaluated_count": len(metric_rows),
        "difficulty_counts": {
            label: sum(
                item.get("difficulty") == label for item in results
            )
            for label in ("low", "medium", "high")
        },
        "metrics": aggregates,
        "cases": [
            {
                "episode_id": item.get("episode_id"),
                "scene_id": item.get("scene_id"),
                "difficulty": item.get("difficulty"),
                "reference_path_length_m": item.get(
                    "reference_path_length_m"
                ),
                "status": item.get("status"),
                "metrics": item.get("metrics"),
            }
            for item in results
        ],
    }


def _mean_available_metric(
    metric_rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    values: list[float] = []
    for row in metric_rows:
        value = row.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return sum(values) / len(values) if values else None


def _summary_markdown(summary: dict[str, Any]) -> str:
    dataset = summary.get("dataset") or {}
    selection = summary.get("difficulty_selection") or {}
    thresholds = selection.get("thresholds_m") or {}
    readiness = summary.get("execution_readiness") or {}
    split_label = summary.get("dataset_split_label") or dataset.get(
        "split", "unknown"
    )
    lines = [
        "# AerialVLN Native-Action Summary",
        "",
        f"- Protocol: `{summary['protocol_variant']}`",
        "- Dataset: "
        f"`{dataset.get('dataset_family', 'unknown')} / "
        f"{dataset.get('split', 'unknown')} / {split_label}`",
        f"- Split SHA256: `{dataset.get('split_sha256', 'unknown')}`",
        "- Difficulty thresholds (m): "
        f"q33={thresholds.get('q33')}, q67={thresholds.get('q67')}",
        "- Scenes: "
        f"population={len(dataset.get('scene_ids') or [])}, "
        f"selected={len(selection.get('selected_scene_ids') or [])}, "
        f"missing={readiness.get('missing_scenes') or []}",
        f"- Cases: {summary['case_count']}",
        f"- Batch status: {summary.get('batch_status', 'completed')}",
        "- Requested/completed: "
        f"{summary.get('requested_case_count', summary['case_count'])}/"
        f"{summary.get('completed_case_count', summary['case_count'])}",
        f"- Abort reason: {summary.get('abort_reason')}",
        f"- Evaluated: {summary['evaluated_count']}",
        "- Difficulty cases: "
        + ", ".join(
            f"{label}={count}"
            for label, count in summary["difficulty_counts"].items()
        ),
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
