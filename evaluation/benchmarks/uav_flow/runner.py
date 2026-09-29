"""Run UAV-Flow cases through SimulationServer and the shared executor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmarks.core import (
    BenchmarkAuditLog,
    BenchmarkAuditReader,
    BenchmarkCaseReview,
    BenchmarkEpisodeExecutor,
    BenchmarkReviewRenderer,
    EpisodeAuditCursor,
    FrameAssetResolver,
    ResolvedFrameAsset,
    write_audit_json,
)
from evaluation.benchmarks.core.report_io import atomic_write_text
from evaluation.benchmarks.uav_flow.adapter import (
    UAV_FLOW_PROTOCOL,
    UavFlowCaseAdapter,
    validate_uav_flow_profile,
)
from evaluation.benchmarks.uav_flow.cases import UavFlowCase, load_cases
from evaluation.benchmarks.uav_flow.campaign import write_run_manifest
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.metric_suite import UavFlowMetricSuite
from evaluation.benchmarks.uav_flow.postprocess import (
    render_evaluation_report,
)
from evaluation.benchmarks.uav_flow.reports import (
    create_eval_dir,
    write_case_outputs,
    write_eval_summary,
    write_trajectory_plots,
)
from evaluation.benchmarks.uav_flow.scenario import (
    UavFlowScenarioController,
)
from evaluation.benchmarks.uav_flow.scene_session import (
    UavFlowBatchSceneSession,
)
from evaluation.benchmarks.uav_flow.review import (
    build_uav_flow_review_sections,
)
from evaluation.benchmarks.uav_flow.sr_judge import judge_success
from evaluation.benchmarks.uav_flow.trajectory import target_local_from_case
from evaluation.benchmarks.uav_flow.visual_diagnostics import (
    write_visual_diagnostics,
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
from src.memory.context.modules import DefaultImageContextModule
from src.memory.context.builder import ModelContextBuilder
from src.memory.context.registry import ContextModuleRegistry
from src.runtime.configs import load_map_config
from src.agent.config import load_model_config
from src.runtime.context import ModelMediaStore
from src.runtime.factory import RuntimeFactory
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime
from src.uav.config import load_uav_action_config, load_uav_safety_config
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.server import SimulationServer


class UavFlowBenchmarkRunner:
    """Outer benchmark harness; TaskRuntime remains the owner of task logic."""

    def __init__(
        self,
        config: UavFlowEvalConfig,
        *,
        simulation_server_factory: Any = None,
        scenario_controller_factory: Any = None,
        stmr_session: Any | None = None,
    ) -> None:
        self._scenario_controller_factory = scenario_controller_factory or UavFlowScenarioController
        self.config = config
        self._simulation_server_factory = simulation_server_factory or SimulationServer
        self.model_config = load_model_config(config.env_path)
        self.action_config = _benchmark_action_config(config)
        self.safety_config = _benchmark_safety_config(config)
        self.runtime_factory = RuntimeFactory()
        self.plugin_registry = self.runtime_factory.plugin_registry
        self._stmr_session = stmr_session
        self._stmr_warmed = False
        self._case_reviews: dict[str, BenchmarkCaseReview] = {}
        self._report_errors: list[dict[str, Any]] = []
        self._batch_session: UavFlowBatchSceneSession | None = None
        self._batch_eval_dir: Path | None = None
        self._validate_review_config()
        self.map_packages, self.port_policy = load_map_config(
            config.map_config_path
        )
        self.map_package = next(
            (
                package
                for package in self.map_packages
                if package.map_id == config.map_id
            ),
            None,
        )
        if self.map_package is None:
            raise ValueError(
                f"UAV-Flow map_id {config.map_id!r} is not present in "
                f"{config.map_config_path}"
            )
        if self.map_package.control_protocol != "unrealcv":
            raise ValueError("UAV-Flow requires an UnrealCV MapPackage")
        self.observation_profile = self._load_profile()
        validate_uav_flow_profile(
            self.observation_profile, stmr_mode=config.stmr_mode
        )

    def load_cases(self) -> list[UavFlowCase]:
        cases = load_cases(
            self.config.uav_flow_repo_path,
            case_ids=self.config.case_ids,
            classes=self.config.classes,
            limit_per_class=self.config.limit_per_class,
            instruction_field=self.config.instruction_field,
            test_jsons_path=self.config.test_jsons_path,
            classified_instr_path=self.config.classified_instr_path,
        )
        if (
            self.config.case_ids is None
            and self.config.max_cases is not None
        ):
            cases = cases[: max(0, int(self.config.max_cases))]
        return cases

    def run(self, *, eval_id: str | None = None) -> dict[str, Any]:
        # A runner instance may be reused by a manual harness.  Report state is
        # evaluation-local and must never leak case pages or errors across IDs.
        self._case_reviews.clear()
        self._report_errors.clear()
        eval_dir = create_eval_dir(self.config.output_dir, eval_id)
        console_lines = [f"uav_flow_eval: output={eval_dir}"]
        cases = self.load_cases()
        # Freeze the selected roster and effective, redacted configuration
        # before any simulator side effect. Continuation tooling treats this
        # manifest as the reproducibility boundary between evaluation segments.
        write_run_manifest(
            eval_dir,
            config=self.config,
            model_config=self.model_config,
            cases=cases,
        )
        self._prepare_stmr()
        results: list[dict[str, Any]] = []
        summary_paths: dict[str, str] = {}
        active_case_id: str | None = None
        self._batch_eval_dir = eval_dir
        try:
            try:
                for case in cases:
                    active_case_id = case.case_id
                    result = self.run_case(case, eval_dir)
                    results.append(result)
                    active_case_id = None
                    console_lines.append(_result_console_line(result))
                    summary_paths = self._write_report_snapshot(
                        eval_dir,
                        results,
                        console_lines,
                        run_status="running",
                        expected_case_count=len(cases),
                        incomplete_case_ids=[],
                    )
            except BaseException as exc:
                interrupted = isinstance(exc, KeyboardInterrupt)
                summary_paths = self._write_report_snapshot(
                    eval_dir,
                    results,
                    console_lines,
                    run_status="interrupted" if interrupted else "failed",
                    expected_case_count=len(cases),
                    incomplete_case_ids=(
                        [active_case_id] if active_case_id is not None else []
                    ),
                )
                raise
            summary_paths = self._write_report_snapshot(
                eval_dir,
                results,
                console_lines,
                run_status="completed",
                expected_case_count=len(cases),
                incomplete_case_ids=[],
            )
            return {
                "eval_dir": str(eval_dir),
                "summary_paths": summary_paths,
                "results": results,
            }
        finally:
            # A batch owns one UE scene.  This outer finally is the only normal
            # process-stop boundary, including Ctrl+C and report failures.
            try:
                self._close_batch_session()
            finally:
                self._batch_eval_dir = None

    def _write_report_snapshot(
        self,
        eval_dir: Path,
        results: list[dict[str, Any]],
        console_lines: list[str],
        *,
        run_status: str,
        expected_case_count: int,
        incomplete_case_ids: list[str],
    ) -> dict[str, str]:
        """Refresh progress without allowing report I/O to alter case status."""

        metadata = {
            "run_status": run_status,
            "expected_case_count": expected_case_count,
            "incomplete_case_ids": list(incomplete_case_ids),
            "excluded_case_ids": [],
            "report_generated_from": "live_runner",
        }
        try:
            if self.config.generate_html_review:
                paths = render_evaluation_report(
                    eval_dir,
                    results,
                    (
                        self._case_reviews[item["case_id"]]
                        for item in results
                        if item.get("case_id") in self._case_reviews
                    ),
                    console_lines,
                    report_metadata=metadata,
                    render_cases=False,
                )
            else:
                paths = write_eval_summary(
                    eval_dir,
                    results,
                    console_lines,
                    report_metadata=metadata,
                )
        except Exception as exc:
            self._report_errors.append(
                {
                    "stage": "evaluation_report_snapshot",
                    "run_status": run_status,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            paths = {}
        self._write_live_report_manifest(
            eval_dir,
            metadata=metadata,
            finalized_case_count=len(results),
            summary_paths=paths,
        )
        return paths

    def _write_live_report_manifest(
        self,
        eval_dir: Path,
        *,
        metadata: dict[str, Any],
        finalized_case_count: int,
        summary_paths: dict[str, str],
    ) -> None:
        """Persist report errors separately from benchmark outcome data."""

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
            "finalized_case_count": finalized_case_count,
            "report_errors": list(self._report_errors),
            "summary_paths": dict(summary_paths),
        }
        try:
            atomic_write_text(
                eval_dir / "report_manifest.json",
                json.dumps(payload, indent=2, ensure_ascii=False),
            )
        except Exception:
            # There is no safer report channel left at this point.  The case
            # evidence remains authoritative and can be recovered offline.
            pass

    def run_case(
        self, case: UavFlowCase, eval_dir: str | Path
    ) -> dict[str, Any]:
        """Run one case using the active batch scene or a temporary session."""

        session, owns_session = self._scene_session_for_case(Path(eval_dir))
        try:
            return self._run_case_with_session(case, eval_dir, session)
        finally:
            if owns_session:
                session.close()

    def _run_case_with_session(
        self,
        case: UavFlowCase,
        eval_dir: str | Path,
        session: UavFlowBatchSceneSession,
    ) -> dict[str, Any]:
        self._prepare_stmr()
        case_dir = Path(eval_dir) / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_evaluation_truth(case_dir, case)
        episode = UavFlowCaseAdapter(
            map_id=self.config.map_id,
            scene_id=self.map_package.scene_id,
            agent_id="uav_flow_drone",
            observation_profile=self.observation_profile,
        ).prepare_case(case)
        recovery_attempts: list[dict[str, Any]] = []
        attempt_number = 1
        episode_prepared = False
        execution = None
        setup_error: dict[str, str] | None = None
        stmr_module: StmrMapContextModule | None = None
        scenario = self._scenario_controller_factory()

        while True:
            run_dir = RunStore(self.config.run_root).create_run_dir(
                f"uav_flow_{case.case_id}_attempt{attempt_number}"
            )
            recorder = AsyncRunLogger(run_dir)
            scenario = self._scenario_controller_factory()
            metric_suite = UavFlowMetricSuite(
                case,
                case_dir,
                action_config=self.action_config,
            )
            setup_error = None
            execution = None
            try:
                if not episode_prepared:
                    # Place the UAV before TaskParser invokes a potentially slow
                    # model. Runtime repeats and audits the same placement.
                    session.prepare_episode(episode)
                episode_prepared = False
                planner = ModelActionPlanner(self.model_config)
                context_builder, stmr_module = self._context_builder(run_dir)
                runtime = self._make_runtime(
                    run_dir,
                    recorder,
                    planner=planner,
                    context_builder=context_builder,
                    env_client=session.client,
                )
                execution = BenchmarkEpisodeExecutor().execute(
                    episode,
                    runtime=session.runtime_view(runtime),
                    scenario_controller=scenario,
                    metric_suite=metric_suite,
                    run_dir=run_dir,
                )
            except Exception as exc:
                recorder.event(
                    "uav_flow_case_failed",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
                setup_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            finally:
                recorder.close()

            attempt_error = (
                setup_error
                if execution is None
                else execution.outcome.error
            )
            if not _is_recoverable_simulator_error(attempt_error):
                break
            if attempt_number >= 3:
                break

            recovery_action = (
                "resynchronize" if attempt_number == 1 else "restart"
            )
            recovery_record = {
                "attempt": attempt_number,
                "run_dir": str(run_dir),
                "error": dict(attempt_error or {}),
                "recovery_action": recovery_action,
            }
            try:
                if recovery_action == "restart":
                    session.restart_scene(
                        reason=str((attempt_error or {}).get("error", ""))
                    )
                session.prepare_episode(episode)
                episode_prepared = True
                recovery_record["status"] = "completed"
                recovery_attempts.append(recovery_record)
                attempt_number += 1
                continue
            except Exception as recovery_exc:
                recovery_record.update(
                    {
                        "status": "failed",
                        "recovery_error": {
                            "error_type": type(recovery_exc).__name__,
                            "error": str(recovery_exc),
                        },
                    }
                )
                recovery_attempts.append(recovery_record)
                if recovery_action == "resynchronize":
                    # A failed in-place probe skips directly to the one allowed
                    # process restart rather than wasting another model attempt.
                    try:
                        session.restart_scene(reason=str(recovery_exc))
                        session.prepare_episode(episode)
                        episode_prepared = True
                        recovery_attempts.append(
                            {
                                "attempt": attempt_number,
                                "run_dir": str(run_dir),
                                "recovery_action": "restart",
                                "status": "completed",
                            }
                        )
                        attempt_number = 3
                        continue
                    except Exception as restart_exc:
                        setup_error = {
                            "error_type": type(restart_exc).__name__,
                            "error": str(restart_exc),
                        }
                break

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
            trace_payload: dict[str, Any] = {
                "trace_schema_version": None,
                "trace_integrity": "episode_setup_failed",
                "trace_issues": ["episode_setup_failed"],
                "samples": [],
                "action_count": 0,
                "physical_call_count": 0,
                "physical_window_count": 0,
                "motion_call_count": 0,
                "physical_windows": [],
                "coordinate_frames": [],
            }
            ndtw_result = {
                "status": "skipped",
                "reason": "episode_setup_failed",
                "ndtw": None,
            }
            trajectory_meta = {
                "path": str(case_dir / "uav_agent_trajectory.json"),
                "point_count": 0,
                "unit": "cm",
            }
            turn_diagnostics = None
            backend_action_tracking = {
                "schema_version": "uav_flow_window_tracking_v2",
                "association_source": "episode_setup_failed",
                "trace_valid": False,
                "tracking_valid": False,
                "physical_call_count": 0,
                "motion_call_count": 0,
                "physical_window_count": 0,
                "tracked_window_count": 0,
                "observed_window_count": 0,
                "unavailable_window_count": 0,
                "zero_motion_completion": False,
                "windows": [],
            }
            metric_error = None
            scene_capabilities = None
        else:
            outcome = execution.outcome.to_dict()
            trace_payload = execution.trace.to_dict()
            ndtw_result = execution.metrics.get(
                "ndtw",
                {
                    "status": "skipped",
                    "reason": "metric_failed",
                    "ndtw": None,
                },
            )
            trajectory_meta = execution.metrics.get(
                "trajectory",
                {
                    "path": str(case_dir / "uav_agent_trajectory.json"),
                    "point_count": 0,
                    "unit": "cm",
                },
            )
            turn_diagnostics = execution.metrics.get("turn_diagnostics")
            backend_action_tracking = execution.metrics.get(
                "backend_action_tracking",
                {
                    "schema_version": "uav_flow_window_tracking_v2",
                    "association_source": "metric_failed",
                    "trace_valid": trace_payload.get("trace_integrity") == "ok",
                    "tracking_valid": False,
                    "physical_call_count": trace_payload.get("physical_call_count", 0),
                    "motion_call_count": trace_payload.get("motion_call_count", 0),
                    "physical_window_count": trace_payload.get("physical_window_count", 0),
                    "tracked_window_count": 0,
                    "observed_window_count": 0,
                    "unavailable_window_count": trace_payload.get("physical_window_count", 0),
                    "zero_motion_completion": (
                        bool(outcome.get("completed"))
                        and trace_payload.get("motion_call_count", 0) == 0
                    ),
                    "windows": [],
                },
            )
            metric_error = execution.metric_error
            scene_capabilities = execution.scene_capabilities
            if metric_error is not None and outcome["status"] == "ok":
                outcome["status"] = "error"
                outcome["error"] = metric_error

        stmr_summary = _stmr_summary(stmr_module)
        stmr_summary["mode"] = self.config.stmr_mode
        scenario_diagnostics = scenario.diagnostics_for(case.case_id)
        scenario_contract = scenario_diagnostics.setdefault("contract", {})
        actor_config = self.map_package.unrealcv.get("actor", {})
        if isinstance(actor_config, dict):
            scenario_contract.update(
                {
                    "camera_follow": actor_config.get("camera_follow"),
                    "activate_viewport": actor_config.get(
                        "activate_viewport", False
                    ),
                    "rotation_method": actor_config.get("rotation_method"),
                    "yaw_offset_deg": actor_config.get("yaw_offset_deg"),
                    "camera_validation": actor_config.get(
                        "camera_validation"
                    ),
                    "capture_camera_id": self.map_package.unrealcv.get(
                        "camera_id"
                    ),
                }
            )
            if actor_config.get("activate_viewport", False):
                scenario_contract.update(
                    {
                        "viewport_actor": actor_config.get("name"),
                        "viewport_activation": "prepare",
                        "viewport_method": "set_viewport",
                    }
                )
            if actor_config.get("camera_follow") == "blueprint_pose":
                scenario_contract.update(
                    {
                        "blueprint_camera_actor": actor_config.get("name"),
                        "blueprint_camera_pose_source": (
                            "controlled_actor_absolute_pose"
                        ),
                        "blueprint_viewport_method": "set_viewport",
                    }
                )
            elif actor_config.get("camera_follow") == "standard_pose":
                scenario_contract.update(
                    {
                        "capture_camera_pose_source": (
                            "controlled_actor_absolute_pose"
                        ),
                        "capture_camera_sync_method": (
                            "standard_unrealcv_pose"
                        ),
                    }
                )
        audit_summary = _audit_summary(
            audit=audit,
            frame_assets=frame_assets,
        )
        review_html = str(case_dir / "review.html")
        plot_result = write_trajectory_plots(
            case_dir, trajectory_meta["path"], case
        )
        visual_diagnostics = write_visual_diagnostics(
            case,
            run_dir=run_dir,
            case_dir=case_dir,
            enabled=self.config.enable_visual_diagnostics,
            model_config=self.model_config,
            max_frames=self.config.visual_diagnostics_max_frames,
        )
        trajectory_summary = {
            "point_count": trajectory_meta["point_count"],
            "target_local": target_local_from_case(case),
            "run_dir": str(run_dir),
            "trajectory_path": trajectory_meta["path"],
            "plot_result": plot_result,
        }
        sr_result = judge_success(
            case,
            trajectory_summary,
            ndtw_result,
            enabled=self.config.enable_sr_judge,
            model_config=self.model_config,
        )
        case_payload = {
            "case": case.to_dict(include_evaluation_truth=True),
            "planner_safe_case": case.planner_safe_metadata(),
            "run_dir": str(run_dir),
            "review_html": review_html,
            **outcome,
            "audit_summary": audit_summary,
            "audit": {
                "path": audit_path,
                "frame_assets": [
                    item.to_dict() for item in frame_assets
                ],
            },
            "trajectory": trajectory_meta,
            "trace": trace_payload,
            "scene_capabilities": scene_capabilities,
            "observation_profile": self.observation_profile.to_dict(),
            "mask_label_map": scenario.label_map_for(case.case_id),
            "scenario_diagnostics": scenario_diagnostics,
            "visual_diagnostics": visual_diagnostics,
            "plot_result": plot_result,
            "ndtw": ndtw_result,
            "turn_diagnostics": turn_diagnostics,
            "backend_action_tracking": backend_action_tracking,
            "trace_valid": backend_action_tracking.get("trace_valid") is True,
            "tracking_valid": backend_action_tracking.get("tracking_valid") is True,
            "infrastructure_clean": (
                backend_action_tracking.get("tracking_valid") is True
                and backend_action_tracking.get("all_observed_within_tolerance")
                is not False
                and backend_action_tracking.get("failed_window_count", 0) == 0
                and backend_action_tracking.get(
                    "failed_or_skipped_step_count", 0
                )
                == 0
            ),
            "sr_judge": sr_result,
            "stmr": stmr_summary,
            "metric_error": metric_error,
            "simulator_recovery_attempts": recovery_attempts,
        }
        write_case_outputs(
            case_dir,
            {
                "case.json": case_payload,
                "trace.json": trace_payload,
                "ndtw.json": ndtw_result,
                "turn_diagnostics.json": turn_diagnostics,
                "backend_action_tracking.json": backend_action_tracking,
                "scenario_diagnostics.json": scenario_diagnostics,
                "sr_judge.json": sr_result,
                "plot_result.json": plot_result,
                "visual_diagnostics_summary.json": visual_diagnostics,
                "stmr_summary.json": stmr_summary,
                "mask_label_map.json": scenario.label_map_for(case.case_id),
            },
        )
        case_review = BenchmarkCaseReview(
            case_id=case.case_id,
            benchmark_name="UAV-Flow",
            protocol_variant=UAV_FLOW_PROTOCOL,
            instruction=case.selected_instruction,
            outcome=dict(outcome),
            metrics={
                "class_name": case.class_name,
                "ndtw": ndtw_result.get("ndtw"),
                "ndtw_status": ndtw_result.get("status"),
                "sr_status": sr_result.get("status"),
                "reference_yaw_delta_deg": (
                    turn_diagnostics or {}
                ).get("reference_yaw_delta_deg"),
                "actual_yaw_delta_deg": (
                    turn_diagnostics or {}
                ).get("actual_yaw_delta_deg"),
                "final_yaw_error_deg": (
                    turn_diagnostics or {}
                ).get("final_yaw_error_deg"),
                "translation_drift_m": (
                    turn_diagnostics or {}
                ).get("translation_drift_m"),
                "turn_tracking_max_abs_error_deg": (
                    (turn_diagnostics or {}).get("backend_turn_tracking")
                    or {}
                ).get("max_abs_error_deg"),
                "translation_tracking_max_error_m": (
                    backend_action_tracking or {}
                ).get("max_position_error_m"),
                "zero_motion_completion": (
                    backend_action_tracking or {}
                ).get("zero_motion_completion"),
            },
            audit=audit,
            frame_assets=frame_assets,
            sections=build_uav_flow_review_sections(
                case_dir=case_dir,
                ndtw_result=ndtw_result,
                sr_result=sr_result,
                visual_diagnostics=visual_diagnostics,
                stmr_summary=stmr_summary,
                plot_result=plot_result,
                turn_diagnostics=turn_diagnostics,
                backend_action_tracking=backend_action_tracking,
                scenario_diagnostics=scenario_diagnostics,
            ),
            metadata=case.planner_safe_metadata(),
        )
        self._case_reviews[case.case_id] = case_review
        if self.config.generate_html_review:
            try:
                # Case HTML is flushed now rather than held until every case
                # finishes.  A later Ctrl+C can therefore only leave the
                # currently active case incomplete.
                BenchmarkReviewRenderer().write_case(eval_dir, case_review)
            except Exception as exc:
                self._report_errors.append(
                    {
                        "stage": "case_html",
                        "case_id": case.case_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        return {
            "case_id": case.case_id,
            "class_name": case.class_name,
            **outcome,
            "run_dir": str(run_dir),
            "case_dir": str(case_dir),
            "review_html": review_html,
            "audit_path": audit_path,
            "audit_summary": audit_summary,
            "visual_diagnostics": visual_diagnostics,
            "plot_result": plot_result,
            "ndtw": ndtw_result,
            "turn_diagnostics": turn_diagnostics,
            "backend_action_tracking": backend_action_tracking,
            "trace_valid": backend_action_tracking.get("trace_valid") is True,
            "tracking_valid": backend_action_tracking.get("tracking_valid") is True,
            "infrastructure_clean": (
                backend_action_tracking.get("tracking_valid") is True
                and backend_action_tracking.get("all_observed_within_tolerance")
                is not False
                and backend_action_tracking.get("failed_window_count", 0) == 0
                and backend_action_tracking.get(
                    "failed_or_skipped_step_count", 0
                )
                == 0
            ),
            "scenario_diagnostics": scenario_diagnostics,
            "sr_judge": sr_result,
            "stmr": stmr_summary,
            "metric_error": metric_error,
            "simulator_recovery_attempts": recovery_attempts,
        }

    def _validate_review_config(self) -> None:
        """Fail before scene launch when a real audit cannot be complete."""

        if (
            not self.config.generate_html_review
        ):
            return
        missing = []
        if not self.config.save_artifacts:
            missing.append("save_artifacts=true")
        if not self.config.enable_action_planner_debug:
            missing.append("enable_action_planner_debug=true")
        if missing:
            raise ValueError(
                "Real UAV-Flow HTML review requires "
                + " and ".join(missing)
            )

    def _make_runtime(
        self,
        run_dir: Path,
        recorder: AsyncRunLogger,
        *,
        planner: Any,
        context_builder: ModelContextBuilder | None,
        env_client: SimulatorEnvClient | None = None,
    ) -> TaskRuntime:
        if env_client is None:
            server = self._make_simulation_server(run_dir)
            env_client = SimulatorEnvClient(
                server, observation_profile=self.observation_profile
            )
        return self.runtime_factory.create_task_runtime(
            env_client,
            recorder,
            planner=planner,
            safety=SafetySupervisor(self.safety_config),
            shutdown_policy=self.config.shutdown_policy,
            runtime_loop_guard_windows=self.config.runtime_loop_guard_windows,
            context_builder=context_builder,
            model_config=self.model_config,
        )

    def _make_simulation_server(self, run_dir: str | Path) -> SimulationServer:
        """Build the one service owned by a real batch session."""

        server_mode = SimulationExecutionMode.parse(self.config.execution_mode)
        return self._simulation_server_factory(
            self.map_packages,
            port_policy=self.port_policy,
            run_dir=run_dir,
            execution_mode=server_mode,
            show_display=not self.config.headless,
            action_config=self.action_config,
            save_artifacts=self.config.save_artifacts,
        )

    def _scene_session_for_case(
        self,
        eval_dir: Path,
    ) -> tuple[UavFlowBatchSceneSession, bool]:
        """Return the batch session, or a one-case compatible session."""

        if self._batch_eval_dir is not None:
            if self._batch_session is None:
                self._batch_session = self._new_scene_session(
                    self._batch_eval_dir
                )
            return self._batch_session, False
        return self._new_scene_session(eval_dir), True

    def _new_scene_session(
        self,
        eval_dir: Path,
    ) -> UavFlowBatchSceneSession:
        service = self._make_simulation_server(
            eval_dir / "scene_session_runtime"
        )
        client = SimulatorEnvClient(
            service,
            observation_profile=self.observation_profile,
        )
        return UavFlowBatchSceneSession(
            service,
            client,
            map_id=self.config.map_id,
            scene_id=self.map_package.scene_id,
            eval_dir=eval_dir,
        )

    def _close_batch_session(self) -> None:
        session = self._batch_session
        self._batch_session = None
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                # Do not replace a KeyboardInterrupt or the original batch
                # failure with a secondary cleanup exception. The session
                # manifest already contains the exact stop failure.
                self._report_errors.append(
                    {
                        "stage": "batch_scene_cleanup",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    def _load_profile(self) -> ObservationProfile:
        if self.config.observation_profile_path:
            return load_observation_profile(
                self.config.observation_profile_path
            )
        if self.config.stmr_mode != "off":
            return ObservationProfile.from_dict(
                {
                    "modalities": [
                        "rgb",
                        "depth",
                        "segmentation",
                        "pose",
                    ]
                }
            )
        return ObservationProfile()

    def _context_builder(
        self,
        run_dir: Path,
    ) -> tuple[
        ModelContextBuilder | None, StmrMapContextModule | None
    ]:
        include_depth = self.config.planner_depth_input == "now"
        include_stmr = self.config.stmr_mode != "off"
        if not include_depth and not include_stmr:
            return None, None
        default_registry = self.runtime_factory.context_registry(run_dir)
        modules = dict(default_registry.modules)
        if include_depth:
            modules["image"] = DefaultImageContextModule(
                include_current_depth=True,
                media_writer=ModelMediaStore(run_dir),
            )
        stmr_module: StmrMapContextModule | None = None
        if include_stmr:
            stmr_module = StmrMapContextModule(
                mode=self.config.stmr_mode,
                session=self._shared_stmr_session(),
                prior=self.config.stmr_prior,
                max_queries=self.config.stmr_max_queries,
                device=self.config.stmr_device,
            )
            modules["map"] = stmr_module
        registry = ContextModuleRegistry(
            modules,
            disabled=default_registry.disabled,
            module_order=default_registry.module_order,
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
        if callable(warmup):
            warmup()
        self._stmr_warmed = True


def _audit_summary(
    *,
    audit: BenchmarkAuditLog,
    frame_assets: tuple[ResolvedFrameAsset, ...],
) -> dict[str, int]:
    """Expose compact machine-readable counts without re-reading Runtime logs.

    Full actions, messages, and frame provenance remain owned by Benchmark
    Core's ``audit.json``. This summary is intentionally count-only so UAV-Flow
    does not maintain a second audit interpretation path.
    """

    return {
        "rgb_count": sum(
            1
            for item in frame_assets
            if item.modality == "rgb" and item.display_path
        ),
        "action_count": len(audit.actions),
        "agent_message_count": len(audit.agent_messages),
    }



def _write_evaluation_truth(case_dir: Path, case: UavFlowCase) -> None:
    path = case_dir / "evaluation_truth.json"
    path.write_text(
        json.dumps(case.evaluation_truth(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _benchmark_action_config(config: UavFlowEvalConfig) -> Any:
    loaded = load_uav_action_config()
    base = loaded.to_dict()
    base.update(config.action_config_overrides)
    return loaded.__class__.from_dict(base)


def _benchmark_safety_config(config: UavFlowEvalConfig) -> Any:
    loaded = load_uav_safety_config()
    base = loaded.to_dict()
    base.update(config.safety_overrides)
    return loaded.__class__.from_dict(base)


def _stmr_summary(
    module: StmrMapContextModule | None,
) -> dict[str, Any]:
    if module is None:
        return {"mode": "off", "status": "disabled"}
    try:
        return module.debug_summary()
    except Exception as exc:
        return {
            "status": "summary_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _result_console_line(result: dict[str, Any]) -> str:
    """Keep the historical detailed console summary for one finalized case."""

    turn = result.get("turn_diagnostics") or {}
    tracking = turn.get("backend_turn_tracking") or {}
    action_tracking = result.get("backend_action_tracking") or {}
    return (
        f"{result.get('case_id')}: status={result.get('status')} "
        f"ndtw={(result.get('ndtw') or {}).get('ndtw')} "
        f"reference_yaw_delta_deg={turn.get('reference_yaw_delta_deg')} "
        f"actual_yaw_delta_deg={turn.get('actual_yaw_delta_deg')} "
        f"final_yaw_error_deg={turn.get('final_yaw_error_deg')} "
        f"translation_drift_m={turn.get('translation_drift_m')} "
        "turn_tracking_max_abs_error_deg="
        f"{tracking.get('max_abs_error_deg')} "
        "translation_tracking_max_error_m="
        f"{action_tracking.get('max_position_error_m')} "
        "zero_motion_completion="
        f"{action_tracking.get('zero_motion_completion')}"
    )


def _is_recoverable_simulator_error(
    error: dict[str, Any] | None,
) -> bool:
    """Recognize protocol/scene failures without restarting UE for model I/O.

    BenchmarkEpisodeExecutor serializes exceptions at its boundary, so this
    classifier intentionally relies on stable simulator-owned message prefixes
    rather than broad network exception classes.  In particular, generic SSL,
    HTTP, TaskParser, Planner, and model-provider failures must not restart a
    healthy UnrealCV process.
    """

    if not error:
        return False
    text = " ".join(
        str(error.get(key, "")) for key in ("error_type", "error")
    ).lower()
    model_markers = (
        "taskparser",
        "task parser",
        "planner",
        "model provider",
        "dashscope",
        "ssl",
        "https://",
    )
    if any(marker in text for marker in model_markers):
        return False
    simulator_markers = (
        "unrealcv",
        "camera follow validation failed",
        "actor pose validation failed",
        "simulator warm-up",
        "simulator initial pose",
        "simulator batch pre-position",
        "scene endpoint",
    )
    return any(marker in text for marker in simulator_markers)
