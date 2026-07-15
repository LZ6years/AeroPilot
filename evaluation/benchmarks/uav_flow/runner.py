"""Run UAV-Agent native runtime against UAV-Flow benchmark cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase, load_cases
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.html_review import write_html_review
from evaluation.benchmarks.uav_flow.metrics import evaluate_case_ndtw
from evaluation.benchmarks.uav_flow.reports import (
    create_eval_dir,
    run_artifact_summary,
    write_case_outputs,
    write_case_review,
    write_eval_summary,
    write_trajectory_plots,
)
from evaluation.benchmarks.uav_flow.sr_judge import judge_success
from evaluation.benchmarks.uav_flow.trajectory import export_trajectory, target_local_from_case
from evaluation.benchmarks.uav_flow.unrealcv_scenario import UavFlowBenchmarkEnvClient
from evaluation.benchmarks.uav_flow.visual_diagnostics import write_visual_diagnostics
from plugins.research.stmr_map.context_modules.map import PluginContextModule as StmrMapContextModule
from plugins.research.stmr_map.perception import StmrPerceptionSession
from src.agent.safety import SafetySupervisor
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.semantic.task_parser import TaskParser
from src.memory.context.registry import ContextModuleRegistry
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.runtime.configs.action_config import load_uav_action_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.configs.safety_config import load_uav_safety_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime


class UavFlowBenchmarkRunner:
    """Outer benchmark harness; TaskRuntime remains the owner of task logic."""

    def __init__(self, config: UavFlowEvalConfig, *, stmr_session: Any | None = None) -> None:
        self.config = config
        self.model_config = load_model_config(config.env_path)
        self.action_config = _benchmark_action_config(config)
        self.safety_config = _benchmark_safety_config(config)
        self.plugin_registry = PluginRegistry.from_config()
        self._stmr_session = stmr_session
        self._stmr_warmed = False

    def load_cases(self) -> list[UavFlowCase]:
        cases = load_cases(
            self.config.uav_flow_repo_path,
            classes=self.config.classes,
            limit_per_class=self.config.limit_per_class,
            instruction_field=self.config.instruction_field,
            test_jsons_path=self.config.test_jsons_path,
            classified_instr_path=self.config.classified_instr_path,
        )
        if self.config.max_cases is not None:
            cases = cases[: max(0, int(self.config.max_cases))]
        return cases

    def run(self, *, eval_id: str | None = None) -> dict[str, Any]:
        self._prepare_stmr()
        eval_dir = create_eval_dir(self.config.output_dir, eval_id)
        console_lines = [f"uav_flow_eval: output={eval_dir}"]
        results = []
        cases = self.load_cases()
        shared_env_client: UavFlowBenchmarkEnvClient | None = None
        reuse_real_env = bool(self.config.real_unrealcv and self.config.reuse_real_unrealcv_env)
        try:
            for case in cases:
                result, shared_env_client = self._run_case(
                    case,
                    eval_dir,
                    env_client=shared_env_client if reuse_real_env else None,
                    reuse_scene=reuse_real_env,
                )
                results.append(result)
                console_lines.append(
                    f"{case.case_id}: status={result.get('status')} ndtw={result.get('ndtw', {}).get('ndtw')}"
                )
                if reuse_real_env and _shared_env_failure(result):
                    console_lines.append(f"{case.case_id}: stopping batch after shared UnrealCV env failure")
                    break
        finally:
            if shared_env_client is not None:
                shared_env_client.close_real_env()
        summary_paths = write_eval_summary(eval_dir, results, console_lines)
        html_paths = _write_html_review_from_summary(eval_dir, summary_paths)
        summary_paths.update(html_paths)
        return {"eval_dir": str(eval_dir), "summary_paths": summary_paths, "results": results}

    def run_case(self, case: UavFlowCase, eval_dir: str | Path) -> dict[str, Any]:
        self._prepare_stmr()
        result, _ = self._run_case(case, eval_dir)
        return result

    def _run_case(
        self,
        case: UavFlowCase,
        eval_dir: str | Path,
        *,
        env_client: UavFlowBenchmarkEnvClient | None = None,
        reuse_scene: bool = False,
    ) -> tuple[dict[str, Any], UavFlowBenchmarkEnvClient]:
        case_dir = Path(eval_dir) / "cases" / case.case_id
        run_dir = RunStore(self.config.run_root).create_run_dir(f"uav_flow_{case.case_id}")
        recorder = AsyncRunLogger(run_dir)
        needs_prepare = env_client is not None
        if env_client is None:
            env_client = self._make_env_client(case, run_dir)
        status = "ok"
        error: dict[str, Any] | None = None
        stmr_module: StmrMapContextModule | None = None
        try:
            if needs_prepare:
                env_client.prepare_case(case, run_dir, scene_id=case.case_id)
            recorder.write_json(
                "benchmark/uav_flow_case_planner_safe.json",
                case.planner_safe_metadata(),
            )
            recorder.write_json(
                "benchmark/uav_flow_case_evaluation_truth.json",
                case.evaluation_truth(),
            )
            planner = ModelActionPlanner(self.model_config)
            context_builder, stmr_module = self._context_builder(planner)
            runtime = TaskRuntime(
                env_client,
                recorder,
                TaskParser(mode=self.config.parser_mode, model_config=self.model_config),
                planner=planner,
                safety=SafetySupervisor(self.safety_config),
                shutdown_policy=self.config.shutdown_policy,
                action_config=self.action_config,
                runtime_loop_guard_windows=self.config.runtime_loop_guard_windows,
                enable_action_planner_debug=self.config.enable_action_planner_debug,
                context_builder=context_builder,
            )
            state = runtime.run_task(
                "uav_flow_downtownwest",
                case.selected_instruction,
                agent_id="uav_flow_drone",
                scene_policy="reuse_running" if reuse_scene else "isolated",
                stop_scene_on_finish=not reuse_scene,
                scene_id=case.case_id,
            )
            if reuse_scene and self.config.shutdown_policy != "none":
                runtime.safe_hover_reused_scene(
                    "uav_flow_downtownwest",
                    agent_id="uav_flow_drone",
                    scene_id=case.case_id,
                )
            completed = state.completed
            stopped = state.stopped
            stopped_reason = state.stopped_reason
            stop_source = state.stop_source
            if stopped and status == "ok":
                status = "stopped"
        except Exception as exc:
            status = "error"
            completed = False
            stopped = False
            stopped_reason = None
            stop_source = None
            error = {"error_type": type(exc).__name__, "error": str(exc)}
            recorder.event("uav_flow_case_failed", error)
        finally:
            recorder.close()

        stmr_summary = _stmr_summary(stmr_module)
        stmr_summary["mode"] = self.config.stmr_mode
        trajectory_meta = export_trajectory(case, env_client.pose_history, case_dir / "uav_agent_trajectory.json")
        run_artifacts = run_artifact_summary(run_dir, self.action_config)
        ndtw_result = evaluate_case_ndtw(
            trajectory_meta["path"],
            case.reference_path_preprocessed,
            class_name=case.class_name,
        )
        plot_result = write_trajectory_plots(case_dir, trajectory_meta["path"], case)
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
            "status": status,
            "completed": completed,
            "stopped": stopped,
            "stopped_reason": stopped_reason,
            "stop_source": stop_source,
            "error": error,
            "run_artifacts": run_artifacts,
            "trajectory": trajectory_meta,
            "visual_diagnostics": visual_diagnostics,
            "plot_result": plot_result,
            "ndtw": ndtw_result,
            "sr_judge": sr_result,
            "stmr": stmr_summary,
        }
        review_path = write_case_review(case_dir, case_payload)
        write_case_outputs(
            case_dir,
            {
                "case.json": case_payload,
                "ndtw.json": ndtw_result,
                "sr_judge.json": sr_result,
                "plot_result.json": plot_result,
                "visual_diagnostics_summary.json": visual_diagnostics,
                "stmr_summary.json": stmr_summary,
            },
        )
        return {
            "case_id": case.case_id,
            "class_name": case.class_name,
            "status": status,
            "completed": completed,
            "stopped": stopped,
            "stopped_reason": stopped_reason,
            "stop_source": stop_source,
            "run_dir": str(run_dir),
            "case_dir": str(case_dir),
            "review_path": review_path,
            "run_artifacts": run_artifacts,
            "visual_diagnostics": visual_diagnostics,
            "plot_result": plot_result,
            "ndtw": ndtw_result,
            "sr_judge": sr_result,
            "stmr": stmr_summary,
            "error": error,
        }, env_client

    def _context_builder(
        self,
        planner: Any,
    ) -> tuple[ObservationContextBuilder | None, StmrMapContextModule | None]:
        """Build a case-local experimental map provider without global config changes."""

        if self.config.stmr_mode == "off":
            return None, None
        stmr_module = StmrMapContextModule(
            mode=self.config.stmr_mode,
            session=self._shared_stmr_session(),
            prior=self.config.stmr_prior,
            max_queries=self.config.stmr_max_queries,
            device=self.config.stmr_device,
        )
        default_registry = ContextModuleRegistry.default(
            plugin_registry=self.plugin_registry,
            enable_model_skill_selection=getattr(planner, "planner_mode", "") == "model",
        )
        modules = dict(default_registry.modules)
        modules["map"] = stmr_module
        registry = ContextModuleRegistry(
            modules,
            disabled=default_registry.disabled,
            module_order=default_registry.module_order,
        )
        return (
            ObservationContextBuilder(
                action_config=self.action_config,
                safety_config=self.safety_config,
                registry=registry,
                plugin_registry=self.plugin_registry,
                enable_model_skill_selection=getattr(planner, "planner_mode", "") == "model",
            ),
            stmr_module,
        )

    def _shared_stmr_session(self) -> Any:
        if self._stmr_session is None:
            self._stmr_session = StmrPerceptionSession(device=self.config.stmr_device)
        return self._stmr_session

    def _prepare_stmr(self) -> None:
        """Load optional perception models before launching a real UE process."""

        if self.config.stmr_mode == "off" or self._stmr_warmed:
            return
        session = self._shared_stmr_session()
        warmup = getattr(session, "warmup", None)
        if callable(warmup):
            warmup()
        self._stmr_warmed = True

    def _make_env_client(self, case: UavFlowCase, run_dir: str | Path) -> UavFlowBenchmarkEnvClient:
        return UavFlowBenchmarkEnvClient(
            case,
            run_dir=run_dir,
            env_id=self.config.env_id,
            uav_flow_repo_path=self.config.uav_flow_repo_path,
            real_unrealcv=self.config.real_unrealcv,
            time_dilation=self.config.time_dilation,
            resolution=self.config.resolution,
            seed=self.config.seed,
            action_config=self.action_config,
            # The default UAV-Flow adapter intentionally captures RGB/pose
            # only.  STMR explicitly opts into its existing raw UnrealCV depth
            # path without changing baseline benchmark behavior.
            enable_research_frame_capture=self.config.stmr_mode != "off",
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


def _shared_env_failure(result: dict[str, Any]) -> bool:
    error = result.get("error") or {}
    if not isinstance(error, dict):
        return False
    text = f"{error.get('error_type', '')} {error.get('error', '')}".lower()
    return any(term in text for term in ("unrealcv", "scene initialization", "object placement", "prepare_case"))


def _stmr_summary(module: StmrMapContextModule | None) -> dict[str, Any]:
    """Keep evaluation reporting fail-open when optional parity diagnostics fail."""

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


def _write_html_review_from_summary(eval_dir: str | Path, summary_paths: dict[str, str]) -> dict[str, str]:
    """Generate static HTML review pages from the just-written summary."""

    import json

    summary_path = Path(summary_paths["summary_json"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    html_paths = write_html_review(eval_dir, summary)
    html_review = html_paths.get("html_review")
    if html_review:
        summary_md = Path(summary_paths["summary_md"])
        console_txt = Path(summary_paths["console_txt"])
        summary_md.write_text(
            summary_md.read_text(encoding="utf-8") + f"\n## HTML Review\n\n- `{html_review}`\n",
            encoding="utf-8",
        )
        console_txt.write_text(
            console_txt.read_text(encoding="utf-8") + f"html_review: {html_review}\n",
            encoding="utf-8",
        )
    return html_paths
