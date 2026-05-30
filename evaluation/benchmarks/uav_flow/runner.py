"""Run UAV-Agent native runtime against UAV-Flow benchmark cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase, load_cases
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.metrics import evaluate_case_ndtw
from evaluation.benchmarks.uav_flow.reports import (
    create_eval_dir,
    evaluation_truth_leak_check,
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
from src.agent.safety import SafetySupervisor
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.semantic.task_parser import TaskParser
from src.runtime.configs.action_config import load_uav_action_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.configs.safety_config import load_uav_safety_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime


class UavFlowBenchmarkRunner:
    """Outer benchmark harness; TaskRuntime remains the owner of task logic."""

    def __init__(self, config: UavFlowEvalConfig) -> None:
        self.config = config
        self.model_config = load_model_config(config.env_path)
        self.action_config = _benchmark_action_config(config)
        self.safety_config = _benchmark_safety_config(config)

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
        eval_dir = create_eval_dir(self.config.output_dir, eval_id)
        console_lines = [f"uav_flow_eval: output={eval_dir}"]
        results = []
        for case in self.load_cases():
            result = self.run_case(case, eval_dir)
            results.append(result)
            console_lines.append(
                f"{case.case_id}: status={result.get('status')} ndtw={result.get('ndtw', {}).get('ndtw')}"
            )
        summary_paths = write_eval_summary(eval_dir, results, console_lines)
        return {"eval_dir": str(eval_dir), "summary_paths": summary_paths, "results": results}

    def run_case(self, case: UavFlowCase, eval_dir: str | Path) -> dict[str, Any]:
        case_dir = Path(eval_dir) / "cases" / case.case_id
        run_dir = RunStore(self.config.run_root).create_run_dir(f"uav_flow_{case.case_id}")
        recorder = AsyncRunLogger(run_dir)
        env_client = UavFlowBenchmarkEnvClient(
            case,
            run_dir=run_dir,
            env_id=self.config.env_id,
            uav_flow_repo_path=self.config.uav_flow_repo_path,
            real_unrealcv=self.config.real_unrealcv,
            time_dilation=self.config.time_dilation,
            resolution=self.config.resolution,
            seed=self.config.seed,
            action_config=self.action_config,
        )
        status = "ok"
        error: dict[str, Any] | None = None
        try:
            recorder.write_json(
                "benchmark/uav_flow_case_planner_safe.json",
                case.planner_safe_metadata(),
            )
            recorder.write_json(
                "benchmark/uav_flow_case_evaluation_truth.json",
                case.evaluation_truth(),
            )
            runtime = TaskRuntime(
                env_client,
                recorder,
                TaskParser(mode=self.config.parser_mode, model_config=self.model_config),
                planner=ModelActionPlanner(self.model_config),
                safety=SafetySupervisor(self.safety_config),
                shutdown_policy=self.config.shutdown_policy,
                action_config=self.action_config,
                runtime_loop_guard_windows=self.config.runtime_loop_guard_windows,
            )
            state = runtime.run_task(
                "uav_flow_downtownwest",
                case.selected_instruction,
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

        trajectory_meta = export_trajectory(case, env_client.pose_history, case_dir / "uav_agent_trajectory.json")
        run_artifacts = run_artifact_summary(run_dir, self.action_config)
        leak_check = evaluation_truth_leak_check(run_dir)
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
        if leak_check.get("status") == "failed":
            status = "invalid_context_leak"
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
            "evaluation_truth_leak_check": leak_check,
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
            "evaluation_truth_leak_check": leak_check,
            "error": error,
        }


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
