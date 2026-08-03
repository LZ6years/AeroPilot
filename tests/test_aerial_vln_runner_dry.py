"""In-memory AirSim episode test for the AerialVLN runner."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import (
    AerialVlnBenchmarkRunner,
)
from src.agent.phase import COMPLETE_TASK, PhaseTransition
from src.agent.planning.schemas import CoupledActionPhasePlan
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.validator import semantic_plan_from_dict
from src.runtime.logging.async_run_logger import AsyncRunLogger


def test_aerial_vln_dry_episode_writes_metrics_manifest_without_truth_leak(
    tmp_path, monkeypatch
):
    import evaluation.benchmarks.aerial_vln.runner as runner_module

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "val_seen.json").write_text(
        json.dumps({"episodes": [episode()]}), encoding="utf-8"
    )
    monkeypatch.setattr(runner_module, "TaskParser", FakeTaskParser)
    monkeypatch.setattr(
        runner_module, "ModelActionPlanner", FakePlanner
    )
    config = AerialVlnEvalConfig.from_dict(
        {
            "dataset_root": str(dataset),
            "split": "val_seen",
            "output_dir": str(tmp_path / "eval"),
            "run_root": str(tmp_path / "runs"),
            "execution_mode": "dry",
            "parser_mode": "fake",
            "scene_map": {"1": "airvln_env_1"},
            "initialization_min_settle_sec": 0.25,
        }
    )

    runner = AerialVlnBenchmarkRunner(config)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    runtime_recorder = AsyncRunLogger(runtime_dir)
    try:
        runtime = runner._make_runtime(
            runtime_dir,
            recorder=runtime_recorder,
            task_parser=FakeTaskParser(),
            planner=FakePlanner(),
        )
        assert (
            runtime.stabilizer.config.initialization_min_settle_sec
            == 0.25
        )
    finally:
        runtime_recorder.close()
    result = runner.run(eval_id="dry")

    case_result = result["results"][0]
    assert case_result["status"] == "ok"
    assert case_result["completed"] is True
    assert case_result["metrics"]["success"] is True
    assert case_result["metrics"]["steps_taken"] == 1
    case_dir = Path(case_result["case_dir"])
    assert (case_dir / "case_result.json").exists()
    assert (case_dir / "trajectory.json").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "manifest.json").exists()
    assert (case_dir / "audit.json").exists()
    assert (case_dir / "review.html").exists()
    assert Path(result["summary_paths"]["html_review"]).exists()
    run_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in Path(case_result["run_dir"]).rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt"}
    )
    assert "goal_position_ned" not in run_text
    assert "reference_path_ned" not in run_text
    assert "official_actions" not in run_text
    manifest = json.loads(
        Path(result["summary_paths"]["manifest"]).read_text(
            encoding="utf-8"
        )
    )
    assert (
        manifest["protocol_variant"]
        == "aerial_vln.uav_agent_native_actions.v1"
    )
    assert manifest["scene_map"] == {"1": "airvln_env_1"}
    assert manifest["runtime_loop_guard_windows"] == 40
    assert manifest["initialization_min_settle_sec"] == 0.25


def test_aerial_real_review_requires_persistent_debug_artifacts(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "val_seen.json").write_text(
        json.dumps({"episodes": [episode()]}), encoding="utf-8"
    )
    config = AerialVlnEvalConfig.from_dict(
        {
            "dataset_root": str(dataset),
            "execution_mode": "attach",
            "scene_map": {"1": "airvln_env_1"},
        }
    )

    try:
        AerialVlnBenchmarkRunner(config)
    except ValueError as exc:
        assert "save_artifacts=true" in str(exc)
    else:
        raise AssertionError("real HTML review config should fail fast")


class FakeTaskParser:
    def __init__(self, *args, **kwargs) -> None:
        return

    def parse(self, instruction: str) -> TaskParseResult:
        plan = semantic_plan_from_dict(
            {
                "intent": "navigation",
                "task_goal": {
                    "target": "goal",
                    "target_attributes": {},
                    "final_state": "at_goal",
                },
                "phases": [
                    {
                        "name": "navigate",
                        "goal": "complete dry episode",
                        "done_when": ["dry episode complete"],
                        "continue_when": ["not complete"],
                        "metrics": [],
                    }
                ],
            }
        )
        return TaskParseResult(
            instruction=instruction, parser_mode="fake", plan=plan
        )


class FakePlanner:
    planner_mode = "fake"

    def __init__(self, *args, **kwargs) -> None:
        return

    def plan_next_with_phase_decision(
        self, state, context
    ) -> CoupledActionPhasePlan:
        return CoupledActionPhasePlan(
            phase=state.current_phase_name,
            phase_decision=PhaseTransition(
                COMPLETE_TASK, "dry episode complete"
            ),
            steps=[],
            agent_messages=[],
        )


def episode() -> dict:
    return {
        "episode_id": "ep_1",
        "trajectory_id": "traj_1",
        "scene_id": "1",
        "instruction": {"instruction_text": "Hold at the start"},
        "start_position": [1.0, 2.0, -3.0],
        "start_rotation": [1.0, 0.0, 0.0, 0.0],
        "goals": [{"position": [1.0, 2.0, -3.0]}],
        "reference_path": [[1.0, 2.0, -3.0]],
        "actions": [0],
    }
