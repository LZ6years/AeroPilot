"""Dry-run tests for the migrated UAV-Flow benchmark runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.runner import UavFlowBenchmarkRunner
from evaluation.benchmarks.uav_flow.runner import (
    _is_recoverable_simulator_error,
)
from src.agent.phase import COMPLETE_TASK, PhaseTransition
from src.agent.planning.schemas import CoupledActionPhasePlan
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.validator import semantic_plan_from_dict


def make_case() -> UavFlowCase:
    return UavFlowCase(
        case_id="dry",
        file_name="dry.json",
        source_path="dry.json",
        class_name="Turn",
        instruction="Turn to the person",
        instruction_unified="Turn to the direction of the person",
        selected_instruction="Turn to the direction of the person",
        initial_pos=[0, 0, 100, 0, 0, 0],
        reference_path_preprocessed=[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 10, 0]],
    )


def test_dry_runner_uses_simulation_server_profile_and_canonical_pose(tmp_path):
    runner = UavFlowBenchmarkRunner(
        UavFlowEvalConfig.from_dict({"execution_mode": "dry"})
    )

    assert runner.observation_profile.modalities == ("rgb", "depth", "pose")
    assert runner.map_package.control_protocol == "unrealcv"
    assert runner.map_package.unrealcv["control_target"] == "actor"


class FakeTaskParser:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def parse(self, instruction: str) -> TaskParseResult:
        plan = semantic_plan_from_dict(
            {
                "intent": "benchmark",
                "task_goal": {
                    "target": "person",
                    "target_attributes": {},
                    "final_state": "facing_target",
                },
                "phases": [
                    {
                        "name": "benchmark_phase",
                        "goal": "complete benchmark smoke",
                        "done_when": ["benchmark smoke task is complete"],
                        "continue_when": ["benchmark smoke task is incomplete"],
                    }
                ],
            }
        )
        return TaskParseResult(instruction=instruction, parser_mode="fake", plan=plan)


class FakePlanner:
    planner_mode = "fake"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def plan_next_with_phase_decision(self, state, context) -> CoupledActionPhasePlan:
        return CoupledActionPhasePlan(
            phase="benchmark_phase",
            phase_decision=PhaseTransition(COMPLETE_TASK, "dry runner complete"),
            steps=[],
            agent_messages=[],
        )


def test_runner_dry_writes_core_audit_truth_artifact_and_summary(tmp_path, monkeypatch):
    import evaluation.benchmarks.uav_flow.runner as runner_module

    monkeypatch.setattr(runner_module, "TaskParser", FakeTaskParser)
    monkeypatch.setattr(runner_module, "ModelActionPlanner", FakePlanner)
    config = UavFlowEvalConfig.from_dict(
        {
            "output_dir": str(tmp_path / "eval"),
            "run_root": str(tmp_path / "runs"),
            "execution_mode": "dry",
            "parser_mode": "fake",
            "safety_overrides": {"max_yaw_step_deg": 10.0},
            "runtime_loop_guard_windows": 7,
        }
    )
    runner = UavFlowBenchmarkRunner(config)
    case = make_case()

    result = runner.run_case(case, tmp_path / "eval" / "manual")

    assert result["status"] == "ok"
    assert "evaluation_truth_leak_check" not in result
    assert result["audit_summary"] == {
        "rgb_count": 0,
        "action_count": 0,
        "agent_message_count": 0,
    }
    assert result["visual_diagnostics"]["truth_leak_check"]["status"] == "passed"
    assert result["review_html"].endswith("review.html")
    assert runner.safety_config.max_yaw_step_deg == 10.0
    assert runner.config.runtime_loop_guard_windows == 7
    assert result["completed"] is True
    assert result["stopped"] is False
    assert result["turn_diagnostics"]["reference_yaw_delta_deg"] == 10.0
    assert result["turn_diagnostics"]["actual_yaw_delta_deg"] == 0.0
    assert result["turn_diagnostics"]["initial_pose_within_tolerance"] is True
    assert result["stmr"] == {"mode": "off", "status": "disabled"}
    assert not list(
        (tmp_path / "eval" / "manual" / "cases" / case.case_id).glob("*.md")
    )
    assert (tmp_path / "eval" / "manual" / "cases" / case.case_id / "plot_result.json").exists()
    assert (tmp_path / "eval" / "manual" / "cases" / case.case_id / "visual_diagnostics.json").exists()
    assert (tmp_path / "eval" / "manual" / "cases" / case.case_id / "stmr_summary.json").exists()
    assert (
        tmp_path
        / "eval"
        / "manual"
        / "cases"
        / case.case_id
        / "turn_diagnostics.json"
    ).exists()
    assert (
        tmp_path / "eval" / "manual" / "cases" / case.case_id / "audit.json"
    ).exists()
    case_payload = (
        tmp_path / "eval" / "manual" / "cases" / case.case_id / "case.json"
    ).read_text(encoding="utf-8")
    assert "reference_path_preprocessed" in case_payload
    assert "evaluation_truth_leak_check" not in case_payload
    assert '"audit_summary"' in case_payload
    run_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in Path(result["run_dir"]).rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt"}
    )
    assert "reference_path_preprocessed" not in run_text
    assert "target_pos" not in run_text
    assert (
        tmp_path
        / "eval"
        / "manual"
        / "cases"
        / case.case_id
        / "evaluation_truth.json"
    ).exists()


def test_runner_run_reuses_one_scene_for_multiple_cases(tmp_path, monkeypatch):
    """The batch boundary, not an individual case, owns scene stop."""

    import evaluation.benchmarks.uav_flow.runner as runner_module

    monkeypatch.setattr(runner_module, "TaskParser", FakeTaskParser)
    monkeypatch.setattr(runner_module, "ModelActionPlanner", FakePlanner)
    runner = UavFlowBenchmarkRunner(
        UavFlowEvalConfig.from_dict(
            {
                "output_dir": str(tmp_path / "eval"),
                "run_root": str(tmp_path / "runs"),
                "execution_mode": "dry",
                "parser_mode": "fake",
                "generate_html_review": False,
            }
        )
    )
    first = replace(make_case(), case_id="case_a", file_name="case_a.json")
    second = replace(make_case(), case_id="case_b", file_name="case_b.json")
    monkeypatch.setattr(runner, "load_cases", lambda: [first, second])

    result = runner.run(eval_id="batch_reuse")

    manifest = json.loads(
        (
            Path(result["eval_dir"]) / "scene_session.json"
        ).read_text(encoding="utf-8")
    )
    assert len(result["results"]) == 2
    assert manifest["status"] == "closed"
    assert manifest["start_count"] == 1
    assert manifest["restart_count"] == 0
    assert manifest["stop_count"] == 1


def test_simulator_recovery_classifier_does_not_restart_for_model_network_errors():
    assert _is_recoverable_simulator_error(
        {
            "error_type": "RuntimeError",
            "error": "UnrealCV camera follow validation failed",
        }
    )
    assert not _is_recoverable_simulator_error(
        {
            "error_type": "ConnectionError",
            "error": "TaskParser model provider SSL EOF at dashscope",
        }
    )


def test_runner_retries_one_case_after_in_place_simulator_recovery(
    tmp_path, monkeypatch
):
    import evaluation.benchmarks.uav_flow.runner as runner_module

    monkeypatch.setattr(runner_module, "TaskParser", FakeTaskParser)
    monkeypatch.setattr(runner_module, "ModelActionPlanner", FakePlanner)
    runner = UavFlowBenchmarkRunner(
        UavFlowEvalConfig.from_dict(
            {
                "output_dir": str(tmp_path / "eval"),
                "run_root": str(tmp_path / "runs"),
                "execution_mode": "dry",
                "parser_mode": "fake",
                "generate_html_review": False,
            }
        )
    )
    eval_dir = tmp_path / "eval" / "recovery"
    session = runner._new_scene_session(eval_dir)
    real_prepare = session.prepare_episode
    calls = 0

    def flaky_prepare(episode):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("UnrealCV camera follow validation failed")
        return real_prepare(episode)

    monkeypatch.setattr(session, "prepare_episode", flaky_prepare)
    try:
        result = runner._run_case_with_session(make_case(), eval_dir, session)
    finally:
        session.close()

    assert result["status"] == "ok"
    assert result["simulator_recovery_attempts"] == [
        {
            "attempt": 1,
            "run_dir": result["simulator_recovery_attempts"][0]["run_dir"],
            "error": {
                "error_type": "RuntimeError",
                "error": "UnrealCV camera follow validation failed",
            },
            "recovery_action": "resynchronize",
            "status": "completed",
        }
    ]
    assert calls == 2


def test_config_roundtrip_includes_runtime_scene_fields():
    config = UavFlowEvalConfig.from_dict(
        {
            "execution_mode": "attach",
            "map_config_path": "configs/maps/unrealcv_actor_control.example.json",
            "map_id": "downtownwest_local_unrealcv",
            "headless": False,
            "save_artifacts": True,
            "limit_per_class": 3,
        }
    )

    assert config.execution_mode == "attach"
    assert config.headless is False
    assert config.save_artifacts is True
    assert config.limit_per_class == 3
    assert UavFlowEvalConfig.from_dict({}).execution_mode == "dry"
    assert UavFlowEvalConfig.from_dict({}).runtime_loop_guard_windows == 40
    assert UavFlowEvalConfig.from_dict({}).planner_depth_input == "off"
    assert config.to_dict()["save_artifacts"] is True


def test_config_rejects_legacy_gym_scene_fields():
    with pytest.raises(ValueError, match="Legacy UAV-Flow gym scene fields"):
        UavFlowEvalConfig.from_dict({"env_id": "old", "real_unrealcv": True})


def test_config_exact_case_ids_disable_class_and_count_selectors():
    config = UavFlowEvalConfig.from_dict(
        {
            "case_ids": ["move_a", "turn_a"],
            "classes": ["Turn"],
            "limit_per_class": 1,
            "max_cases": 1,
        }
    )

    assert config.case_ids == ("move_a", "turn_a")
    assert config.classes is None
    assert config.limit_per_class is None
    assert config.max_cases is None


@pytest.mark.parametrize(
    "case_ids",
    [
        ["case.json"],
        ["../case"],
        ["case", "case"],
        [],
    ],
)
def test_config_rejects_invalid_exact_case_ids(case_ids):
    with pytest.raises(ValueError, match="case_ids"):
        UavFlowEvalConfig.from_dict({"case_ids": case_ids})


def test_manual_eval_cli_omitted_options_preserve_base_config():
    from evaluation.benchmarks.uav_flow.scripts.run_manual_eval import (
        _config_from_args,
        _parser,
    )

    args = _parser().parse_args([])
    config = _config_from_args(args)

    # ``None`` means "no CLI override". The effective values below come from
    # uav_flow_eval.example.json rather than a second set of script defaults.
    assert args.runtime_loop_guard_windows is None
    assert args.parser_mode is None
    assert args.enable_sr_judge is None
    assert args.enable_visual_diagnostics is None
    assert config.runtime_loop_guard_windows == 40
    assert config.parser_mode == "model"
    assert config.classes == ["Turn", "Move", "Approach"]
    assert config.limit_per_class == 1
    assert config.enable_sr_judge is False
    assert config.enable_visual_diagnostics is False


def test_manual_eval_cli_all_cases_clears_every_selector_and_preserves_switches(
    tmp_path,
):
    from evaluation.benchmarks.uav_flow.scripts.run_manual_eval import (
        _config_from_args,
        _parser,
    )

    config_path = tmp_path / "full.local"
    config_path.write_text(
        json.dumps(
            {
                "case_ids": ["representative_case"],
                "classes": ["Turn"],
                "limit_per_class": 1,
                "max_cases": 1,
                "runtime_loop_guard_windows": 80,
                "planner_depth_input": "now",
                "enable_sr_judge": False,
                "enable_visual_diagnostics": False,
                "stmr_mode": "off",
            }
        ),
        encoding="utf-8",
    )

    args = _parser().parse_args(
        ["--config", str(config_path), "--all-cases"]
    )
    config = _config_from_args(args)

    assert config.case_ids is None
    assert config.classes is None
    assert config.limit_per_class is None
    assert config.max_cases is None
    assert config.runtime_loop_guard_windows == 80
    assert config.planner_depth_input == "now"
    assert config.enable_sr_judge is False
    assert config.enable_visual_diagnostics is False
    assert config.stmr_mode == "off"


def test_manual_eval_cli_accepts_exact_case_ids():
    from evaluation.benchmarks.uav_flow.scripts.run_manual_eval import (
        _config_from_args,
        _parser,
    )

    args = _parser().parse_args(
        ["--case-ids", "move_a,turn_a", "--max-cases", "1"]
    )
    config = _config_from_args(args)

    assert config.case_ids == ("move_a", "turn_a")
    assert config.classes is None
    assert config.limit_per_class is None
    assert config.max_cases is None


def test_config_roundtrip_includes_stmr_experiment_fields():
    config = UavFlowEvalConfig.from_dict(
        {
            "stmr_mode": "planner",
            "stmr_prior": "urban_uav_prior_v1",
            "stmr_max_queries": 12,
            "stmr_device": "cpu",
        }
    )

    assert config.stmr_mode == "planner"
    assert config.stmr_prior == "urban_uav_prior_v1"
    assert config.stmr_max_queries == 12
    assert config.stmr_device == "cpu"
    assert UavFlowEvalConfig.from_dict({}).stmr_mode == "off"


def test_runner_builds_case_local_stmr_context_without_global_config():
    class Planner:
        planner_mode = "model"

    session = object()
    runner = UavFlowBenchmarkRunner(
        UavFlowEvalConfig.from_dict({"stmr_mode": "shadow", "stmr_device": "cpu"}),
        stmr_session=session,
    )

    context_builder, stmr_module = runner._context_builder(Planner())

    assert context_builder is not None
    assert stmr_module is not None
    assert stmr_module.mode == "shadow"
    assert stmr_module.online_builder.session is session
    assert context_builder.composer.registry.modules["map"] is stmr_module


def test_runner_profile_capture_is_independent_from_planner_depth_context():
    baseline = UavFlowBenchmarkRunner(UavFlowEvalConfig.from_dict({"stmr_mode": "off"}))
    depth = UavFlowBenchmarkRunner(
        UavFlowEvalConfig.from_dict(
            {
                "stmr_mode": "off",
                "planner_depth_input": "now",
            }
        )
    )
    stmr = UavFlowBenchmarkRunner(
        UavFlowEvalConfig.from_dict({"stmr_mode": "shadow"}),
        stmr_session=object(),
    )

    assert baseline.observation_profile.modalities == ("rgb", "depth", "pose")
    assert depth.observation_profile.modalities == ("rgb", "depth", "pose")
    assert stmr.observation_profile.modalities == (
        "rgb",
        "depth",
        "segmentation",
        "pose",
    )


def test_runner_builds_depth_image_context_without_stmr():
    class Planner:
        planner_mode = "model"

    runner = UavFlowBenchmarkRunner(
        UavFlowEvalConfig.from_dict(
            {
                "stmr_mode": "off",
                "planner_depth_input": "now",
            }
        )
    )

    context_builder, stmr_module = runner._context_builder(Planner())

    assert context_builder is not None
    assert stmr_module is None
    image_module = context_builder.composer.registry.modules["image"]
    assert image_module.include_current_depth is True
    assert context_builder.composer.registry.modules["map"].name == "map"
    assert any(
        capability.startswith("NOW metric depth overlay:")
        for capability in context_builder.semantic_evidence_capabilities()
    )


def test_runner_warms_shared_stmr_session_once_before_cases():
    class Session:
        def __init__(self):
            self.calls = 0

        def warmup(self):
            self.calls += 1

    session = Session()
    runner = UavFlowBenchmarkRunner(
        UavFlowEvalConfig.from_dict({"stmr_mode": "shadow"}),
        stmr_session=session,
    )

    runner._prepare_stmr()
    runner._prepare_stmr()

    assert session.calls == 1


def test_runner_uses_independent_runtime_scene_session_per_case(tmp_path, monkeypatch):
    import evaluation.benchmarks.uav_flow.runner as runner_module

    monkeypatch.setattr(runner_module, "TaskParser", FakeTaskParser)
    monkeypatch.setattr(runner_module, "ModelActionPlanner", FakePlanner)
    cases = [
        UavFlowCase(
            **{
                **make_case().__dict__,
                "case_id": f"dry_{class_name.lower()}",
                "file_name": f"dry_{class_name.lower()}.json",
                "class_name": class_name,
                "instruction": instruction,
                "instruction_unified": instruction,
                "selected_instruction": instruction,
            }
        )
        for class_name, instruction in (
            ("Turn", "Turn to face the person."),
            ("Move", "Move toward the person."),
            ("Approach", "Approach the person and stop nearby."),
        )
    ]
    monkeypatch.setattr(UavFlowBenchmarkRunner, "load_cases", lambda self: cases)
    config = UavFlowEvalConfig.from_dict(
        {
            "output_dir": str(tmp_path / "eval"),
            "run_root": str(tmp_path / "runs"),
            "execution_mode": "dry",
            "parser_mode": "fake",
            "enable_sr_judge": False,
            "enable_visual_diagnostics": False,
        }
    )

    result = UavFlowBenchmarkRunner(config).run(eval_id="reuse")

    assert len(result["results"]) == 3
    assert len({item["run_dir"] for item in result["results"]}) == 3
    assert all(item["status"] == "ok" for item in result["results"])
    for class_name in ("turn", "move", "approach"):
        case_dir = (
            tmp_path / "eval" / "reuse" / "cases" / f"dry_{class_name}"
        )
        assert not list(case_dir.glob("*.md"))
        assert (case_dir / "audit.json").exists()
        html = (case_dir / "review.html").read_text(encoding="utf-8")
        assert "Observation Gallery" in html
        assert "Planner Windows" in html
        assert "Action Timeline" in html
        assert "Trajectory Plots" in html
    assert result["summary_paths"]["html_review"].endswith("review.html")
    assert (tmp_path / "eval" / "reuse" / "review.html").exists()
    assert (
        tmp_path
        / "eval"
        / "reuse"
        / "review_assets"
        / "benchmark_review.css"
    ).exists()


def test_real_review_requires_artifacts_and_planner_debug():
    with pytest.raises(ValueError, match="save_artifacts=true"):
        UavFlowBenchmarkRunner(
            UavFlowEvalConfig.from_dict(
                {
                    "execution_mode": "attach",
                    "generate_html_review": True,
                }
            )
        )
