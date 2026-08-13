"""In-memory AirSim episode test for the AerialVLN runner."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

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
            "dataset_family": "aerialvln-s",
            "difficulties": ["low", "medium", "high"],
            "limit_per_difficulty": None,
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
    assert case_result["difficulty"] == "low"
    assert case_result["reference_path_length_m"] == 0.0
    assert case_result["dataset_split"] == "val_seen"
    assert case_result["dataset_split_label"] == "Validation Seen"
    review_html = (case_dir / "review.html").read_text(encoding="utf-8")
    assert "Dataset And Difficulty" in review_html
    assert "aerialvln-s" in review_html
    assert "Validation Seen" in review_html
    assert "reference_path_length_m" in review_html
    assert Path(result["summary_paths"]["html_review"]).exists()
    assert Path(result["summary_paths"]["dataset_provenance"]).exists()
    assert Path(result["summary_paths"]["difficulty_assignment"]).exists()
    assert Path(result["summary_paths"]["case_selection"]).exists()
    run_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in Path(case_result["run_dir"]).rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".txt"}
    )
    assert "goal_position_ned" not in run_text
    assert "reference_path_ned" not in run_text
    assert "official_actions" not in run_text
    assert "reference_path_length_m" not in run_text
    assert '"difficulty"' not in run_text
    assert '"dataset_split"' not in run_text
    assert "Validation Seen" not in run_text
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
    assert manifest["dataset"]["dataset_family"] == "aerialvln-s"
    assert manifest["dataset"]["split"] == "val_seen"
    assert manifest["dataset_split_label"] == "Validation Seen"
    assert manifest["difficulty_selection"]["selected_counts"] == {
        "low": 1,
        "medium": 0,
        "high": 0,
    }
    summary = json.loads(
        Path(result["summary_paths"]["summary_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["dataset"]["dataset_family"] == "aerialvln-s"
    assert summary["dataset_split_label"] == "Validation Seen"
    assert summary["difficulty_selection"]["thresholds_m"] == {
        "q33": 0.0,
        "q67": 0.0,
    }
    assert summary["execution_readiness"]["missing_scenes"] == []
    assert summary["batch_status"] == "completed"
    assert summary["abort_reason"] is None
    assert summary["requested_case_count"] == 1
    assert summary["completed_case_count"] == 1
    difficulty_assignment = json.loads(
        Path(result["summary_paths"]["difficulty_assignment"]).read_text(
            encoding="utf-8"
        )
    )
    case_selection = json.loads(
        Path(result["summary_paths"]["case_selection"]).read_text(
            encoding="utf-8"
        )
    )
    for artifact in (difficulty_assignment, case_selection):
        assert artifact["dataset"]["split"] == "val_seen"
        assert artifact["dataset"]["split_sha256"]
        assert artifact["dataset_split_label"] == "Validation Seen"


def test_runner_aborts_after_infrastructure_failure_and_writes_partial_summary(
    tmp_path, monkeypatch
):
    runner = _runner_with_three_cases(tmp_path)
    seen: list[str] = []

    def fake_run_case(case, eval_dir, **kwargs):
        seen.append(case.episode_id)
        infrastructure_error = (
            {"stage": "cleanup", "errors": [{"error": "leaked scene"}]}
            if len(seen) == 2
            else None
        )
        return _fake_case_result(case, infrastructure_error)

    monkeypatch.setattr(runner, "run_case", fake_run_case)

    result = runner.run(eval_id="abort")

    assert seen == ["ep_1", "ep_2"]
    assert result["summary"]["batch_status"] == "aborted"
    assert result["summary"]["requested_case_count"] == 3
    assert result["summary"]["completed_case_count"] == 2
    assert result["summary"]["abort_reason"]["episode_id"] == "ep_2"
    assert Path(result["summary_paths"]["summary_json"]).exists()
    assert Path(result["summary_paths"]["manifest"]).exists()


def test_runner_continues_after_benchmark_failure(tmp_path, monkeypatch):
    runner = _runner_with_three_cases(tmp_path)
    seen: list[str] = []

    def fake_run_case(case, eval_dir, **kwargs):
        seen.append(case.episode_id)
        result = _fake_case_result(case, None)
        result["status"] = "error"
        result["error"] = {
            "error_type": "PlannerError",
            "error": "model action failed",
        }
        return result

    monkeypatch.setattr(runner, "run_case", fake_run_case)

    result = runner.run(eval_id="benchmark_failure")

    assert seen == ["ep_1", "ep_2", "ep_3"]
    assert result["summary"]["batch_status"] == "completed"
    assert result["summary"]["completed_case_count"] == 3
    assert result["summary"]["abort_reason"] is None


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

    runner = AerialVlnBenchmarkRunner(config)
    with pytest.raises(ValueError, match="save_artifacts=true"):
        runner.run(eval_id="must_not_start")


def test_validate_only_preparation_reports_all_scenes_without_model_or_server(
    tmp_path, monkeypatch
):
    import evaluation.benchmarks.aerial_vln.runner as runner_module

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    episodes = [
        {
            **episode(),
            "episode_id": f"ep_{scene_id}",
            "trajectory_id": f"traj_{scene_id}",
            "scene_id": scene_id,
            "reference_path": [[0, 0, 0], [length, 0, 0]],
        }
        for scene_id, length in (("1", 1), ("2", 2), ("3", 3))
    ]
    (dataset / "val_seen.json").write_text(
        json.dumps({"episodes": episodes}), encoding="utf-8"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("validate-only must not load model or server")

    monkeypatch.setattr(runner_module, "load_model_config", forbidden)
    monkeypatch.setattr(runner_module, "SimulationServer", forbidden)
    runner = AerialVlnBenchmarkRunner(
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": str(dataset),
                "dataset_family": "aerialvln-s",
                "map_config_path": "configs/maps/airvln_maps.json",
                "scene_map": {"1": "airvln_env_1"},
            }
        )
    )

    prepared = runner.prepare_evaluation(
        require_execution_ready=False
    )

    assert prepared.provenance.scene_ids == ("1", "2", "3")
    assert prepared.selection.population_scene_ids == ("1", "2", "3")
    assert prepared.execution_readiness["ready"] is False
    assert prepared.execution_readiness["missing_scenes"] == ["2", "3"]
    assert prepared.execution_readiness["missing_scene_map"] == ["2", "3"]


def test_aerial_vln_s_validate_only_allows_no_scene_mappings(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "val_seen.json").write_text(
        json.dumps(
            {
                "episodes": [
                    episode(),
                    {
                        **episode(),
                        "episode_id": "ep_2",
                        "trajectory_id": "traj_2",
                        "scene_id": "2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = AerialVlnBenchmarkRunner(
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": str(dataset),
                "dataset_family": "aerialvln-s",
                "scene_map": {},
            }
        )
    )

    prepared = runner.prepare_evaluation(
        require_execution_ready=False
    )

    assert prepared.execution_readiness["missing_scenes"] == ["1", "2"]
    assert prepared.execution_readiness["missing_scene_map"] == ["1", "2"]


def test_real_aerial_vln_s_fails_before_server_when_hash_or_scenes_not_ready(
    tmp_path, monkeypatch
):
    import evaluation.benchmarks.aerial_vln.runner as runner_module

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    split_path = dataset / "val_unseen.json"
    split_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        **episode(),
                        "scene_id": "2",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner_module,
        "SimulationServer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("server must not start")
        ),
    )
    config = AerialVlnEvalConfig.from_dict(
        {
            "dataset_root": str(dataset),
            "dataset_family": "aerialvln-s",
            "split": "val_unseen",
            "dataset_version": "official",
            "expected_split_sha256": hashlib.sha256(
                split_path.read_bytes()
            ).hexdigest(),
            "execution_mode": "attach",
            "generate_html_review": False,
            "map_config_path": "configs/maps/airvln_maps.json",
            "scene_map": {"1": "airvln_env_1"},
        }
    )

    with pytest.raises(ValueError, match="not execution-ready"):
        AerialVlnBenchmarkRunner(config).run(eval_id="must_not_start")


def test_aerial_vln_s_sha_mismatch_fails_during_offline_preparation(
    tmp_path,
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "val_unseen.json").write_text(
        json.dumps({"episodes": [episode()]}),
        encoding="utf-8",
    )
    runner = AerialVlnBenchmarkRunner(
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": str(dataset),
                "dataset_family": "aerialvln-s",
                "split": "val_unseen",
                "expected_split_sha256": "0" * 64,
                "scene_map": {"1": "airvln_env_1"},
            }
        )
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runner.prepare_evaluation(require_execution_ready=False)


def test_real_aerial_vln_s_requires_expected_hash_before_server(
    tmp_path, monkeypatch
):
    import evaluation.benchmarks.aerial_vln.runner as runner_module

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "val_unseen.json").write_text(
        json.dumps({"episodes": [episode()]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner_module,
        "SimulationServer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("server must not start")
        ),
    )
    runner = AerialVlnBenchmarkRunner(
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": str(dataset),
                "dataset_family": "aerialvln-s",
                "split": "val_unseen",
                "dataset_version": "official",
                "execution_mode": "attach",
                "generate_html_review": False,
                "scene_map": {"1": "airvln_env_1"},
            }
        )
    )

    with pytest.raises(ValueError, match="expected_split_sha256"):
        runner.run(eval_id="must_not_start")


def test_seen_and_unseen_compute_independent_split_thresholds(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split, scale in (("val_seen", 1.0), ("val_unseen", 10.0)):
        episodes = [
            {
                **episode(),
                "episode_id": f"{split}_{index}",
                "trajectory_id": f"{split}_traj_{index}",
                "reference_path": [
                    [0, 0, 0],
                    [index * scale, 0, 0],
                ],
            }
            for index in range(1, 10)
        ]
        (dataset / f"{split}.json").write_text(
            json.dumps({"episodes": episodes}),
            encoding="utf-8",
        )

    def prepare(split: str):
        runner = AerialVlnBenchmarkRunner(
            AerialVlnEvalConfig.from_dict(
                {
                    "dataset_root": str(dataset),
                    "dataset_family": "aerialvln-s",
                    "split": split,
                    "scene_map": {},
                }
            )
        )
        return runner.prepare_evaluation(
            require_execution_ready=False
        )

    unseen_first = prepare("val_unseen")
    seen = prepare("val_seen")
    unseen_second = prepare("val_unseen")

    assert seen.selection.q33_m == pytest.approx(11.0 / 3.0)
    assert seen.selection.q67_m == pytest.approx(19.0 / 3.0)
    assert unseen_first.selection.q33_m == pytest.approx(110.0 / 3.0)
    assert unseen_first.selection.q67_m == pytest.approx(190.0 / 3.0)
    assert unseen_first.selection.to_dict() == (
        unseen_second.selection.to_dict()
    )
    assert seen.provenance.split == "val_seen"
    assert unseen_first.provenance.split == "val_unseen"


def test_seen_sha_cannot_validate_unseen_split(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    seen_path = dataset / "val_seen.json"
    unseen_path = dataset / "val_unseen.json"
    seen_path.write_text(
        json.dumps({"episodes": [episode()]}),
        encoding="utf-8",
    )
    unseen_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        **episode(),
                        "episode_id": "unseen_ep",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = AerialVlnBenchmarkRunner(
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": str(dataset),
                "dataset_family": "aerialvln-s",
                "split": "val_unseen",
                "expected_split_sha256": hashlib.sha256(
                    seen_path.read_bytes()
                ).hexdigest(),
                "scene_map": {},
            }
        )
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runner.prepare_evaluation(require_execution_ready=False)


def test_multi_scene_dry_runner_executes_selected_cases_without_scene_one_limit(
    tmp_path, monkeypatch
):
    import evaluation.benchmarks.aerial_vln.runner as runner_module

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    episodes = [
        {
            **episode(),
            "episode_id": f"ep_{index}",
            "trajectory_id": f"traj_{index}",
            "scene_id": str(((index - 1) % 3) + 1),
            "reference_path": [[0, 0, 0], [index, 0, 0]],
        }
        for index in range(1, 10)
    ]
    (dataset / "val_unseen.json").write_text(
        json.dumps({"episodes": episodes}),
        encoding="utf-8",
    )
    map_config = tmp_path / "maps.json"
    map_config.write_text(
        json.dumps(
            {
                "maps": [
                    {
                        "map_id": f"dry_scene_{scene_id}",
                        "provider": "packaged_unreal",
                        "scene_id": scene_id,
                        "root": None,
                        "connection": {
                            "mode": "local",
                            "data_transport": "direct",
                        },
                        "launch": {"manual": True, "args": []},
                        "control": {"protocol": "airsim"},
                        "airsim": {
                            "mode": "Multirotor",
                            "vehicle_name": "Drone_1",
                            "camera_name": "front_0",
                        },
                    }
                    for scene_id in ("1", "2", "3")
                ],
                "port_policy": {
                    "control_port_start": 30001,
                    "max_scene_num": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_module, "TaskParser", FakeTaskParser)
    monkeypatch.setattr(runner_module, "ModelActionPlanner", FakePlanner)
    monkeypatch.setattr(
        runner_module, "load_model_config", lambda *_args, **_kwargs: object()
    )
    runner = AerialVlnBenchmarkRunner(
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": str(dataset),
                "dataset_family": "aerialvln-s",
                "split": "val_unseen",
                "difficulties": ["low"],
                "limit_per_difficulty": 3,
                "map_config_path": str(map_config),
                "scene_map": {
                    scene_id: f"dry_scene_{scene_id}"
                    for scene_id in ("1", "2", "3")
                },
                "output_dir": str(tmp_path / "eval"),
                "run_root": str(tmp_path / "runs"),
                "execution_mode": "dry",
                "parser_mode": "fake",
            }
        )
    )

    result = runner.run(eval_id="multi_scene")

    assert len(result["results"]) == 3
    assert {
        item["scene_id"] for item in result["results"]
    } == {"1", "2", "3"}
    manifest = json.loads(
        Path(result["summary_paths"]["manifest"]).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["dataset"]["scene_ids"] == ["1", "2", "3"]
    assert manifest["dataset"]["split"] == "val_unseen"
    assert manifest["dataset_split_label"] == "Validation Unseen"
    assert manifest["execution_readiness"]["ready"] is True
    summary_html = Path(
        result["summary_paths"]["html_review"]
    ).read_text(encoding="utf-8")
    assert "val_unseen" in summary_html
    assert "Validation Unseen" in summary_html
    case_html = (
        Path(result["results"][0]["case_dir"]) / "review.html"
    ).read_text(encoding="utf-8")
    assert "val_unseen" in case_html
    assert "Validation Unseen" in case_html


def test_validate_only_cli_does_not_require_execution_ready(
    tmp_path, monkeypatch, capsys
):
    import evaluation.benchmarks.aerial_vln.runner as runner_module
    from evaluation.benchmarks.aerial_vln.scripts.run_eval import main

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "val_unseen.json").write_text(
        json.dumps(
            {
                "episodes": [
                    episode(),
                    {
                        **episode(),
                        "episode_id": "ep_2",
                        "trajectory_id": "traj_2",
                        "scene_id": "2",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset),
                "dataset_family": "aerialvln-s",
                "map_config_path": "configs/maps/airvln_maps.json",
                "scene_map": {"1": "airvln_env_1"},
                "execution_mode": "dry",
                "output_dir": str(tmp_path / "validation"),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        runner_module,
        "load_model_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model config must remain lazy")
        ),
    )
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--split",
                "val_unseen",
                "--validate-only",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["dataset_provenance"]["dataset_family"] == "aerialvln-s"
    assert payload["dataset_provenance"]["split"] == "val_unseen"
    assert (
        payload["difficulty_assignment"]["dataset_split_label"]
        == "Validation Unseen"
    )
    assert (
        payload["case_selection"]["dataset"]["split"]
        == "val_unseen"
    )
    assert payload["execution_readiness"]["missing_scenes"] == ["2"]
    assert payload["execution_readiness"]["missing_scene_map"] == ["2"]
    for path in payload["validation_paths"].values():
        assert Path(path).exists()


def test_cli_overrides_difficulty_limit_and_dataset_identity(tmp_path):
    from evaluation.benchmarks.aerial_vln.scripts.run_eval import (
        _config_from_args,
        _parser,
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_root": "old",
                "dataset_family": "aerialvln-s",
                "difficulties": ["low", "medium", "high"],
                "limit_per_difficulty": 3,
                "scene_map": {"1": "airvln_env_1"},
            }
        ),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--config",
            str(config_path),
            "--dataset-root",
            "new",
            "--split",
            "val_unseen",
            "--expected-split-sha256",
            "a" * 64,
            "--difficulties",
            "medium,high",
            "--all-per-difficulty",
        ]
    )

    config = _config_from_args(args)

    assert config.dataset_root == "new"
    assert config.split == "val_unseen"
    assert config.expected_split_sha256 == "a" * 64
    assert config.difficulties == ("medium", "high")
    assert config.limit_per_difficulty is None


def _runner_with_three_cases(tmp_path: Path) -> AerialVlnBenchmarkRunner:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    episodes = [
        {
            **episode(),
            "episode_id": f"ep_{index}",
            "trajectory_id": f"traj_{index}",
            "reference_path": [[0, 0, 0], [index, 0, 0]],
        }
        for index in range(1, 4)
    ]
    (dataset / "val_seen.json").write_text(
        json.dumps({"episodes": episodes}), encoding="utf-8"
    )
    return AerialVlnBenchmarkRunner(
        AerialVlnEvalConfig.from_dict(
            {
                "dataset_root": str(dataset),
                "dataset_family": "aerialvln-s",
                "split": "val_seen",
                "scene_map": {"1": "airvln_env_1"},
                "output_dir": str(tmp_path / "eval"),
                "run_root": str(tmp_path / "runs"),
                "execution_mode": "dry",
                "generate_html_review": False,
            }
        )
    )


def _fake_case_result(case, infrastructure_error):
    return {
        "episode_id": case.episode_id,
        "scene_id": case.scene_id,
        "difficulty": None,
        "reference_path_length_m": None,
        "status": "ok",
        "metrics": {},
        "infrastructure_error": infrastructure_error,
    }


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
