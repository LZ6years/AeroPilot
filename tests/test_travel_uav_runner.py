"""TravelUAV runner integration using explicitly injected test doubles."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.benchmark_fakes import TravelUavBenchmarkRunner
from src.agent.transcript import AgentToolCall, AssistantTurn

from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.protocol import TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
from evaluation.benchmarks.travel_uav.runner import (
    _infrastructure_error,
)
from tests.travel_uav_test_support import case, write_index, write_map_config


class CompletePlanner:
    planner_mode = "fake"
    last_protocol_context = None

    def __init__(self, *args, **kwargs):
        pass

    def plan_turn(self, transcript, registry, **kwargs):
        type(self).last_protocol_context = transcript.instruction_envelope().get("protocol_context")
        return AssistantTurn("fixture_turn", "Hold at the observed endpoint", [
            AgentToolCall("fixture_done", "runtime.report_completed", {
                "reason": "Fixture endpoint observed",
                "evidence_refs": [transcript.latest_observation_id()],
            })
        ])


def test_validate_only_writes_provenance_without_loading_model(tmp_path):
    runner = _runner(tmp_path)
    result = runner.validate(eval_id="validate")

    assert result["case_selection"]["selected_case_ids"] == ["easy", "hard"]
    assert Path(result["validation_paths"]["dataset_provenance"]).is_file()
    assert runner._model_config_value is None


def test_dry_runner_uses_protocol_and_generates_core_html(tmp_path):
    runner = _runner(tmp_path)
    runner._planner_factory = lambda model: CompletePlanner()

    result = runner.run(eval_id="dry")

    assert result["summary"]["batch_status"] == "completed"
    assert len(result["results"]) == 2
    assert CompletePlanner.last_protocol_context == TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
    assert Path(result["summary_paths"]["html_review"]).is_file()
    for item in result["results"]:
        assert (Path(item["case_dir"]) / "review.html").is_file()
        assert (Path(item["case_dir"]) / "audit.json").is_file()


def test_real_mode_rejects_missing_index_sha_before_simulator(tmp_path):
    runner = _runner(tmp_path, execution_mode="managed", with_sha=False)
    with pytest.raises(ValueError, match="expected_index_sha256"):
        runner.prepare_evaluation(require_execution_ready=True)


def test_missing_map_is_reported_offline_and_rejected_for_run(tmp_path):
    index = write_index(tmp_path / "index.json", [case("missing", map_name="Other")])
    maps = write_map_config(tmp_path / "maps.json")
    runner = TravelUavBenchmarkRunner(
        TravelUavEvalConfig.from_dict(
            {**{"execution_mode": "managed", "save_artifacts": True, "enable_action_planner_debug": True},
                "index_path": str(index),
                "expected_index_sha256": __import__("hashlib").sha256(index.read_bytes()).hexdigest(),
                "map_config_path": str(maps),
                "scene_map": {"MapSeen": "travel_MapSeen"},
                "execution_mode": "managed", "save_artifacts": True, "enable_action_planner_debug": True,
            }
        )
    )
    validated = runner.prepare_evaluation(require_execution_ready=False)
    assert validated.execution_readiness["missing_maps"] == ["Other"]
    with pytest.raises(ValueError, match="not execution-ready"):
        runner.prepare_evaluation(require_execution_ready=True)


def test_scene_start_error_is_preserved_as_infrastructure_abort_reason():
    scene_error = {
        "error_type": "RuntimeError",
        "error": "SSH command failed before scene capabilities were available",
    }
    result = _infrastructure_error(
        execution=SimpleNamespace(
            outcome=SimpleNamespace(error=scene_error),
        ),
        setup_error=None,
        capabilities=None,
        cleanup_errors=[],
    )

    assert result == scene_error


def _runner(tmp_path, execution_mode="managed", with_sha=True):
    index = write_index(
        tmp_path / "index.json",
        [case("easy", length=100), case("hard", length=300)],
    )
    maps = write_map_config(tmp_path / "maps.json")
    return TravelUavBenchmarkRunner(
        TravelUavEvalConfig.from_dict(
            {**{"execution_mode": "managed", "save_artifacts": True, "enable_action_planner_debug": True},
                "index_path": str(index),
                "map_config_path": str(maps),
                "scene_map": {"MapSeen": "travel_MapSeen"},
                "output_dir": str(tmp_path / "outputs"),
                "run_root": str(tmp_path / "runs"),
                "execution_mode": execution_mode,
                "expected_index_sha256": __import__("hashlib").sha256(index.read_bytes()).hexdigest() if with_sha else None,
                "limit_per_difficulty": None,
                "generate_html_review": True,
                "save_artifacts": execution_mode == "managed",
                "enable_action_planner_debug": execution_mode == "managed",
            }
        ),
        planner_factory=lambda model: CompletePlanner(),
    )
