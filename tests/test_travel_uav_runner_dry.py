"""Dry TravelUAV runner integration and readiness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.protocol import TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner
from src.agent.phase import COMPLETE_TASK, PhaseTransition
from src.agent.planning.schemas import CoupledActionPhasePlan
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.validator import semantic_plan_from_dict
from tests.travel_uav_test_support import case, write_index, write_map_config


class FakeParser:
    protocol_context = None

    def __init__(self, protocol_context=None):
        type(self).protocol_context = protocol_context

    def parse(self, instruction: str) -> TaskParseResult:
        return TaskParseResult(
            instruction=instruction,
            parser_mode="fake",
            plan=semantic_plan_from_dict(
                {
                    "intent": "target navigation",
                    "task_goal": {
                        "target": "described object",
                        "target_attributes": {},
                        "final_state": "target region confirmed",
                    },
                    "phases": [
                        {
                            "name": "find_target",
                            "goal": "confirm the described target region",
                            "done_when": ["target region is confirmed"],
                            "continue_when": ["target region is not confirmed"],
                        }
                    ],
                }
            ),
        )


class CompletePlanner:
    planner_mode = "fake"

    def plan_next_with_phase_decision(self, state, context):
        return CoupledActionPhasePlan(
            phase=state.current_phase_name,
            phase_decision=PhaseTransition(COMPLETE_TASK, "fixture complete"),
            steps=[],
            agent_messages=[],
        )


def test_validate_only_writes_provenance_without_loading_model(tmp_path):
    runner = _runner(tmp_path)
    result = runner.validate(eval_id="validate")

    assert result["case_selection"]["selected_case_ids"] == ["easy", "hard"]
    assert Path(result["validation_paths"]["dataset_provenance"]).is_file()
    assert runner._model_config_value is None


def test_dry_runner_uses_protocol_and_generates_core_html(tmp_path):
    runner = _runner(tmp_path)
    runner._task_parser_factory = lambda model: FakeParser(
        TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
    )
    runner._planner_factory = lambda model: CompletePlanner()

    result = runner.run(eval_id="dry")

    assert result["summary"]["batch_status"] == "completed"
    assert len(result["results"]) == 2
    assert FakeParser.protocol_context == TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
    assert Path(result["summary_paths"]["html_review"]).is_file()
    for item in result["results"]:
        assert (Path(item["case_dir"]) / "review.html").is_file()
        assert (Path(item["case_dir"]) / "audit.json").is_file()


def test_real_mode_rejects_missing_index_sha_before_simulator(tmp_path):
    runner = _runner(tmp_path, execution_mode="managed")
    with pytest.raises(ValueError, match="expected_index_sha256"):
        runner.prepare_evaluation(require_execution_ready=True)


def test_missing_map_is_reported_offline_and_rejected_for_run(tmp_path):
    index = write_index(tmp_path / "index.json", [case("missing", map_name="Other")])
    maps = write_map_config(tmp_path / "maps.json")
    runner = TravelUavBenchmarkRunner(
        TravelUavEvalConfig.from_dict(
            {
                "index_path": str(index),
                "map_config_path": str(maps),
                "scene_map": {"MapSeen": "travel_MapSeen"},
                "execution_mode": "dry",
            }
        )
    )
    validated = runner.prepare_evaluation(require_execution_ready=False)
    assert validated.execution_readiness["missing_maps"] == ["Other"]
    with pytest.raises(ValueError, match="not execution-ready"):
        runner.prepare_evaluation(require_execution_ready=True)


def _runner(tmp_path, execution_mode="dry"):
    index = write_index(
        tmp_path / "index.json",
        [case("easy", length=100), case("hard", length=300)],
    )
    maps = write_map_config(tmp_path / "maps.json")
    return TravelUavBenchmarkRunner(
        TravelUavEvalConfig.from_dict(
            {
                "index_path": str(index),
                "map_config_path": str(maps),
                "scene_map": {"MapSeen": "travel_MapSeen"},
                "output_dir": str(tmp_path / "outputs"),
                "run_root": str(tmp_path / "runs"),
                "execution_mode": execution_mode,
                "limit_per_difficulty": None,
                "generate_html_review": True,
                "save_artifacts": execution_mode == "managed",
                "enable_action_planner_debug": execution_mode == "managed",
            }
        ),
        task_parser_factory=lambda model: FakeParser(
            TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
        ),
        planner_factory=lambda model: CompletePlanner(),
    )
