"""Tests for UAV-Flow benchmark case loading."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmarks.uav_flow.cases import UavFlowCase, load_cases


def write_case(path: Path, instruction: str = "Turn to the person") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "instruction": instruction,
                "instruction_unified": "Turn to the direction of the person",
                "initial_pos": [10, 20, 30, 0, 90, 0],
                "target_pos": [20, 20, 30, 0, 0, 0],
                "obj_id": 19,
                "use_obj": 1,
                "reference_path_preprocessed": [[0, 0, 0, 0, 0, 0], [0, -10, 0, 0, -5, 0]],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_load_case_keeps_evaluation_truth_out_of_planner_metadata(tmp_path):
    case_path = tmp_path / "test_jsons" / "case_a.json"
    write_case(case_path)

    case = UavFlowCase.from_file(case_path, class_name="Turn")

    assert case.case_id == "case_a"
    assert case.class_name == "Turn"
    assert case.selected_instruction == "Turn to the direction of the person"
    planner_safe = case.planner_safe_metadata()
    assert "target_pos" not in planner_safe
    assert "reference_path_preprocessed" not in planner_safe
    assert case.evaluation_truth()["target_pos"] == [20.0, 20.0, 30.0, 0.0, 0.0, 0.0]


def test_load_cases_uses_classification_and_limit(tmp_path):
    write_case(tmp_path / "test_jsons" / "turn_a.json")
    write_case(tmp_path / "test_jsons" / "turn_b.json")
    write_case(tmp_path / "test_jsons" / "move_a.json", instruction="Move forward")
    (tmp_path / "classified_instr.json").write_text(
        json.dumps({"Turn": ["turn_a.json", "turn_b.json"], "Move": ["move_a.json"]}),
        encoding="utf-8",
    )

    cases = load_cases(tmp_path, classes=["Turn", "Move"], limit_per_class=1)

    assert [case.file_name for case in cases] == ["turn_a.json", "move_a.json"]
    assert [case.class_name for case in cases] == ["Turn", "Move"]


def test_invalid_case_requires_initial_pos(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"instruction": "bad", "initial_pos": [1, 2]}), encoding="utf-8")

    try:
        UavFlowCase.from_file(path)
    except ValueError as exc:
        assert "initial_pos" in str(exc)
    else:
        raise AssertionError("expected invalid initial_pos to fail")

