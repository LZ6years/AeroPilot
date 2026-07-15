"""Tests for the configurable UAV-Flow STMR native-report canary."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parent / "manual" / "uav_flow_eval" / "run_stmr_canary.py"
SPEC = importlib.util.spec_from_file_location("run_stmr_canary", SCRIPT)
assert SPEC and SPEC.loader
canary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canary)


def test_csv_values_supports_subset_and_rejects_unknown_mode():
    assert canary._csv_values("Approach") == ["Approach"]
    assert canary._csv_values("off,planner,off", allowed=set(canary.VALID_MODES)) == ["off", "planner"]
    with pytest.raises(ValueError, match="Unsupported"):
        canary._csv_values("off,bogus", allowed=set(canary.VALID_MODES))


def test_canary_defaults_to_planner_and_has_no_comparison_output():
    args = canary._parser().parse_args([])

    assert args.modes == "planner"
    source = SCRIPT.read_text(encoding="utf-8")
    assert "comparison.html" not in source
    assert "comparison.json" not in source
    assert "comparison.md" not in source


def test_run_summary_returns_native_batch_and_case_reviews(tmp_path):
    eval_dir = tmp_path / "planner"
    case_dir = eval_dir / "cases" / "approach_case"
    result = {
        "eval_dir": str(eval_dir),
        "summary_paths": {
            "html_review": str(eval_dir / "review.html"),
            "summary_json": str(eval_dir / "summary.json"),
        },
        "results": [
            {
                "case_id": "approach_case",
                "class_name": "Approach",
                "status": "ok",
                "completed": True,
                "case_dir": str(case_dir),
            }
        ],
    }

    summary = canary._summarize_run(result)

    assert summary["review_html"] == str(eval_dir / "review.html")
    assert summary["case_reviews"] == [
        {
            "case_id": "approach_case",
            "class_name": "Approach",
            "status": "ok",
            "completed": True,
            "review_html": str(case_dir / "review.html"),
        }
    ]
