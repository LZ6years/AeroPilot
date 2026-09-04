"""Tests for the read-only DoneWhen experiment comparison utility."""

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parent / "manual" / "done_when_flow_compare.py"
SPEC = importlib.util.spec_from_file_location("done_when_flow_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_case(root, case_id, *, status="stopped", ndtw=0.25):
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "case": {"class_name": "Approach"},
                "status": status,
                "stopped_reason": "runtime_loop_guard_reached",
                "ndtw": {"ndtw": ndtw},
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "audit.json").write_text(
        json.dumps(
            {
                "planner_debug": [{"current_phase": {"done_when": ["target remains visible in view"]}}],
                "events": [],
                "debug_events": [{"event": "adaptive_done_when_kept"}],
                "assessment": {
                    "done_when": [
                        {"index": 0, "status": "unknown", "evidence": "no exact distance measurement"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_and_compare_finalized_outputs(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_case(left, "case-1", ndtw=0.2)
    write_case(right, "case-1", ndtw=0.4)
    (left / "cases" / "partial").mkdir(parents=True)

    report = MODULE.compare_runs(
        [MODULE.load_run("A", left), MODULE.load_run("B", right)]
    )

    assert report["same_case_ids"] is True
    assert report["common_case_ids"] == ["case-1"]
    run = report["runs"][0]
    assert run["finalized_case_count"] == 1
    assert run["mean_ndtw"] == 0.2
    assert run["loop_guard_count"] == 1
    assert run["unstable_terminal_condition_count"] >= 1
    assert run["ideal_evidence_unknown_count"] == 1
    assert run["adaptive_kept_count"] == 1
    assert run["skipped"] == [{"case_id": "partial", "reason": "missing case.json or audit.json"}]
