"""Tests for UAV-Flow static reports and leak checks."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmarks.uav_flow.reports import (
    evaluation_truth_leak_check,
    failure_counts,
    run_artifact_summary,
    write_case_review,
    write_eval_summary,
    write_trajectory_plots,
)
from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.html_review import write_html_review


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


def test_leak_check_flags_evaluation_truth_terms(tmp_path):
    context_path = tmp_path / "contexts" / "latest_context.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_text(json.dumps({"safe": True, "target_pos": [1, 2, 3]}), encoding="utf-8")

    result = evaluation_truth_leak_check(tmp_path)

    assert result["status"] == "failed"
    assert result["leaked_terms"] == ["target_pos"]


def test_run_artifact_summary_extracts_rgb_actions_and_agent_messages(tmp_path):
    rgb = tmp_path / "artifacts" / "rgb" / "frame_000001.png"
    rgb.parent.mkdir(parents=True)
    rgb.write_bytes(b"png")
    events = [
        {"event": "tool_step_completed", "payload": {"phase": "p", "tool": "uav.hover", "ok": True}},
        {
            "event": "agent_message",
            "payload": {"message": {"type": "progress", "content": "ok"}},
        },
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")

    summary = run_artifact_summary(tmp_path)

    assert summary["first_rgb"] == str(rgb)
    assert summary["actions"][0]["tool"] == "uav.hover"
    assert summary["agent_messages"][0]["message"]["content"] == "ok"


def test_run_artifact_summary_adds_motion_deltas(tmp_path):
    events = [
        {"event": "tool_step_completed", "payload": {"phase": "p", "step_index": 0, "tool": "uav.move_body", "ok": True}},
    ]
    actions = [
        {
            "id": "act_0001",
            "tool": "uav.move_body",
            "args": {"direction": "forward", "duration": 1.0},
            "result": {"ok": True},
        }
    ]
    poses = [
        {"id": "pose_0001", "source": "observation", "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}},
        {
            "id": "pose_0002",
            "source": "action_result",
            "action_id": "act_0001",
            "pose": {"x": 2.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
        },
    ]
    (tmp_path / "memory").mkdir()
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")
    (tmp_path / "memory" / "actions.jsonl").write_text("\n".join(json.dumps(item) for item in actions), encoding="utf-8")
    (tmp_path / "memory" / "poses.jsonl").write_text("\n".join(json.dumps(item) for item in poses), encoding="utf-8")

    summary = run_artifact_summary(tmp_path)

    assert summary["actions"][0]["is_move"] is True
    assert summary["actions"][0]["expected_delta_m"]["dx"] == 2.0
    assert summary["actions"][0]["actual_delta_m"]["dx"] == 2.0
    assert summary["actions"][0]["actual_planar_m"] == 2.0


def test_case_review_and_summary_are_text_only(tmp_path):
    payload = {
        "case": {"case_id": "c1", "selected_instruction": "Turn", "class_name": "Turn"},
        "run_dir": "runs/c1",
        "status": "ok",
        "completed": True,
        "stopped": False,
        "stopped_reason": None,
        "run_artifacts": {
            "first_rgb": "frame_1.png",
            "last_rgb": "frame_2.png",
            "rgb_count": 2,
            "actions": [{"tool": "uav.hover", "ok": True, "phase": "p"}],
            "agent_messages": [{"message": {"type": "final", "content": "done"}}],
        },
        "ndtw": {
            "status": "ok",
            "ndtw": 1.0,
            "metric_interpretation": {
                "mode": "rotation_sequence",
                "position_zeroed": True,
                "sample_step": 2,
                "summary": "Turn/Rotate nDTW follows the original UAV-Flow metric.",
            },
        },
        "sr_judge": {"status": "skipped"},
        "evaluation_truth_leak_check": {"status": "passed", "leaked_terms": []},
    }

    review_path = write_case_review(tmp_path, payload)
    paths = write_eval_summary(tmp_path, [payload | {"case_id": "c1", "class_name": "Turn"}], ["ok"])

    review_text = Path(review_path).read_text(encoding="utf-8")
    summary_text = Path(paths["summary_md"]).read_text(encoding="utf-8")
    assert "data:image/png;base64" not in review_text
    assert "- Stopped: False" in review_text
    assert "Metric mode: rotation_sequence position_zeroed=True sample_step=2" in review_text
    assert "Failure Counts" in summary_text
    assert "Metric Notes" in summary_text
    assert "Turn/Rotate use the original UAV-Flow rotation-sequence nDTW mode" in summary_text
    assert "sr_skipped" in summary_text
    assert not any(pattern in review_text + summary_text for pattern in ["�", "锟", "閿"])


def test_html_review_combines_case_metrics_rgb_actions_and_messages(tmp_path):
    eval_dir = tmp_path / "eval"
    case_dir = eval_dir / "cases" / "c1"
    run_dir = tmp_path / "runs" / "c1"
    rgb_dir = run_dir / "artifacts" / "rgb"
    rgb_dir.mkdir(parents=True)
    case_dir.mkdir(parents=True)
    for name in ["frame_000001.png", "frame_000002.png", "frame_000003.png"]:
        (rgb_dir / name).write_bytes(b"png")
    (run_dir / "memory").mkdir()
    (run_dir / "memory" / "agent_messages.jsonl").write_text(
        json.dumps(
            {
                "source": "action_planner",
                "phase": "p",
                "message": {"type": "progress", "content": "我来自原始JSONL。", "confidence": "high"},
                "message_ref": {"timestamp": "2026-06-05T00:00:00Z"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "case": {
                    "case_id": "c1",
                    "selected_instruction": "Turn to the direction of the person.",
                    "class_name": "Turn",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "metrics": {
            "case_count": 1,
            "evaluated_count": 1,
            "overall_mean_ndtw": 0.42,
            "by_class": {"Turn": {"evaluated_count": 1, "mean_ndtw": 0.42}},
        },
        "failure_counts": {"runtime_failed": 0},
        "cases": [
            {
                "case_id": "c1",
                "class_name": "Turn",
                "status": "ok",
                "completed": True,
                "case_dir": str(case_dir),
                "run_dir": str(run_dir),
                "review_path": str(case_dir / "review.md"),
                "run_artifacts": {
                    "rgb_count": 3,
                    "actions": [
                        {
                            "step_index": 0,
                            "tool": "uav.turn_relative",
                            "phase": "p",
                            "args": {"yaw_deg": -10},
                            "ok": True,
                            "actual_delta_m": {"dyaw_deg": -10},
                        }
                    ],
                    "agent_messages": [
                        {
                            "source": "action_planner",
                            "phase": "p",
                            "message": {"type": "progress", "content": "summary fallback should not win", "confidence": "high"},
                            "message_ref": {"timestamp": "2026-06-05T00:00:00Z"},
                        }
                    ],
                },
                "ndtw": {"status": "ok", "ndtw": 0.42, "metric_interpretation": {"mode": "rotation_sequence"}},
                "sr_judge": {"status": "ok", "success": True, "confidence": 0.8},
                "visual_diagnostics": {"status": "ok", "enabled": True, "judged_count": 3, "frame_count": 3},
                "plot_result": {"trajectory_2d": str(case_dir / "plots" / "trajectory_2d.png")},
                "evaluation_truth_leak_check": {"status": "passed"},
            }
        ],
    }

    paths = write_html_review(eval_dir, summary)

    index = Path(paths["html_review"]).read_text(encoding="utf-8")
    case_html = (case_dir / "review.html").read_text(encoding="utf-8")
    assert '<meta charset="utf-8">' in index
    assert "c1" in index
    assert "0.42" in index
    assert "Case Evidence" in index
    assert "Initial" in index
    assert "After first action" in index
    assert "Before" in index
    assert "After" in index
    assert "Turn to the direction of the person." in index
    assert "uav.turn_relative" in index
    assert "我来自原始JSONL。" in index
    assert "artifacts/rgb/frame_000001.png" in index
    assert "uav.turn_relative" in case_html
    assert "我来自原始JSONL。" in case_html
    assert "summary fallback should not win" not in case_html
    assert "frame_000002.png" in case_html
    assert "Before RGB" in case_html
    assert "After RGB" in case_html
    assert "Observation after action #0" in case_html
    assert "artifacts/rgb/frame_000001.png" in case_html
    assert "data:image/png;base64" not in case_html


def test_html_review_handles_missing_rgb_actions_and_messages(tmp_path):
    eval_dir = tmp_path / "eval"
    case_dir = eval_dir / "cases" / "empty"
    summary = {
        "metrics": {"case_count": 1},
        "failure_counts": {},
        "cases": [
            {
                "case_id": "empty",
                "class_name": "Move",
                "status": "ok",
                "case_dir": str(case_dir),
                "run_dir": str(tmp_path / "missing_run"),
                "run_artifacts": {},
                "ndtw": {},
                "sr_judge": {},
            }
        ],
    }

    paths = write_html_review(eval_dir, summary)

    assert Path(paths["html_review"]).exists()
    case_html = (case_dir / "review.html").read_text(encoding="utf-8")
    assert "No RGB frames found." in case_html
    assert "No tool actions recorded." in case_html
    assert "No agent messages recorded." in case_html


def test_html_review_marks_missing_after_frame_without_reusing_last_rgb(tmp_path):
    eval_dir = tmp_path / "eval"
    case_dir = eval_dir / "cases" / "short"
    run_dir = tmp_path / "runs" / "short"
    rgb_dir = run_dir / "artifacts" / "rgb"
    rgb_dir.mkdir(parents=True)
    case_dir.mkdir(parents=True)
    (rgb_dir / "frame_000001.png").write_bytes(b"png")
    summary = {
        "metrics": {"case_count": 1},
        "failure_counts": {},
        "cases": [
            {
                "case_id": "short",
                "class_name": "Turn",
                "status": "ok",
                "case_dir": str(case_dir),
                "run_dir": str(run_dir),
                "run_artifacts": {
                    "rgb_count": 1,
                    "actions": [
                        {
                            "step_index": 0,
                            "tool": "uav.turn_relative",
                            "phase": "p",
                            "args": {"yaw_deg": 5},
                            "ok": True,
                        }
                    ],
                },
                "ndtw": {},
                "sr_judge": {},
            }
        ],
    }

    paths = write_html_review(eval_dir, summary)

    index = Path(paths["html_review"]).read_text(encoding="utf-8")
    case_html = (case_dir / "review.html").read_text(encoding="utf-8")
    assert "frame_000001.png" in index
    assert "missing" in index
    assert "missing" in case_html
    assert case_html.count("frame_000001.png") >= 1


def test_failure_counts_classifies_common_failures():
    counts = failure_counts(
        [
            {"status": "error", "error": {"error": "gym env failed"}, "ndtw": {"status": "skipped"}, "sr_judge": {"status": "skipped"}},
            {"status": "invalid_context_leak", "ndtw": {"status": "ok"}, "sr_judge": {"status": "ok"}},
        ]
    )

    assert counts["env_start_failed"] == 1
    assert counts["metric_skipped"] == 1
    assert counts["sr_skipped"] == 1
    assert counts["invalid_context_leak"] == 1


def test_failure_counts_tracks_runtime_loop_guard_stop():
    counts = failure_counts(
        [
            {
                "status": "stopped",
                "stopped_reason": "runtime_loop_guard_reached",
                "ndtw": {"status": "ok"},
                "sr_judge": {"status": "skipped"},
            }
        ]
    )

    assert counts["runtime_loop_guard_reached"] == 1


def test_trajectory_plots_include_reference_and_actual_paths(tmp_path):
    case = make_case()
    trajectory_path = tmp_path / "uav_agent_trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            [
                {"state": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
                {"state": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]},
            ]
        ),
        encoding="utf-8",
    )

    result = write_trajectory_plots(tmp_path, trajectory_path, case)

    if result["status"] == "skipped":
        assert result["reason"] == "matplotlib_unavailable"
        return
    assert result["status"] == "ok"
    assert result["has_reference_path"] is True
    assert result["reference_point_count"] == len(case.reference_path_preprocessed)
    assert Path(result["trajectory_2d"]).exists()
    assert Path(result["trajectory_3d"]).exists()
