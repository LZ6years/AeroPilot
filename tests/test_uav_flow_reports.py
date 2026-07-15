"""Tests for UAV-Flow static reports."""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.benchmarks.uav_flow.reports import (
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
    debug_dir = run_dir / "debug" / "action_planner"
    debug_dir.mkdir(parents=True)
    (debug_dir / "p_window_000.json").write_text(
        json.dumps(
            {
                "phase": "p",
                "window_index": 0,
                "planner_context": {
                    "active_instruction": "Turn to the direction of the person.",
                    "current_phase": {"name": "p"},
                    "safety_constraints": {"yaw_scan_remaining_deg": 15.0},
                    "observation_policy": {"mode": "automatic"},
                    "phase_metrics": [],
                    "phase_execution_summary": {"window_count": 0},
                    "fragments": [
                        {
                            "name": "image",
                            "metadata": {
                                "available_skill_names": ["skill_selection_debug_marker"],
                                "selection": {"status": "debug_only"},
                            },
                            "model_inputs": [
                                {
                                    "type": "image",
                                    "role": "latest_observation",
                                    "data": "<omitted encoded image data: 123 chars>",
                                }
                            ],
                        },
                        {
                            "name": "action",
                            "content": {
                                "action_trend": {
                                    "motion_pattern": {"type": "alternating_yaw"},
                                }
                            },
                        },
                    ],
                },
                "model_context": {
                    "active_instruction": "Turn to the direction of the person.",
                    "current_phase": {"name": "p"},
                    "phase_metrics": [],
                    "phase_progress_evidence": {
                        "motion_progress": {
                            "has_translation": False,
                            "yaw_only_so_far": True,
                            "forward_m": 0.0,
                            "planar_m": 0.0,
                            "yaw_delta_deg": 10.0,
                            "move_duration_success_sum_sec": 0.0,
                            "tools_executed": {"uav.turn_relative": 1},
                        }
                    },
                    "phase_execution_summary": {"window_count": 0},
                    "fragments": [
                        {
                            "name": "image",
                            "model_inputs": [
                                {
                                    "type": "image",
                                    "input_id": "image_1_current_observation",
                                    "role": "current_observation",
                                    "usage": "Use as the current visual evidence for this decision.",
                                    "data": "<omitted encoded image data: 123 chars>",
                                }
                            ],
                        },
                        {
                            "name": "skills",
                            "content": {
                                "selected_skills": [
                                    {
                                        "name": "visual_target_task_reasoning",
                                        "description": "Use visual evidence for target alignment.",
                                        "summary": "Selector-only summary should not appear in model-facing context.",
                                        "skill_text": "# Visual Target Task Reasoning\n\nBefore choosing a visual-target action, check whether the target is left, right, or centered.",
                                        "skill_text_truncated": False,
                                    }
                                ]
                            },
                        },
                        {
                            "name": "action",
                            "content": {
                                "action_trend": {
                                    "motion_pattern": {"type": "alternating_yaw"},
                                }
                            },
                        },
                    ],
                },
                "raw_plan": {
                    "action_window": {"intent": "debug raw"},
                    "agent_messages": [{"content": "raw message"}],
                    "steps": [{"kind": "tool", "tool": "uav.turn_relative", "args": {"yaw_deg": 5}}],
                },
                    "safe_plan": {
                        "action_window": {"intent": "debug safe"},
                        "agent_messages": [{"content": "safe message"}],
                        "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}}],
                    },
                    "coupled_phase_decision": {
                        "decision": "continue_current_phase",
                        "reason": "target still unclear",
                    },
                    "safety_decision": {"interventions": [{"reason": "test"}], "plan_changed": True},
                },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "coupled_planner_phase_decided",
                        "payload": {
                            "phase": "p",
                            "decision": "continue_current_phase",
                            "reason": "target still unclear",
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "event": "phase_transition_continued",
                        "payload": {
                            "phase": "p",
                            "decision": "continue_current_phase",
                            "reason": "target still unclear",
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
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
    assert "Windows" in index
    assert "Flags" in index
    assert "alternating yaw" in index
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
    assert "Decision Timeline" in case_html
    assert "Tool Timeline" in case_html
    assert "Tool execution" in case_html
    assert "Executed uav.hover" in case_html
    assert "Coupled Planner Debug" in case_html
    assert "Decision Summary" in case_html
    assert "Model-Facing Context" in case_html
    assert "Runtime Applied Phase State" in case_html
    assert "Runtime Applied Phase State Events" in case_html
    assert "coupled_planner_phase_decided" in case_html
    assert "phase_transition_continued" in case_html
    assert "Coupled mode: one model call made both the phase decision and action decision" in case_html
    assert "Judge Context Summary" not in case_html
    assert "Executed Window" not in case_html
    assert "open full phase decision json" not in case_html
    phase_model_blocks = re.findall(r"<summary>Model-Facing Context</summary>\s*<pre>(.*?)</pre>", case_html, flags=re.S)
    assert all("raw action window should stay out of model-facing block" not in block for block in phase_model_blocks)
    assert all("yaw_scan_remaining_deg" not in block for block in phase_model_blocks)
    assert "debug raw" in case_html
    assert "debug safe" in case_html
    assert "image_1_current_observation" in case_html
    assert "current_observation" in case_html
    assert "alternating_yaw" in case_html
    assert "alternating yaw" in case_html
    assert "yaw-only so far" in case_html
    assert "Phase progress evidence" in case_html
    assert "Selected skill guidance" in case_html
    assert "visual_target_task_reasoning" in case_html
    assert "skill_text_chars" in case_html
    assert "skill_text_truncated" in case_html
    assert "Before choosing a visual-target action" in case_html
    assert "Selector-only summary should not appear in model-facing context" not in case_html
    assert "yaw_only_so_far" in case_html
    assert "has_translation" in case_html
    assert "yaw_scan_remaining_deg" not in case_html
    assert "skill_selection_debug_marker" not in case_html
    assert "phase_judge_debug_marker" not in case_html
    assert "data:image/png;base64" not in case_html
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


def test_html_review_embeds_closed_stmr_render_history_panel(tmp_path):
    eval_dir = tmp_path / "eval"
    case_dir = eval_dir / "cases" / "stmr"
    run_dir = tmp_path / "runs" / "stmr"
    stmr_dir = run_dir / "contexts" / "stmr_map"
    snapshot_dir = stmr_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    case_dir.mkdir(parents=True)
    latest = stmr_dir / "latest_stmr_map.png"
    latest.write_bytes(b"png")
    latest_json = stmr_dir / "latest_stmr_map.json"
    latest_json.write_text("{}", encoding="utf-8")
    snapshots = []
    for index in range(1, 11):
        path = snapshot_dir / f"{index:04d}_obs_{index:04d}.png"
        path.write_bytes(b"png")
        snapshots.append(
            {
                "sequence": index,
                "observation_id": f"obs_{index:04d}",
                "processed_observation_count": index,
                "point_count": index * 100,
                "voxel_count": index * 10,
                "png_path": str(path),
            }
        )
    manifest = stmr_dir / "render_history.json"
    manifest.write_text(
        json.dumps({"schema_version": "stmr_render_history_v1", "snapshot_count": 10, "snapshots": snapshots}),
        encoding="utf-8",
    )
    summary = {
        "metrics": {"case_count": 1},
        "failure_counts": {},
        "cases": [
            {
                "case_id": "stmr",
                "class_name": "Approach",
                "status": "ok",
                "case_dir": str(case_dir),
                "run_dir": str(run_dir),
                "run_artifacts": {},
                "ndtw": {},
                "sr_judge": {},
                "stmr": {
                    "mode": "planner",
                    "status": "ready",
                    "png_path": str(latest),
                    "json_path": str(latest_json),
                    "processed_observation_count": 10,
                    "failed_observation_count": 0,
                    "point_count": 1000,
                    "voxel_count": 100,
                    "semantic_cell_counts": {"route_surface": 12},
                    "batch_incremental_parity": {"equivalent": True},
                    "render_history": {"snapshot_count": 10, "manifest_path": str(manifest)},
                    "profiling": {
                        "stages": {
                            "stmr_incremental_projection": {
                                "sample_count": 10,
                                "p50_ms": 12.5,
                                "p95_ms": 20.0,
                            }
                        }
                    },
                },
            }
        ],
    }

    write_html_review(eval_dir, summary)
    case_html = (case_dir / "review.html").read_text(encoding="utf-8")

    assert '<details class="stmr-render-panel">' in case_html
    assert '<details class="stmr-render-panel" open>' not in case_html
    assert "显示/隐藏 STMR 渲染过程" in case_html
    key_section = case_html.split('<details class="stmr-history">', 1)[0].rsplit(
        "<h3>Incremental Key Frames</h3>", 1
    )[1]
    assert key_section.count("<figure>") == 8
    assert "全部 STMR 历史地图 (10)" in case_html
    assert 'loading="lazy"' in case_html
    assert "route_surface" in case_html
    assert "stmr_incremental_projection" in case_html
    assert "12.5" in case_html


def test_html_review_hides_off_stmr_and_falls_back_for_legacy_history(tmp_path):
    eval_dir = tmp_path / "eval"
    off_case_dir = eval_dir / "cases" / "off"
    legacy_case_dir = eval_dir / "cases" / "legacy"
    legacy_map = tmp_path / "legacy_stmr.png"
    legacy_map.write_bytes(b"png")
    summary = {
        "metrics": {"case_count": 2},
        "failure_counts": {},
        "cases": [
            {
                "case_id": "off",
                "class_name": "Move",
                "status": "ok",
                "case_dir": str(off_case_dir),
                "run_artifacts": {},
                "ndtw": {},
                "sr_judge": {},
                "stmr": {"mode": "off", "status": "disabled"},
            },
            {
                "case_id": "legacy",
                "class_name": "Approach",
                "status": "ok",
                "case_dir": str(legacy_case_dir),
                "run_artifacts": {},
                "ndtw": {},
                "sr_judge": {},
                "stmr": {"mode": "planner", "status": "ready", "png_path": str(legacy_map)},
            },
        ],
    }

    write_html_review(eval_dir, summary)
    off_html = (off_case_dir / "review.html").read_text(encoding="utf-8")
    legacy_html = (legacy_case_dir / "review.html").read_text(encoding="utf-8")

    assert "显示/隐藏 STMR 渲染过程" not in off_html
    assert "<dt>STMR</dt>" not in off_html
    assert "显示/隐藏 STMR 渲染过程" in legacy_html
    assert "Legacy run: incremental render history was not recorded." in legacy_html


def test_html_review_displays_coupled_planner_decision(tmp_path):
    eval_dir = tmp_path / "eval"
    case_dir = eval_dir / "cases" / "coupled"
    run_dir = tmp_path / "runs" / "coupled"
    (run_dir / "debug" / "action_planner").mkdir(parents=True)
    case_dir.mkdir(parents=True)
    (run_dir / "debug" / "action_planner" / "p_window_001.json").write_text(
        json.dumps(
            {
                "phase": "p",
                "window_index": 1,
                "model_context": {
                    "active_instruction": "Turn",
                    "current_phase": {"name": "p"},
                    "phase_progress_evidence": {
                        "motion_progress": {
                            "has_translation": False,
                            "yaw_only_so_far": True,
                            "forward_m": 0.0,
                            "planar_m": 0.0,
                            "tools_executed": {"uav.turn_relative": 2},
                        }
                    },
                    "fragments": [],
                },
                "raw_plan": {"steps": [], "action_window": {}, "agent_messages": []},
                "safe_plan": {"steps": [], "action_window": {}, "agent_messages": []},
                "coupled_phase_decision": {
                    "decision": "advance_to_next_phase",
                    "reason": "target already aligned",
                    "agent_messages": [],
                },
                "safety_decision": {"interventions": [], "plan_changed": False},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "coupled_planner_phase_decided",
                        "payload": {"phase": "p", "decision": "advance_to_next_phase", "reason": "target aligned"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "event": "phase_transition_applied",
                        "payload": {
                            "phase": "p",
                            "decision": "advance_to_next_phase",
                            "reason": "target aligned",
                            "transition_source": "coupled_planner",
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / "case.json").write_text(
        json.dumps({"case": {"case_id": "coupled", "selected_instruction": "Turn", "class_name": "Turn"}}),
        encoding="utf-8",
    )
    summary = {
        "metrics": {"case_count": 1},
        "failure_counts": {},
        "cases": [
            {
                "case_id": "coupled",
                "class_name": "Turn",
                "status": "ok",
                "case_dir": str(case_dir),
                "run_dir": str(run_dir),
                "run_artifacts": {"actions": []},
                "ndtw": {},
                "sr_judge": {},
            }
        ],
    }

    write_html_review(eval_dir, summary)

    case_html = (case_dir / "review.html").read_text(encoding="utf-8")
    assert "Coupled Planner Decision" in case_html
    assert "coupled_planner_phase_decided" in case_html
    assert "coupled_planner" in case_html
    assert "No UAV tool is executed" in case_html
    assert "Phase progress evidence" in case_html
    assert "yaw_only_so_far" in case_html


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


def test_html_review_uses_phase_start_observation_anchor(tmp_path):
    eval_dir = tmp_path / "eval"
    case_dir = eval_dir / "cases" / "anchored"
    run_dir = tmp_path / "runs" / "anchored"
    rgb_dir = run_dir / "artifacts" / "rgb"
    memory_dir = run_dir / "memory"
    state_dir = run_dir / "state"
    rgb_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    case_dir.mkdir(parents=True)
    first_frame = rgb_dir / "frame_000001.png"
    task_initial_frame = rgb_dir / "frame_000002.png"
    phase_start_frame = rgb_dir / "frame_000003.png"
    first_frame.write_bytes(b"png")
    task_initial_frame.write_bytes(b"png")
    phase_start_frame.write_bytes(b"png")
    (memory_dir / "frames.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "frame_0001", "observation_id": "obs_0001", "modality": "rgb", "path": str(first_frame)}),
                json.dumps({"id": "frame_0002", "observation_id": "obs_0002", "modality": "rgb", "path": str(task_initial_frame)}),
                json.dumps({"id": "frame_0003", "observation_id": "obs_0003", "modality": "rgb", "path": str(phase_start_frame)}),
            ]
        ),
        encoding="utf-8",
    )
    (memory_dir / "observations.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "obs_0001", "frame_refs": [{"ref": "memory/frames.jsonl#1"}]}),
                json.dumps({"id": "obs_0002", "frame_refs": [{"ref": "memory/frames.jsonl#2"}]}),
                json.dumps({"id": "obs_0003", "frame_refs": [{"ref": "memory/frames.jsonl#3"}]}),
            ]
        ),
        encoding="utf-8",
    )
    (state_dir / "flight_memory.json").write_text(
        json.dumps(
            {
                "observation_anchors": {
                    "task_initial": {"ref": "memory/observations.jsonl#2"},
                    "phase_start": {"ref": "memory/observations.jsonl#3"},
                }
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "metrics": {"case_count": 1},
        "failure_counts": {},
        "cases": [
            {
                "case_id": "anchored",
                "class_name": "Move",
                "status": "ok",
                "case_dir": str(case_dir),
                "run_dir": str(run_dir),
                "run_artifacts": {"rgb_count": 3},
                "ndtw": {},
                "sr_judge": {},
            }
        ],
    }

    paths = write_html_review(eval_dir, summary)

    case_html = (case_dir / "review.html").read_text(encoding="utf-8")
    index = Path(paths["html_review"]).read_text(encoding="utf-8")
    assert "Initial - <code>frame_000003.png</code>" in case_html
    assert "Initial - <code>frame_000003.png</code>" in index


def test_failure_counts_classifies_common_failures():
    counts = failure_counts(
        [
            {"status": "error", "error": {"error": "gym env failed"}, "ndtw": {"status": "skipped"}, "sr_judge": {"status": "skipped"}},
        ]
    )

    assert counts["env_start_failed"] == 1
    assert counts["metric_skipped"] == 1
    assert counts["sr_skipped"] == 1
    assert "invalid_context_leak" not in counts


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
