"""Tests for UAV-Flow report data and Benchmark Core review sections."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.reports import (
    failure_counts,
    write_eval_summary,
    write_trajectory_plots,
)
from evaluation.benchmarks.uav_flow.review import (
    build_uav_flow_review_sections,
)


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
        reference_path_preprocessed=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 10, 0],
        ],
    )


def test_eval_summary_is_text_only(tmp_path):
    payload = {
        "case": {
            "case_id": "c1",
            "selected_instruction": "Turn",
            "class_name": "Turn",
        },
        "run_dir": "runs/c1",
        "status": "ok",
        "completed": True,
        "stopped": False,
        "stopped_reason": None,
        "audit_summary": {
            "rgb_count": 2,
            "action_count": 1,
            "agent_message_count": 1,
        },
        "ndtw": {
            "status": "ok",
            "ndtw": 1.0,
            "metric_interpretation": {
                "mode": "rotation_sequence",
                "position_zeroed": True,
                "sample_step": 2,
                "summary": (
                    "Turn/Rotate nDTW follows the original UAV-Flow metric."
                ),
            },
        },
        "turn_diagnostics": {
            "reference_yaw_delta_deg": -30.0,
            "actual_yaw_delta_deg": -25.0,
            "final_yaw_error_deg": 5.0,
            "translation_drift_m": 0.0,
            "unexpected_translation": False,
            "initial_position_error_m": 0.0,
            "initial_yaw_error_deg": 0.2,
            "initial_pose_within_tolerance": True,
            "backend_turn_tracking": {
                "turn_action_count": 1,
                "observed_count": 1,
                "mean_abs_error_deg": 0.2,
                "max_abs_error_deg": 0.2,
                "all_observed_within_tolerance": True,
                "samples": [],
            },
        },
        "scenario_diagnostics": {
            "contract": {
                "appearance_method": "set_app",
                "camera_follow": "blueprint_pose",
            },
            "target": {"object_name": "BP_Character_21"},
        },
        "sr_judge": {"status": "skipped"},
    }

    paths = write_eval_summary(
        tmp_path,
        [payload | {"case_id": "c1", "class_name": "Turn"}],
        ["ok"],
    )

    summary_text = Path(paths["summary_md"]).read_text(encoding="utf-8")
    assert "data:image/png;base64" not in summary_text
    assert "Failure Counts" in summary_text
    assert "Metric Notes" in summary_text
    assert (
        "Turn/Rotate use the original UAV-Flow rotation-sequence nDTW mode"
        in summary_text
    )
    assert "sr_skipped" in summary_text
    assert not any(pattern in summary_text for pattern in ["�", "锟", "閿"])


def test_uav_flow_review_sections_include_metrics_tracking_and_plots(
    tmp_path,
):
    plot = tmp_path / "trajectory.png"
    plot.write_bytes(b"png")
    sections = build_uav_flow_review_sections(
        case_dir=tmp_path,
        ndtw_result={
            "ndtw": 0.8,
            "metric_interpretation": {"summary": "Move path nDTW."},
        },
        sr_result={"status": "ok", "success": True},
        visual_diagnostics={"status": "ok"},
        stmr_summary={"mode": "off", "status": "disabled"},
        plot_result={"trajectory_2d": str(plot)},
        turn_diagnostics={
            "reference_yaw_delta_deg": 10.0,
            "actual_yaw_delta_deg": 8.0,
            "final_yaw_error_deg": 2.0,
            "translation_drift_m": 0.0,
            "backend_turn_tracking": {
                "turn_action_count": 1,
                "observed_count": 1,
                "unavailable_count": 0,
                "mean_abs_error_deg": 0.2,
                "max_abs_error_deg": 0.2,
                "all_observed_within_tolerance": True,
                "samples": [
                    {
                        "action_id": "act_1",
                        "requested_yaw_delta_deg": 10.0,
                        "observed_yaw_delta_deg": 9.8,
                    }
                ],
            },
        },
        backend_action_tracking={
            "schema_version": "uav_flow_window_tracking_v2",
            "physical_call_count": 1,
            "physical_window_count": 1,
            "motion_call_count": 1,
            "tracked_window_count": 1,
            "observed_window_count": 1,
            "unavailable_window_count": 0,
            "zero_motion_completion": False,
            "max_position_error_m": 0.01,
            "windows": [
                {
                    "decision_id": "decision_0001",
                    "window_index": 1,
                    "position_tracking_error_m": 0.01,
                }
            ],
        },
        scenario_diagnostics={
            "contract": {
                "appearance_method": "set_app",
                "camera_follow": "blueprint_pose",
            }
        },
    )

    by_title = {section.title: section for section in sections}
    assert by_title["Metric Interpretation"].payload == "Move path nDTW."
    assert (
        by_title["Turn Trajectory Diagnostics"].payload[
            "final_yaw_error_deg"
        ]
        == 2.0
    )
    assert (
        by_title["Backend Turn Tracking"].payload["max_abs_error_deg"]
        == 0.2
    )
    assert (
        by_title["Backend Turn Tracking Samples"].payload[0]["action_id"]
        == "act_1"
    )
    assert (
        by_title["Physical Window Tracking"].payload[
            "max_position_error_m"
        ]
        == 0.01
    )
    assert (
        by_title["Physical Window Tracking Details"].payload[0][
            "decision_id"
        ]
        == "decision_0001"
    )
    assert (
        by_title["Scenario Contract Diagnostics"].payload["contract"][
            "camera_follow"
        ]
        == "blueprint_pose"
    )
    assert "STMR" not in by_title
    assert by_title["Trajectory Plots"].payload[0]["path"] == str(plot)


def test_uav_flow_review_materializes_stmr_history(tmp_path):
    stmr_dir = tmp_path / "run" / "contexts" / "stmr_map"
    snapshot_dir = stmr_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    snapshots = []
    for index in range(1, 4):
        image = snapshot_dir / f"{index:04d}.png"
        image.write_bytes(b"png")
        snapshots.append(
            {
                "sequence": index,
                "observation_id": f"obs_{index:04d}",
                "processed_observation_count": index,
                "point_count": index * 100,
                "voxel_count": index * 10,
                "png_path": str(image),
            }
        )
    manifest = stmr_dir / "render_history.json"
    manifest.write_text(
        json.dumps({"snapshots": snapshots}),
        encoding="utf-8",
    )
    latest = stmr_dir / "latest.png"
    latest.write_bytes(b"png")

    sections = build_uav_flow_review_sections(
        case_dir=tmp_path / "case",
        ndtw_result={},
        sr_result={},
        visual_diagnostics={},
        stmr_summary={
            "mode": "planner",
            "status": "ready",
            "png_path": str(latest),
            "render_history": {
                "snapshot_count": 3,
                "manifest_path": str(manifest),
            },
            "profiling": {
                "stages": {
                    "stmr_incremental_projection": {
                        "sample_count": 3,
                        "p50_ms": 12.5,
                        "p95_ms": 20.0,
                    }
                }
            },
        },
        plot_result={},
    )

    by_title = {section.title: section for section in sections}
    images = by_title["STMR Render History"].payload
    assert len(images) == 4
    assert all(Path(item["path"]).is_file() for item in images)
    assert (
        "review_assets/stmr"
        in Path(images[0]["path"]).as_posix()
    )
    profiling = by_title["STMR Profiling"].payload
    assert profiling[0]["stage"] == "stmr_incremental_projection"
    assert profiling[0]["p50_ms"] == 12.5


def test_failure_counts_classifies_common_failures():
    counts = failure_counts(
        [
            {
                "status": "error",
                "error": {"error": "UnrealCV scene failed"},
                "ndtw": {"status": "skipped"},
                "sr_judge": {"status": "skipped"},
            },
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
    assert result["reference_point_count"] == len(
        case.reference_path_preprocessed
    )
    assert Path(result["trajectory_2d"]).exists()
    assert Path(result["trajectory_3d"]).exists()
