"""Tests for UAV-Flow nDTW metrics."""

from __future__ import annotations

import json

from evaluation.benchmarks.uav_flow.metrics import (
    dtw_distance,
    evaluate_case_ndtw,
    metric_interpretation_for_class,
    ndtw,
    summarize_metrics,
)


def test_dtw_and_ndtw_identical_paths_are_perfect():
    vecs = [[0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1]]

    dist = dtw_distance(vecs, vecs)

    assert dist == 0.0
    assert ndtw(dist, 1.0) == 1.0


def test_evaluate_case_ndtw_turn_zeroes_position(tmp_path):
    trajectory = tmp_path / "traj.json"
    trajectory.write_text(
        json.dumps(
            [
                {"state": [[999, 999, 0], [0, 0, 0]]},
                {"state": [[999, 999, 0], [0, 10, 0]]},
                {"state": [[999, 999, 0], [0, 20, 0]]},
            ]
        ),
        encoding="utf-8",
    )
    ref = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 10, 0], [0, 0, 0, 0, 20, 0]]

    result = evaluate_case_ndtw(trajectory, ref, class_name="Turn")

    assert result["status"] == "ok"
    assert result["zero_pos"] is True
    assert result["sample_step"] == 2
    assert result["metric_interpretation"]["mode"] == "rotation_sequence"
    assert result["metric_interpretation"]["position_zeroed"] is True
    assert result["ndtw"] is not None


def test_metric_interpretation_distinguishes_rotation_and_translation_classes():
    turn = metric_interpretation_for_class("Turn", sample_step=2)
    rotate = metric_interpretation_for_class("Rotate", sample_step=5)
    move = metric_interpretation_for_class("Move", sample_step=2)

    assert turn["mode"] == "rotation_sequence"
    assert turn["position_zeroed"] is True
    assert rotate["mode"] == "rotation_sequence"
    assert rotate["position_zeroed"] is True
    assert move["mode"] == "position_and_rotation_sequence"
    assert move["position_zeroed"] is False


def test_missing_reference_is_skipped(tmp_path):
    trajectory = tmp_path / "traj.json"
    trajectory.write_text("[]", encoding="utf-8")

    result = evaluate_case_ndtw(trajectory, [], class_name="Move")

    assert result["status"] == "skipped"
    assert result["reason"] == "reference_path_missing"
    assert result["metric_interpretation"]["mode"] == "position_and_rotation_sequence"


def test_summarize_metrics_groups_by_class():
    summary = summarize_metrics(
        [
            {"class_name": "Turn", "ndtw": {"ndtw": 0.5}},
            {"class_name": "Turn", "ndtw": {"ndtw": 1.0}},
            {"class_name": "Move", "ndtw": {"status": "skipped", "ndtw": None}},
        ]
    )

    assert summary["case_count"] == 3
    assert summary["evaluated_count"] == 2
    assert summary["by_class"]["Turn"]["mean_ndtw"] == 0.75
