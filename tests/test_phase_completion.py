"""Tests for deterministic phase completion evidence."""

from __future__ import annotations

from src.agent.phase import evaluate_phase_completion


def test_phase_completion_marks_within_tolerance():
    result = evaluate_phase_completion(
        [{"kind": "translation", "axis": "forward", "target": 5.0, "unit": "m", "tolerance": 1.0}],
        {"phase_local_measurements": {"forward_displacement_m": 5.4}},
    )

    assert result["all_within_tolerance"] is True
    assert result["evaluations"][0]["status"] == "within_tolerance"


def test_phase_completion_marks_below_target():
    result = evaluate_phase_completion(
        [{"kind": "rotation", "axis": "yaw", "target": 40.0, "unit": "deg", "tolerance": 8.0}],
        {"phase_local_measurements": {"yaw_delta_deg": 20.0}},
    )

    assert result["all_within_tolerance"] is False
    assert result["evaluations"][0]["status"] == "below_target"


def test_phase_completion_marks_overshoot_correction_allowed():
    result = evaluate_phase_completion(
        [
            {
                "kind": "translation",
                "axis": "forward",
                "target": 5.0,
                "unit": "m",
                "tolerance": 1.0,
                "correction_allowed": True,
            }
        ],
        {"phase_local_measurements": {"forward_displacement_m": 7.2}},
    )

    assert result["any_overshoot"] is True
    assert result["evaluations"][0]["status"] == "overshoot_correction_allowed"


def test_phase_completion_signs_left_and_down_targets():
    result = evaluate_phase_completion(
        [
            {"kind": "translation", "axis": "left", "target": 2.0, "unit": "m", "tolerance": 0.8},
            {"kind": "translation", "axis": "down", "target": 1.0, "unit": "m", "tolerance": 0.8},
        ],
        {"phase_local_measurements": {"right_displacement_m": -2.1, "up_displacement_m": -1.2}},
    )

    assert result["all_within_tolerance"] is True
    assert [item["target"] for item in result["evaluations"]] == [-2.0, -1.0]


def test_phase_completion_marks_visual_alignment_for_model_judgment():
    result = evaluate_phase_completion(
        [
            {
                "kind": "visual_alignment",
                "target": "person",
                "evidence": "latest_rgb",
                "require_target_visible": True,
                "correction_allowed": True,
            }
        ],
        {"phase_local_measurements": {"yaw_delta_deg": 15.0}},
    )

    assert result["criteria_count"] == 1
    assert result["measurable_count"] == 0
    assert result["model_judged_count"] == 1
    assert result["all_within_tolerance"] is False
    assert result["any_overshoot"] is False
    assert result["evaluations"][0]["status"] == "requires_model_judgment"
    assert result["evaluations"][0]["target"] == "person"
    assert result["evaluations"][0]["actual"] is None
    assert result["evaluations"][0]["error"] is None


def test_phase_completion_handles_mixed_measurable_and_model_judged_criteria():
    result = evaluate_phase_completion(
        [
            {"kind": "translation", "axis": "forward", "target": 2.0, "unit": "m", "tolerance": 0.5},
            {"kind": "visual_alignment", "target": "sunshade", "evidence": "latest_rgb"},
        ],
        {"phase_local_measurements": {"forward_displacement_m": 2.1}},
    )

    assert result["criteria_count"] == 2
    assert result["measurable_count"] == 1
    assert result["model_judged_count"] == 1
    assert result["all_within_tolerance"] is True
    assert [item["status"] for item in result["evaluations"]] == [
        "within_tolerance",
        "requires_model_judgment",
    ]
