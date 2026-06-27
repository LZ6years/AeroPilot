"""Tests for deterministic phase metric evidence."""

from __future__ import annotations

from src.agent.phase import evaluate_phase_completion


def test_phase_completion_marks_within_tolerance():
    result = evaluate_phase_completion(
        [{"name": "forward_distance", "target": 5.0, "unit": "m", "tolerance": 1.0}],
        {"measurements": {"forward_m": 5.4}},
    )

    assert result["all_within_tolerance"] is True
    assert result["evaluations"][0]["status"] == "within_tolerance"


def test_phase_completion_marks_below_target():
    result = evaluate_phase_completion(
        [{"name": "yaw_delta", "target": 40.0, "unit": "deg", "tolerance": 8.0}],
        {"measurements": {"yaw_delta_deg": 20.0}},
    )

    assert result["all_within_tolerance"] is False
    assert result["evaluations"][0]["status"] == "below_target"


def test_phase_completion_marks_overshoot_correction_allowed():
    result = evaluate_phase_completion(
        [
            {
                "name": "forward_distance",
                "target": 5.0,
                "unit": "m",
                "tolerance": 1.0,
                "correction_allowed": True,
            }
        ],
        {"measurements": {"forward_m": 7.2}},
    )

    assert result["any_overshoot"] is True
    assert result["evaluations"][0]["status"] == "overshoot_correction_allowed"


def test_phase_completion_uses_signed_right_and_up_targets():
    result = evaluate_phase_completion(
        [
            {"name": "right_distance", "target": -2.0, "unit": "m", "tolerance": 0.8},
            {"name": "up_distance", "target": -1.0, "unit": "m", "tolerance": 0.8},
        ],
        {"measurements": {"right_m": -2.1, "up_m": -1.2}},
    )

    assert result["all_within_tolerance"] is True
    assert [item["target"] for item in result["evaluations"]] == [-2.0, -1.0]


def test_phase_completion_accepts_forward_right_displacement_metrics_together():
    result = evaluate_phase_completion(
        [
            {"name": "forward_distance", "target": 7.37, "unit": "m", "tolerance": 0.25},
            {"name": "right_distance", "target": -5.16, "unit": "m", "tolerance": 0.25},
        ],
        {"measurements": {"forward_m": 7.4, "right_m": -5.1, "yaw_delta_deg": 35.0}},
    )

    assert result["all_within_tolerance"] is True
    assert result["measurable_count"] == 2
    assert [item["metric"]["name"] for item in result["evaluations"]] == ["forward_distance", "right_distance"]


def test_phase_completion_accepts_canonical_descend_target():
    result = evaluate_phase_completion(
        [{"name": "up_distance", "target": -9.0, "unit": "m", "tolerance": 0.5}],
        {"measurements": {"up_m": -9.0, "vertical_m": -9.0}},
    )

    assert result["all_within_tolerance"] is True
    assert result["evaluations"][0]["actual"] == -9.0
    assert result["evaluations"][0]["status"] == "within_tolerance"


def test_phase_completion_without_metrics_requires_model_judgment():
    result = evaluate_phase_completion(
        [],
        {"measurements": {"yaw_delta_deg": 15.0}},
    )

    assert result["metrics_count"] == 0
    assert result["measurable_count"] == 0
    assert result["model_judged_count"] == 1
    assert result["all_within_tolerance"] is False
    assert result["any_overshoot"] is False
    assert result["evaluations"][0]["status"] == "requires_model_judgment"
    assert result["evaluations"][0]["metric"] is None
    assert result["evaluations"][0]["actual"] is None
    assert result["evaluations"][0]["error"] is None


def test_phase_completion_handles_mixed_measurable_and_model_judged_metrics():
    result = evaluate_phase_completion(
        [
            {"name": "forward_distance", "target": 2.0, "unit": "m", "tolerance": 0.5},
            {"name": "visual_confidence", "target": "sunshade_visible"},
        ],
        {"measurements": {"forward_m": 2.1}},
    )

    assert result["metrics_count"] == 2
    assert result["measurable_count"] == 1
    assert result["model_judged_count"] == 1
    assert result["all_within_tolerance"] is True
    assert [item["status"] for item in result["evaluations"]] == [
        "within_tolerance",
        "requires_model_judgment",
    ]
