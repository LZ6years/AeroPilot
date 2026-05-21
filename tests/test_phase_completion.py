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
