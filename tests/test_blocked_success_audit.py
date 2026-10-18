"""Scoring projection changes completion eligibility, not technical validity."""
import pytest
from tests.manual.blocked_success_audit import classify


@pytest.mark.parametrize("distance,s20,s25", [(8.36, True, True), (20, True, True), (24, False, True), (25, False, True), (26, False, False), (float('nan'), False, False), (None, False, False)])
def test_blocked_thresholds(distance, s20, s25):
    row = classify("travel_uav", {"metrics": {"navigation_error": distance, "success": False},
                                 "stop_source": "runtime.report_blocked"})
    assert row["revised_sr20_success"] is s20
    assert row["revised_sr25_success"] is s25


def test_technical_error_and_unrecognized_stop_not_promoted():
    base = {"metrics": {"distance_to_goal": 5}, "stop_source": "runtime.report_blocked", "error": "connection failed"}
    assert not classify("aerial_vln", base)["revised_sr25_success"]
    base.pop("error")
    base["stop_source"] = "operator"
    assert not classify("aerial_vln", base)["revised_sr25_success"]
