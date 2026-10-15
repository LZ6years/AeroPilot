"""Report diagnostics retain uncertainty and historical window geometry."""
from tests.manual.flight_teaching_metrics import analyze, command_motion


def test_unknown_parameters_do_not_become_zero_distance():
    assert command_motion({"vx": 2, "vy": 0, "duration": 2}) is None
    assert command_motion({"vx": 2, "vy": 0, "vz": 0, "duration": 2})["planar_m"] == 4


def test_old_two_action_window_keeps_one_shared_motion_measurement():
    args = {"vx": 2, "vy": 0, "vz": 0, "duration": 2}
    events = [
        {"kind": "state_checkpoint", "payload": {"observation_id": "before", "reason": "instruction_start"}},
        {"kind": "assistant_turn", "payload": {"tool_calls": [
            {"tool_call_id": "a", "tool_name": "uav.move_body", "args": args},
            {"tool_call_id": "b", "tool_name": "uav.move_body", "args": args}]}},
        *[{"kind": "tool_result", "payload": {"tool_call_id": key, "status": "ok"}} for key in ("a", "b")],
        {"kind": "state_checkpoint", "payload": {"observation_id": "after", "reason": "physical_window_completed",
            "fragments": [{"name": "motion", "content": {"last_window_uav_motion": {"planar_m": 7.8}}}]}}]
    result = analyze(events)
    assert result["max_calls_in_observed_window"] == 2
    window = result["action_observation_timeline"][0]
    assert window["observation_before_id"] == "before" and window["observation_after_id"] == "after"
    assert window["actual_motion_for_whole_window"] == {"planar_m": 7.8}
    assert all(a["executed_command_motion"] is None and "actual_motion" not in a for a in window["actions"])


def test_rejection_has_no_window_and_unknown_execution_stays_unobserved():
    events = [{"kind": "assistant_turn", "payload": {"tool_calls": [
        {"tool_call_id": "a", "tool_name": "uav.move_body", "args": {}},
        {"tool_call_id": "b", "tool_name": "uav.move_body", "args": {}}]}},
        {"kind": "tool_result", "payload": {"tool_call_id": "a", "status": "rejected"}},
        {"kind": "tool_result", "payload": {"tool_call_id": "b", "status": "error", "diagnostic": {"execution": "unknown"}}}]
    result = analyze(events)
    assert result["movement_choices"]["requested_planar"]["peak"] is None
    assert result["max_calls_in_observed_window"] is None
    assert len(result["action_observation_timeline"]) == 1
    assert result["action_observation_timeline"][0]["observation_after_id"] is None
    assert result["action_observation_timeline"][0]["actions"][0]["diagnostic"]["execution"] == "unknown"
