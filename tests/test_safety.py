"""Test M4 safety supervisor clipping and validation."""

from __future__ import annotations

import pytest

from src.agent.messages import AgentMessage
from src.agent.safety import SafetySupervisor, SafetyViolation
from src.agent.planning.schemas import ActionWindow, PlannedStep, ShortHorizonPlan
from src.runtime.configs.safety_config import UavSafetyConfig


def test_safety_clips_large_turn():
    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0))
    plan = ShortHorizonPlan("orient_right", [PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 90.0})])

    decision = supervisor.review(plan)

    assert decision.plan.steps[0].args["yaw_deg"] == 30.0
    assert decision.interventions[0]["field"] == "yaw_deg"


def test_safety_preserves_action_window_metadata():
    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0))
    window = ActionWindow(
        intent="turn toward the phase target",
        scope="current_phase_only",
        requires_intermediate_observation=False,
        expected_effect="yaw changes toward the target",
    )
    plan = ShortHorizonPlan(
        "orient_right",
        [PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 90.0})],
        action_window=window,
    )

    decision = supervisor.review(plan)

    assert decision.plan.action_window is window
    assert decision.plan.action_window.expected_effect == "yaw changes toward the target"


def test_safety_preserves_agent_messages():
    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0))
    message = AgentMessage("progress", "我会先安全转向。")
    plan = ShortHorizonPlan(
        "orient_right",
        [PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 90.0})],
        agent_messages=[message],
    )

    decision = supervisor.review(plan)

    assert decision.plan.agent_messages == [message]


def test_safety_clips_long_or_fast_move_body():
    supervisor = SafetySupervisor(
        UavSafetyConfig(max_linear_velocity_mps=6.0, max_vertical_velocity_mps=3.0, max_move_duration_sec=8.0)
    )
    plan = ShortHorizonPlan(
        "move",
        [PlannedStep.tool_step("uav.move_body", {"vx": 5.0, "vy": -5.0, "vz": 3.0, "duration": 10.0})],
    )

    decision = supervisor.review(plan)
    args = decision.plan.steps[0].args

    assert args == {"vx": 5.0, "vy": -5.0, "vz": 3.0, "duration": 8.0}
    assert decision.interventions


def test_safety_allows_consecutive_yaw_scan_under_budget():
    class State:
        yaw_scan_offset_deg = 30.0
        yaw_scan_extent_deg = 30.0

    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0, max_yaw_scan_extent_deg=480.0))
    plan = ShortHorizonPlan(
        "scan_again",
        [
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 15.0}, "scan again"),
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": -15.0}, "scan back"),
            PlannedStep.observe_step(),
        ],
    )

    decision = supervisor.review(plan, State())

    assert [step.tool for step in decision.plan.steps[:2]] == ["uav.turn_relative", "uav.turn_relative"]
    assert decision.interventions == []


def test_safety_replaces_yaw_scan_after_budget_exceeded():
    class State:
        yaw_scan_offset_deg = 470.0
        yaw_scan_extent_deg = 470.0

    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0, max_yaw_scan_extent_deg=480.0))
    plan = ShortHorizonPlan("scan", [PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 15.0})])

    decision = supervisor.review(plan, State())

    assert decision.plan.steps[0].tool == "uav.hover"
    assert decision.interventions[0]["reason"] == "yaw_scan_budget_exceeded"
    assert decision.interventions[0]["yaw_scan_offset_deg"] == 470.0
    assert decision.interventions[0]["yaw_scan_extent_deg"] == 470.0
    assert decision.interventions[0]["projected_yaw_scan_extent_deg"] == 485.0


def test_safety_allows_return_toward_phase_start_without_consuming_new_extent():
    class State:
        yaw_scan_offset_deg = 470.0
        yaw_scan_extent_deg = 470.0

    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0, max_yaw_scan_extent_deg=480.0))
    plan = ShortHorizonPlan(
        "scan_back",
        [PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": -30.0})],
    )

    decision = supervisor.review(plan, State())

    assert decision.plan.steps[0].tool == "uav.turn_relative"
    assert decision.plan.steps[0].args["yaw_deg"] == -30.0
    assert decision.interventions == []


def test_safety_projects_yaw_extent_inside_same_plan():
    class State:
        yaw_scan_offset_deg = 450.0
        yaw_scan_extent_deg = 450.0

    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0, max_yaw_scan_extent_deg=480.0))
    plan = ShortHorizonPlan(
        "scan_twice",
        [
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 20.0}),
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 20.0}),
            PlannedStep.observe_step(),
        ],
    )

    decision = supervisor.review(plan, State())

    assert decision.plan.steps[0].tool == "uav.turn_relative"
    assert decision.plan.steps[1].tool == "uav.hover"
    assert decision.interventions[0]["reason"] == "yaw_scan_budget_exceeded"


def test_safety_rejects_unknown_tool():
    supervisor = SafetySupervisor()
    plan = ShortHorizonPlan("bad", [PlannedStep.tool_step("uav.land", {})])

    with pytest.raises(SafetyViolation, match="not allowed"):
        supervisor.review(plan)


def test_safety_rejects_lifecycle_tools_in_short_horizon_plan():
    supervisor = SafetySupervisor()
    plan = ShortHorizonPlan(
        "bad",
        [
            PlannedStep.tool_step("uav.connect", {}),
            PlannedStep.tool_step("uav.takeoff", {}),
        ],
    )

    with pytest.raises(SafetyViolation, match="not allowed"):
        supervisor.review(plan)
