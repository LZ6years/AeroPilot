"""Test M4 safety supervisor clipping and validation."""

from __future__ import annotations

import pytest

from src.agent.safety import SafetySupervisor, SafetyViolation
from src.agent.planning.schemas import PlannedStep
from src.runtime.configs.safety_config import UavSafetyConfig


def test_physical_safety_needs_no_phase_and_reuses_existing_algorithm():
    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0))
    steps = [PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 90.0}),
             PlannedStep.tool_step("uav.move_body", {"vx": 100, "duration": 100})]
    raw = [step.to_dict() for step in steps]
    direct = supervisor.review_steps(steps)
    assert direct.steps[0].args["yaw_deg"] == 30.0
    assert direct.interventions
    assert [step.to_dict() for step in steps] == raw
    assert not hasattr(direct, "phase")


def test_safety_clips_large_turn():
    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0))
    plan = ([PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 90.0})])

    decision = supervisor.review_steps(plan)

    assert decision.steps[0].args["yaw_deg"] == 30.0
    assert decision.interventions[0]["field"] == "yaw_deg"




def test_safety_clips_long_or_fast_move_body():
    supervisor = SafetySupervisor(
        UavSafetyConfig(max_linear_velocity_mps=6.0, max_vertical_velocity_mps=3.0, max_move_duration_sec=8.0)
    )
    plan = ([PlannedStep.tool_step("uav.move_body", {"vx": 5.0, "vy": -5.0, "vz": 3.0, "duration": 10.0})]
    )

    decision = supervisor.review_steps(plan)
    args = decision.steps[0].args

    assert args == {"vx": 5.0, "vy": -5.0, "vz": 3.0, "duration": 8.0}
    assert decision.interventions


def test_safety_allows_consecutive_yaw_scan_under_budget():
    class State:
        yaw_scan_offset_deg = 30.0
        yaw_scan_extent_deg = 30.0

    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0, max_yaw_scan_extent_deg=480.0))
    plan = ([
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 15.0}, "scan again"),
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": -15.0}, "scan back"),
            PlannedStep.observe_step(),
        ]
    )

    decision = supervisor.review_steps(plan, State())

    assert [step.tool for step in decision.steps[:2]] == ["uav.turn_relative", "uav.turn_relative"]
    assert decision.interventions == []


def test_safety_replaces_yaw_scan_after_budget_exceeded():
    class State:
        yaw_scan_offset_deg = 470.0
        yaw_scan_extent_deg = 470.0

    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0, max_yaw_scan_extent_deg=480.0))
    plan = ([PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 15.0})])

    decision = supervisor.review_steps(plan, State())

    assert decision.steps[0].tool == "uav.hover"
    assert decision.interventions[0]["reason"] == "yaw_scan_budget_exceeded"
    assert decision.interventions[0]["yaw_scan_offset_deg"] == 470.0
    assert decision.interventions[0]["yaw_scan_extent_deg"] == 470.0
    assert decision.interventions[0]["projected_yaw_scan_extent_deg"] == 485.0


def test_safety_allows_return_toward_instruction_start_without_consuming_new_extent():
    class State:
        yaw_scan_offset_deg = 470.0
        yaw_scan_extent_deg = 470.0

    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0, max_yaw_scan_extent_deg=480.0))
    plan = ([PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": -30.0})]
    )

    decision = supervisor.review_steps(plan, State())

    assert decision.steps[0].tool == "uav.turn_relative"
    assert decision.steps[0].args["yaw_deg"] == -30.0
    assert decision.interventions == []


def test_safety_projects_yaw_extent_inside_same_plan():
    class State:
        yaw_scan_offset_deg = 450.0
        yaw_scan_extent_deg = 450.0

    supervisor = SafetySupervisor(UavSafetyConfig(max_yaw_step_deg=30.0, max_yaw_scan_extent_deg=480.0))
    plan = ([
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 20.0}),
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 20.0}),
            PlannedStep.observe_step(),
        ]
    )

    decision = supervisor.review_steps(plan, State())

    assert decision.steps[0].tool == "uav.turn_relative"
    assert decision.steps[1].tool == "uav.hover"
    assert decision.interventions[0]["reason"] == "yaw_scan_budget_exceeded"


def test_safety_rejects_unknown_tool():
    supervisor = SafetySupervisor()
    plan = ([PlannedStep.tool_step("uav.land", {})])

    with pytest.raises(SafetyViolation, match="not allowed"):
        supervisor.review_steps(plan)


def test_safety_rejects_lifecycle_tools_in_physical_batch():
    supervisor = SafetySupervisor()
    plan = ([
            PlannedStep.tool_step("uav.connect", {}),
            PlannedStep.tool_step("uav.takeoff", {}),
        ]
    )

    with pytest.raises(SafetyViolation, match="not allowed"):
        supervisor.review_steps(plan)
