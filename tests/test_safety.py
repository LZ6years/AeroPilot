"""Test M4 safety supervisor clipping and validation."""

from __future__ import annotations

import pytest

from src.agent.safety import SafetySupervisor, SafetyViolation
from src.agent.schemas import PlannedStep, ShortHorizonPlan
from src.runtime.configs.action_config import UavActionConfig


def test_safety_clips_large_turn():
    supervisor = SafetySupervisor(UavActionConfig(yaw_step_deg=30.0))
    plan = ShortHorizonPlan("orient_right", [PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 90.0})])

    decision = supervisor.review(plan)

    assert decision.plan.steps[0].args["yaw_deg"] == 30.0
    assert decision.interventions[0]["field"] == "yaw_deg"


def test_safety_clips_long_or_fast_move_body():
    supervisor = SafetySupervisor(
        UavActionConfig(linear_velocity_mps=2.0, vertical_velocity_mps=1.0, move_duration_sec=2.0)
    )
    plan = ShortHorizonPlan(
        "move",
        [PlannedStep.tool_step("uav.move_body", {"vx": 5.0, "vy": -5.0, "vz": 3.0, "duration": 10.0})],
    )

    decision = supervisor.review(plan)
    args = decision.plan.steps[0].args

    assert args == {"vx": 2.0, "vy": -2.0, "vz": 1.0, "duration": 2.0}
    assert decision.interventions


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
