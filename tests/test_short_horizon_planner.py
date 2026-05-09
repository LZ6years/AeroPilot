"""Test short-horizon planning from semantic phases."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.schemas import SemanticPhase
from src.agent.short_horizon_planner import ShortHorizonPlanner
from src.agent.task_parser import TaskParser
from src.runtime.configs.action_config import UavActionConfig


def make_state(instruction: str) -> MissionState:
    result = TaskParser(mode="rule_based").parse(instruction)
    return MissionState("run_test", instruction, result.plan)


def test_orient_right_generates_turn_and_observe():
    state = make_state("前面右转会有一个黑色的路灯，右转面向路灯后停下")
    planner = ShortHorizonPlanner(UavActionConfig(yaw_step_deg=30.0))

    plan = planner.plan_next(state)

    assert plan.phase == "orient_right"
    assert "uav.connect" not in [step.tool for step in plan.steps]
    assert "uav.takeoff" not in [step.tool for step in plan.steps]
    assert [step.kind for step in plan.steps][-1] == "observe"
    assert any(step.tool == "uav.turn_relative" and step.args["yaw_deg"] == 30.0 for step in plan.steps)


def test_orient_left_generates_left_turn_and_observe():
    state = make_state("左转寻找建筑入口并停在入口前")
    planner = ShortHorizonPlanner(UavActionConfig(yaw_step_deg=30.0))

    plan = planner.plan_next(state)

    assert plan.phase == "orient_left"
    assert any(step.tool == "uav.turn_relative" and step.args["yaw_deg"] == -30.0 for step in plan.steps)
    assert plan.steps[-1].kind == "observe"


def test_approach_landmark_without_visual_confirmation_only_hover_observe():
    state = make_state("前方寻找红色车辆，靠近但保持安全距离")
    state.current_phase_index = 1
    state.connected = True
    state.airborne = True

    plan = ShortHorizonPlanner().plan_next(state)

    assert plan.phase == "approach_landmark"
    assert [step.tool for step in plan.steps if step.kind == "tool"] == ["uav.hover"]
    assert plan.steps[-1].kind == "observe"
    assert not any(step.args.get("direction") == "forward" for step in plan.steps)


def test_unknown_phase_generates_only_hover_and_observe():
    state = make_state("前方寻找红色车辆，靠近但保持安全距离")
    original = state.semantic_plan.phases[0]
    state.semantic_plan.phases[0] = SemanticPhase(
        index=original.index,
        total=original.total,
        name="scan_forward_region",
        meaning=original.meaning,
    )

    plan = ShortHorizonPlanner().plan_next(state)

    assert plan.phase == "scan_forward_region"
    assert [step.tool for step in plan.steps if step.kind == "tool"] == ["uav.hover"]
    assert plan.steps[-1].kind == "observe"
    assert "uav.connect" not in [step.tool for step in plan.steps]
    assert "uav.takeoff" not in [step.tool for step in plan.steps]
