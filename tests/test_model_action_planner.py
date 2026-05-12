"""Test model-backed short-horizon action planning."""

from __future__ import annotations

import pytest

from src.agent.mission_state import MissionState
from src.agent.model_action_planner import ModelActionPlanner, ModelActionPlannerConfigError
from src.agent.schemas import SemanticPhase
from src.agent.task_parser import TaskParser
from src.runtime.configs.model_config import ModelConfig


class StubModelActionPlanner(ModelActionPlanner):
    def __init__(self, response: dict, model_config: ModelConfig | None = None) -> None:
        self.response = response
        super().__init__(
            model_config
            or ModelConfig(
                task_parser_mode="model",
                model_name="stub",
                api_key="test-key",
                base_url="https://example.test/v1",
            )
        )

    def _call_openai_compatible(self, context: dict):
        return self.response


def make_state() -> MissionState:
    result = TaskParser(mode="rule_based").parse("左转寻找建筑入口并停在入口前")
    result.plan.phases[0] = SemanticPhase(
        index=1,
        total=len(result.plan.phases),
        name="orient_leftward",
        meaning="向左调整视角，寻找入口可能区域",
    )
    state = MissionState("run_test", result.instruction, result.plan)
    state.record_observation({"source": "preflight", "modalities": ["rgb", "depth", "pose"]})
    return state


def test_model_action_planner_requires_api_key():
    planner = ModelActionPlanner(ModelConfig(model_name="stub", api_key=""))

    with pytest.raises(ModelActionPlannerConfigError, match="API_KEY"):
        planner.plan_next(make_state(), {"current_phase": {"name": "orient_leftward"}})


def test_model_action_planner_accepts_open_phase_name():
    state = make_state()
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.turn_relative",
                    "args": {"yaw_deg": -15},
                    "reason": "small left scan",
                },
                {"kind": "observe", "modalities": ["rgb", "depth", "pose"]},
            ],
        }
    )

    plan = planner.plan_next(state, {"current_phase": {"name": "orient_leftward"}})

    assert plan.phase == "orient_leftward"
    assert plan.steps[0].tool == "uav.turn_relative"
    assert plan.steps[0].args["yaw_deg"] == -15
    assert plan.steps[1].kind == "observe"


def test_model_action_planner_rejects_unknown_tool():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.land", "args": {}, "reason": "bad"}],
        }
    )

    with pytest.raises(ValueError, match="not allowed"):
        planner.plan_next(make_state(), {"current_phase": {"name": "orient_leftward"}})


def test_model_action_planner_rejects_phase_mismatch():
    planner = StubModelActionPlanner(
        {
            "phase": "different_phase",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="phase mismatch"):
        planner.plan_next(make_state(), {"current_phase": {"name": "orient_leftward"}})


def test_model_action_planner_normalizes_move_body_distance_args():
    state = make_state()
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"x_m": 4.0, "y_m": -2.0, "duration_sec": 2.0},
                    "reason": "move using distance args",
                },
                {"kind": "observe", "modalities": ["rgb", "depth", "pose"]},
            ],
        }
    )

    plan = planner.plan_next(state, {"current_phase": {"name": "orient_leftward"}})

    assert plan.steps[0].args == {"vx": 2.0, "vy": -1.0, "vz": 0.0, "duration": 2.0}


def test_model_action_planner_normalizes_move_body_direction_alias():
    state = make_state()
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "rear", "duration_sec": 1.5},
                    "reason": "move backward",
                }
            ],
        }
    )

    plan = planner.plan_next(state, {"current_phase": {"name": "orient_leftward"}})

    assert plan.steps[0].args == {"direction": "backward", "duration": 1.5}


def test_model_action_planner_rejects_unknown_move_body_direction():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "diagonal"},
                    "reason": "bad direction",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="Unsupported uav.move_body direction"):
        planner.plan_next(make_state(), {"current_phase": {"name": "orient_leftward"}})
