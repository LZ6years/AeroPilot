"""Test model-backed short-horizon action planning."""

from __future__ import annotations

import pytest

from src.agent.mission_state import MissionState
from src.agent.planning.model_action_planner import ModelActionPlanner, ModelActionPlannerConfigError
from src.agent.semantic.schemas import SemanticPhase
from src.agent.semantic.task_parser import TaskParser
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


def make_context() -> dict:
    return {
        "current_phase": {"name": "orient_leftward"},
        "observation_policy": {
            "mode": "automatic",
            "modalities": ["rgb", "depth", "pose"],
        },
        "allowed_action_constraints": {
            "planner_tools": ["uav.hover", "uav.turn_relative", "uav.move_body"],
            "action_schemas": {
                "uav.hover": {"args": {}},
                "uav.turn_relative": {
                    "args": {"yaw_deg": "number"},
                    "limits": {"yaw_deg_min": -30.0, "yaw_deg_max": 30.0},
                },
                "uav.move_body": {
                    "args_one_of": [
                        {"direction": "forward|backward|left|right|up|down", "duration": "number"},
                        {"vx": "number", "vy": "number", "vz": "number", "duration": "number"},
                    ],
                    "limits": {
                        "abs_vx_vy_max": 2.0,
                        "abs_vz_max": 1.0,
                        "duration_max": 2.0,
                    },
                },
            },
            "forbidden_tools": [
                "uav.connect",
                "uav.takeoff",
                "uav.land",
                "uav.set_pose",
                "uav.get_pose",
                "uav.get_frames",
                "env.observe",
            ],
        },
    }


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
            ],
        }
    )

    plan = planner.plan_next(state, make_context())

    assert plan.phase == "orient_leftward"
    assert plan.steps[0].tool == "uav.turn_relative"
    assert plan.steps[0].args["yaw_deg"] == -15
    assert len(plan.steps) == 1


def test_model_action_planner_rejects_unknown_tool():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.fake", "args": {}, "reason": "bad"}],
        }
    )

    with pytest.raises(ValueError, match="not allowed"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_env_observe_tool_step():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "env.observe",
                    "args": {"modalities": ["rgb", "depth", "pose"]},
                    "reason": "bad observe",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="forbidden"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_phase_mismatch():
    planner = StubModelActionPlanner(
        {
            "phase": "different_phase",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="phase mismatch"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_yaw_above_context_limit():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.turn_relative",
                    "args": {"yaw_deg": -45.0},
                    "reason": "too large",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="below limit"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_observe_step():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "observe",
                    "modalities": ["rgb", "depth", "pose"],
                    "reason": "model should not request observation",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="runtime perception policy"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_hover_with_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {"duration": 1.0}, "reason": "bad"}],
        }
    )

    with pytest.raises(ValueError, match="uav.hover args must be"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_move_body_distance_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"x_m": 4.0, "y_m": -2.0, "duration_sec": 2.0},
                    "reason": "move using distance args",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="uav.move_body args must be exactly"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_accepts_contract_move_body_direction_args():
    state = make_state()
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "backward", "duration": 1.5},
                    "reason": "move backward",
                }
            ],
        }
    )

    plan = planner.plan_next(state, make_context())

    assert plan.steps[0].args == {"direction": "backward", "duration": 1.5}


def test_model_action_planner_rejects_unknown_move_body_direction():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "diagonal", "duration": 1.0},
                    "reason": "bad direction",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="Unsupported uav.move_body direction"):
        planner.plan_next(make_state(), make_context())
