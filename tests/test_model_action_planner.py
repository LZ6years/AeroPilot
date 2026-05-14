"""Test model-backed short-horizon action planning."""

from __future__ import annotations

import pytest

from src.agent.mission_state import MissionState
from src.agent.planning.model_action_planner import ModelActionPlanner, ModelActionPlannerConfigError
from src.agent.planning.model_repair_planner import ModelRepairPlannerError
from src.agent.planning.phase_transition import ADVANCE_TO_NEXT_PHASE
from src.agent.planning.schemas import ShortHorizonPlan
from src.agent.semantic.schemas import SemanticPhase
from src.agent.semantic.task_parser import TaskParser
from src.runtime.configs.model_config import ModelConfig


class StubModelActionPlanner(ModelActionPlanner):
    def __init__(
        self,
        response: dict,
        model_config: ModelConfig | None = None,
        *,
        max_repair_attempts: int = 0,
        repair_planner=None,
    ) -> None:
        self.response = dict(response)
        if not self.response.pop("_omit_phase_transition", False):
            self.response.setdefault(
                "phase_transition",
                {"decision": ADVANCE_TO_NEXT_PHASE, "reason": "phase complete after this action"},
            )
        super().__init__(
            model_config
            or ModelConfig(
                task_parser_mode="model",
                model_name="stub",
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
            max_repair_attempts=max_repair_attempts,
            repair_planner=repair_planner,
        )

    def _call_openai_compatible(self, context: dict):
        return self.response


class StubRepairPlanner:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.contexts = []

    def repair(self, repair_context):
        self.contexts.append(repair_context)
        if not self.responses:
            raise AssertionError("unexpected repair call")
        return self.responses.pop(0)


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
    assert plan.phase_transition.decision == ADVANCE_TO_NEXT_PHASE


def test_short_horizon_plan_direct_constructor_defaults_to_advance():
    plan = ShortHorizonPlan("orient_leftward", [])

    assert plan.phase_transition.decision == ADVANCE_TO_NEXT_PHASE


def test_model_action_planner_rejects_missing_phase_transition():
    planner = StubModelActionPlanner(
        {
            "_omit_phase_transition": True,
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="requires phase_transition"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_invalid_phase_transition_decision():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
            "phase_transition": {"decision": "jump_somewhere", "reason": "bad"},
        }
    )

    with pytest.raises(ValueError, match="Unsupported phase_transition decision"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_phase_transition_target_fields():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
            "phase_transition": {
                "decision": ADVANCE_TO_NEXT_PHASE,
                "next_phase": "skip_to_final",
                "reason": "bad target",
            },
        }
    )

    with pytest.raises(ValueError, match="must not specify target phase fields"):
        planner.plan_next(make_state(), make_context())


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


def test_model_action_planner_accepts_contract_move_body_velocity_args_without_normalization():
    state = make_state()
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"vx": 1.0, "vy": -0.5, "vz": 0.25, "duration": 1.2},
                    "reason": "move with body-frame velocity",
                }
            ],
        }
    )

    plan = planner.plan_next(state, make_context())

    assert plan.steps[0].args == {"vx": 1.0, "vy": -0.5, "vz": 0.25, "duration": 1.2}


def test_model_action_planner_rejects_legacy_move_body_alias_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "front", "duration": 1.0},
                    "reason": "legacy alias should be repaired, not normalized",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="Unsupported uav.move_body direction"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_legacy_move_body_duration_alias():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "forward", "duration_sec": 1.0},
                    "reason": "legacy duration alias should be repaired, not normalized",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="uav.move_body args must be exactly"):
        planner.plan_next(make_state(), make_context())


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


def test_model_action_planner_repairs_contract_violation_on_first_attempt():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "repaired hold"}],
                "phase_transition": {
                    "decision": ADVANCE_TO_NEXT_PHASE,
                    "reason": "repair uses safe hover",
                },
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    plan = planner.plan_next(make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"
    assert repair_planner.contexts[0].attempt == 1
    assert planner.last_repair_result["status"] == "repaired"
    assert planner.last_repair_result["successful_attempt"] == 1


def test_model_action_planner_repairs_contract_violation_on_third_attempt():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 1.0}, "reason": "bad"}],
                "phase_transition": {
                    "decision": ADVANCE_TO_NEXT_PHASE,
                    "reason": "still bad",
                },
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "observe", "modalities": ["rgb"], "reason": "bad"}],
                "phase_transition": {
                    "decision": ADVANCE_TO_NEXT_PHASE,
                    "reason": "still bad",
                },
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "finally safe"}],
                "phase_transition": {
                    "decision": ADVANCE_TO_NEXT_PHASE,
                    "reason": "third repair succeeds",
                },
            },
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    plan = planner.plan_next(make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"
    assert len(planner.last_repair_attempts) == 3
    assert planner.last_repair_result["successful_attempt"] == 3


def test_model_action_planner_repair_failure_exposes_structured_error():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 1.0}, "reason": "bad"}],
                "phase_transition": {
                    "decision": ADVANCE_TO_NEXT_PHASE,
                    "reason": "still bad",
                },
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 2.0}, "reason": "bad"}],
                "phase_transition": {
                    "decision": ADVANCE_TO_NEXT_PHASE,
                    "reason": "still bad",
                },
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 3.0}, "reason": "bad"}],
                "phase_transition": {
                    "decision": ADVANCE_TO_NEXT_PHASE,
                    "reason": "still bad",
                },
            },
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    with pytest.raises(ModelRepairPlannerError, match="exhausted") as exc_info:
        planner.plan_next(make_state(), make_context())

    data = exc_info.value.to_dict()
    assert data["reason"] == "max_repair_attempts_exhausted"
    assert data["max_repair_attempts"] == 3
    assert len(data["repair_attempts"]) == 3


def test_model_action_planner_repair_repeated_same_invalid_plan_fails_early():
    bad_plan = {
        "phase": "orient_leftward",
        "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        "phase_transition": {
            "decision": ADVANCE_TO_NEXT_PHASE,
            "reason": "bad",
        },
    }
    repair_planner = StubRepairPlanner([bad_plan])
    planner = StubModelActionPlanner(
        bad_plan,
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    with pytest.raises(ModelRepairPlannerError, match="repeated"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_repair_output_missing_phase_transition_fails():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "missing transition"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=1,
        repair_planner=repair_planner,
    )

    with pytest.raises(ModelRepairPlannerError) as exc_info:
        planner.plan_next(make_state(), make_context())

    assert exc_info.value.violations[-1]["code"] == "repair_plan_parse_error"
