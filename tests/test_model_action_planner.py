"""Test model-backed short-horizon action planning."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from src.agent.mission_state import MissionState
from src.agent.planning.model_action_planner import (
    ModelActionPlanner,
    ModelActionPlannerConfigError,
)
from src.agent.planning.model_inputs import build_model_user_content
from src.agent.planning.model_repair_planner import ModelRepairPlannerError
from src.agent.phase import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK
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
        if not self.response.pop("_omit_action_window", False):
            self.response.setdefault("action_window", make_action_window())
        if not self.response.pop("_omit_agent_messages", False):
            self.response.setdefault("agent_messages", make_agent_messages())
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

    def _call_phase_transition_openai_compatible(self, phase_context: dict):
        return {
            "decision": COMPLETE_TASK,
            "reason": "stub post-action phase evidence is complete",
            "agent_messages": make_agent_messages(message_type="final", content="任务完成，我已稳定悬停。"),
        }


class StubRepairPlanner:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.contexts = []

    def repair(self, repair_context):
        self.contexts.append(repair_context)
        if not self.responses:
            raise AssertionError("unexpected repair call")
        response = dict(self.responses.pop(0))
        if not response.pop("_omit_action_window", False):
            response.setdefault("action_window", make_action_window(intent="repair current phase contract"))
        if not response.pop("_omit_agent_messages", False):
            response.setdefault("agent_messages", make_agent_messages(content="我会改用符合约束的动作继续。"))
        return response


def make_action_window(**overrides) -> dict:
    window = {
        "intent": "make current phase progress",
        "scope": "current_phase_only",
        "requires_intermediate_observation": False,
        "expected_effect": "the current phase should have measurable progress",
    }
    window.update(overrides)
    return window


def make_agent_messages(message_type: str = "progress", content: str = "我准备执行当前阶段的安全动作。") -> list[dict]:
    return [
        {
            "type": message_type,
            "content": content,
            "evidence_refs": [],
            "confidence": "medium",
            "requires_user_reply": False,
        }
    ]


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
        },
    }


def test_model_user_content_stays_text_without_model_inputs():
    context = make_context()
    context["safety_constraints"] = {"yaw_scan_remaining_deg": 10.0}
    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    assert text_context["current_phase"]["name"] == "orient_leftward"
    assert "safety_constraints" not in text_context
    assert "observation_policy" not in text_context


def test_model_user_content_attaches_encoded_image_from_any_fragment():
    image_data = "data:image/png;base64," + base64.b64encode(b"fake-png").decode("ascii")
    context = make_context()
    context["fragments"] = [
        {
            "name": "custom_visual",
            "content": {"visual_processing": {"strategy": "select_latest_rgb"}},
            "model_inputs": [
                {
                    "type": "image",
                    "role": "latest_observation",
                    "mime_type": "image/png",
                    "encoding": "data_url",
                    "data": image_data,
                    "metadata": {"source_path": "does-not-need-to-exist.png", "modality": "rgb"},
                }
            ],
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    text_context = json.loads(content[0]["text"])
    assert text_context["current_phase"]["name"] == "orient_leftward"
    assert image_data not in content[0]["text"]
    assert text_context["fragments"][0]["model_inputs"][0]["data"].startswith("<omitted encoded model input data")
    assert content[1] == {"type": "image_url", "image_url": {"url": image_data}}


def test_model_user_content_ignores_unencoded_image_path():
    context = make_context()
    context["fragments"] = [
        {
            "name": "image",
            "model_inputs": [{"type": "image", "path": "missing.png", "metadata": {"modality": "rgb"}}],
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, str)
    assert json.loads(content)["current_phase"]["name"] == "orient_leftward"


def test_model_user_content_adds_text_model_inputs():
    context = make_context()
    context["fragments"] = [
        {
            "name": "visual_caption",
            "model_inputs": [{"type": "text", "role": "visual_caption", "text": "画面中疑似入口位于右前方。"}],
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, list)
    assert json.loads(content[0]["text"])["current_phase"]["name"] == "orient_leftward"
    assert content[1] == {"type": "text", "text": "画面中疑似入口位于右前方。"}


def test_action_planner_prompt_uses_image_sequence_action_trend_and_explicit_targets():
    prompt = (
        Path("src")
        / "agent"
        / "planning"
        / "prompts"
        / "action_planner_system.txt"
    ).read_text(encoding="utf-8")

    assert "selected_frames" in prompt
    assert "action_trend" in prompt
    assert 'target_source="explicit_user"' in prompt
    assert "do not mention exact remaining meters/degrees" in prompt
    assert "alternating_yaw" in prompt


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
    assert plan.agent_messages[0].type == "progress"


def test_short_horizon_plan_direct_constructor_defaults_to_advance():
    plan = ShortHorizonPlan("orient_leftward", [])

    assert plan.phase_transition.decision == ADVANCE_TO_NEXT_PHASE
    assert plan.action_window.scope == "current_phase_only"
    assert plan.to_dict()["action_window"]["intent"]


def test_model_action_planner_accepts_missing_phase_transition():
    planner = StubModelActionPlanner(
        {
            "_omit_phase_transition": True,
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    plan = planner.plan_next(make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"


def test_model_action_planner_rejects_missing_action_window():
    planner = StubModelActionPlanner(
        {
            "_omit_action_window": True,
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="requires action_window"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_missing_agent_messages():
    planner = StubModelActionPlanner(
        {
            "_omit_agent_messages": True,
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="requires agent_messages"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_invalid_action_window_scope():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "action_window": make_action_window(scope="whole_task"),
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="current_phase_only"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_rejects_multi_step_window_that_needs_observation():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "action_window": make_action_window(
                requires_intermediate_observation=True,
                expected_effect="need a fresh observation before the second step",
            ),
            "steps": [
                {"kind": "tool", "tool": "uav.turn_relative", "args": {"yaw_deg": -10}, "reason": "scan"},
                {"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"},
            ],
        }
    )

    with pytest.raises(ValueError, match="exactly one step"):
        planner.plan_next(make_state(), make_context())


def test_model_action_planner_ignores_invalid_phase_transition_decision():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
            "phase_transition": {"decision": "jump_somewhere", "reason": "bad"},
        }
    )

    plan = planner.plan_next(make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"


def test_model_action_planner_ignores_phase_transition_target_fields():
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

    plan = planner.plan_next(make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"


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

    with pytest.raises(ValueError, match="not allowed"):
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


def test_model_action_planner_accepts_forward_5m_duration_from_nominal_scale():
    state = make_state()
    context = make_context()
    context["allowed_action_constraints"]["action_schemas"]["uav.move_body"]["limits"]["duration_max"] = 8.0
    context["allowed_action_constraints"]["nominal_action_scale"] = {
        "linear_velocity_mps": 2.0,
        "vertical_velocity_mps": 1.0,
    }
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "forward", "duration": 2.5},
                    "reason": "move forward about 5 meters using nominal 2 m/s scale",
                }
            ],
        }
    )

    plan = planner.plan_next(state, context)

    assert plan.steps[0].args == {"direction": "forward", "duration": 2.5}


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
        "action_window": make_action_window(),
        "agent_messages": make_agent_messages(),
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


def test_model_action_planner_repair_output_missing_phase_transition_succeeds():
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

    plan = planner.plan_next(make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"


def test_model_action_planner_repair_output_missing_action_window_fails():
    repair_planner = StubRepairPlanner(
        [
            {
                "_omit_action_window": True,
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "missing window"}],
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
    assert "requires action_window" in exc_info.value.violations[-1]["message"]


def test_model_action_planner_repair_output_missing_agent_messages_fails():
    repair_planner = StubRepairPlanner(
        [
            {
                "_omit_agent_messages": True,
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "missing messages"}],
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
    assert "requires agent_messages" in exc_info.value.violations[-1]["message"]


def test_model_action_planner_judges_phase_transition_from_post_action_context():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    transition = planner.judge_phase_transition({"executed_steps": [], "measurements": {}})

    assert transition.decision == COMPLETE_TASK
    assert "post-action" in transition.reason
    assert transition.agent_messages[0].type == "final"
