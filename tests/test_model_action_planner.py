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
from src.agent.planning.schemas import CoupledActionPhasePlan, ShortHorizonPlan
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

    def _call_coupled_action_phase_openai_compatible(self, context: dict):
        response = dict(self.response)
        response.setdefault("phase_decision", {"decision": "continue_current_phase", "reason": "stub continues"})
        return response


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
        goal="orient toward the left-side entrance search area",
        done_when=["left-side entrance search area is visible"],
        observe=["latest_observation", "phase_execution_summary"],
        continue_when=["left-side area is not yet visible"],
        metrics=[],
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


def make_tool_plan(
    planner: ModelActionPlanner,
    state: MissionState | None = None,
    context: dict | None = None,
) -> ShortHorizonPlan:
    coupled_plan = planner.plan_next_with_phase_decision(state or make_state(), context or make_context())
    return coupled_plan.to_short_horizon_plan()


def test_model_user_content_stays_text_without_model_inputs():
    context = make_context()
    context["safety_constraints"] = {"yaw_scan_remaining_deg": 10.0}
    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    assert text_context["current_phase"]["name"] == "orient_leftward"
    assert "safety_constraints" not in text_context
    assert "observation_policy" not in text_context


def test_model_user_content_strips_legacy_move_direction_shorthand():
    context = make_context()
    context["allowed_action_constraints"]["action_schemas"]["uav.move_body"]["args_one_of"].append(
        {"direction": "legacy", "duration": "number"}
    )
    context["phase_execution_summary"] = {
        "tool_facts": {
            "move_duration_by_direction_success": {"forward": 1.0},
            "move_duration_success_sum_sec": 1.0,
        }
    }
    context["fragments"] = [
        {
            "name": "action",
            "content": {
                "recent_actions": [
                    {
                        "tool": "uav.move_body",
                        "args": {"direction": "forward", "duration": 1.0},
                        "history_only": True,
                    }
                ]
            },
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    text = json.dumps(text_context, ensure_ascii=False)
    assert '"direction"' not in text
    assert "move_duration_by_direction_success" not in text
    assert text_context["allowed_action_constraints"]["action_schemas"]["uav.move_body"]["args_one_of"] == [
        {"vx": "number", "vy": "number", "vz": "number", "duration": "number"}
    ]
    assert text_context["fragments"][0]["content"]["recent_actions"][0]["args"] == {
        "duration": 1.0,
        "legacy_named_motion": True,
    }


def test_model_user_content_puts_flyagent_rules_before_task_fragments():
    context = make_context()
    context.update(
        {
            "rules": {"text": "FlyAgent baseline flight rules.", "source": "memory/FLYAGENT.md"},
            "active_instruction": "Turn toward the visual target.",
            "semantic_goal": {"target": "visual target"},
            "phase_progress_evidence": {"motion_progress": {"yaw_only_so_far": False}},
            "fragments": [
                {
                    "name": "skills",
                    "content": {
                        "selected_skills": [
                            {
                                "name": "visual_target_task_reasoning",
                                "summary": "Use visual evidence carefully.",
                                "skill_text": "Full selected skill body.",
                            }
                        ]
                    },
                    "metadata": {"debug_only": True},
                }
            ],
            "phase_execution_summary": {"window_count": 1},
            "user_preferences": {"text": "Be concise."},
            "context_schema": {"planner_context_fields": {"rules": "Global FlyAgent behavior guidance."}},
            "metadata": {"run_id": "run_test"},
        }
    )

    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    assert list(text_context)[:9] == [
        "rules",
        "active_instruction",
        "current_phase",
        "semantic_goal",
        "phase_progress_evidence",
        "fragments",
        "allowed_action_constraints",
        "phase_execution_summary",
        "user_preferences",
    ]
    assert text_context["rules"]["text"] == "FlyAgent baseline flight rules."
    assert text_context["fragments"][0]["content"]["selected_skills"][0]["name"] == "visual_target_task_reasoning"
    assert "summary" not in text_context["fragments"][0]["content"]["selected_skills"][0]
    assert text_context["fragments"][0]["content"]["selected_skills"][0]["skill_text"] == "Full selected skill body."
    assert "metadata" not in text_context["fragments"][0]
    assert "context_schema" not in text_context
    assert "metadata" not in text_context


def test_model_user_content_strips_phase_judge_audit_fields():
    context = {
        "active_instruction": "Turn toward the person.",
        "semantic_goal": {"target": "person"},
        "current_phase": {
            "name": "orient_toward_person",
            "goal": "face the person",
            "done_when": ["person is visible", "view is roughly aligned"],
            "observe": ["latest_observation"],
            "continue_when": ["person is not confirmed"],
            "metrics": [],
        },
        "phase_metrics": [],
        "phase_completion_evaluation": {"model_judged_count": 1},
        "phase_execution_summary": {"window_count": 1},
        "measurements": {"yaw_delta_deg": -5.0},
        "user_preferences": {"style": "concise"},
        "fragments": [{"name": "image", "model_inputs": []}],
        "planner_context_summary": {"fragment_names": ["image"], "safety_constraints": {"yaw_scan_remaining_deg": 340}},
        "action_window": {"intent": "scan", "expected_effect": "target centered"},
        "action_plan": {"steps": [{"tool": "uav.turn_relative", "args": {"yaw_deg": 5}}]},
        "safety_intervention_summary": {"reasons": ["yaw_scan_budget_exceeded"]},
        "yaw_scan_remaining_deg": 340,
        "visual_phase_budget_exhausted": False,
        "recent_executed_tools": ["uav.turn_relative"],
        "recent_safety_reasons": [],
        "safety_interventions": [],
        "recent_history": {"recent_action_refs": [{"ref": "memory/actions.jsonl#1"}]},
        "observation_before": {"pose": {"x": 0, "velocity": {"x": 1}}},
        "phase_window_count": 1,
        "max_phase_window_count": 100,
        "observation_after": {
            "source": "automatic_phase_window",
            "timestamp": "t1",
            "modalities": ["rgb", "depth", "pose"],
            "pose": {"x": 1},
            "frames": {"rgb": {"path": "frame.png"}, "depth": {"path": "depth.png"}},
        },
        "executed_steps": [
            {
                "phase": "orient_toward_person",
                "step_index": 0,
                "step": {"tool": "uav.turn_relative", "args": {"yaw_deg": -5}},
                "result": {"ok": True, "message": "turned", "raw": {"debug": True}},
                "stabilization": {"elapsed_sec": 1.0},
            }
        ],
    }

    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    for debug_key in [
        "planner_context_summary",
        "action_window",
        "action_plan",
        "safety_intervention_summary",
        "yaw_scan_remaining_deg",
        "visual_phase_budget_exhausted",
        "recent_executed_tools",
        "recent_safety_reasons",
        "safety_interventions",
        "recent_history",
        "observation_before",
        "phase_window_count",
        "max_phase_window_count",
    ]:
        assert debug_key not in text_context
    assert text_context["active_instruction"] == "Turn toward the person."
    assert text_context["executed_steps"] == [
        {
            "phase": "orient_toward_person",
            "step_index": 0,
            "tool": "uav.turn_relative",
            "args": {"yaw_deg": -5},
            "ok": True,
            "message": "turned",
        }
    ]
    assert text_context["observation_after"] == {
        "source": "automatic_phase_window",
        "timestamp": "t1",
        "modalities": ["rgb", "depth", "pose"],
        "pose": {"x": 1},
        "has_rgb": True,
        "has_depth": True,
    }


def test_model_user_content_slims_fragments_and_observation_debug_fields():
    context = make_context()
    context.update(
        {
            "context_schema": {"debug": True},
            "metadata": {"run_id": "run_test"},
            "disabled_fragments": [],
            "phase_metrics": [],
            "phase_completion_evaluation": {
                "metrics_count": 0,
                "measurable_count": 0,
                "all_within_tolerance": False,
                "evaluations": [{"status": "requires_model_judgment"}],
            },
            "phase_execution_summary": {
                "phase": "landmark_phase",
                "window_count": 7,
                "measurements": {"forward_m": 10.0, "right_m": 2.0},
                "tool_facts": {"tools_executed": {"uav.move_body": 3}},
            },
            "latest_observation_summary": {
                "source": "automatic_phase_window",
                "timestamp": "t1",
                "modalities": ["rgb", "pose"],
                "pose": {"x": 1.0},
                "frames": {
                    "rgb": {
                        "path": "runs/example/frame.png",
                        "artifact_ref": "runs/example/frame.png",
                        "case_id": "case-debug",
                        "source": "uav_flow_benchmark",
                    }
                },
            },
            "fragments": [
                {
                    "name": "image",
                    "enabled": True,
                    "source": "DefaultImageContextModule",
                    "content": {"selected_frames": [{"role": "current_observation"}]},
                    "model_inputs": [],
                    "facts": [],
                    "inferences": [],
                    "goals": [],
                    "risks": [],
                    "artifact_refs": ["runs/example/frame.png"],
                    "budget": {"tokens_estimate": 300},
                    "confidence": "medium",
                    "warnings": [],
                    "metadata": {"debug_only": True},
                }
            ],
        }
    )

    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    text = json.dumps(text_context, ensure_ascii=False)
    assert "context_schema" not in text_context
    assert "metadata" not in text_context
    assert "disabled_fragments" not in text_context
    assert "phase_metrics" not in text_context
    assert "all_within_tolerance" not in text_context["phase_completion_evaluation"]
    assert text_context["phase_completion_evaluation"]["deterministic_status"] == "not_measurable"
    assert text_context["phase_execution_summary"] == {
        "phase": "landmark_phase",
        "window_count": 7,
        "tools_executed": {"uav.move_body": 3},
    }
    assert text_context["latest_observation_summary"] == {
        "source": "automatic_phase_window",
        "timestamp": "t1",
        "modalities": ["rgb", "pose"],
        "pose": {"x": 1.0},
        "has_rgb": True,
        "has_depth": False,
    }
    assert text_context["fragments"] == [{"name": "image", "content": {"selected_frames": [{"role": "current_observation"}]}}]
    for forbidden in ["artifact_refs", "budget", "confidence", "source_path", "runs/example", "case-debug"]:
        assert forbidden not in text


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


def test_model_user_content_inserts_label_before_semantic_image_input():
    image_data = "data:image/png;base64," + base64.b64encode(b"fake-png").decode("ascii")
    context = make_context()
    context["fragments"] = [
        {
            "name": "image",
            "model_inputs": [
                {
                    "type": "image",
                    "input_id": "image_1_current_observation",
                    "role": "current_observation",
                    "usage": "Use as the current visual evidence for this decision.",
                    "mime_type": "image/png",
                    "encoding": "data_url",
                    "data": image_data,
                }
            ],
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, list)
    assert len(content) == 3
    text_context = json.loads(content[0]["text"])
    model_input = text_context["fragments"][0]["model_inputs"][0]
    assert model_input["input_id"] == "image_1_current_observation"
    assert model_input["role"] == "current_observation"
    assert model_input["usage"] == "Use as the current visual evidence for this decision."
    assert model_input["data"].startswith("<omitted encoded model input data")
    assert content[1] == {
        "type": "text",
        "text": (
            "Model input label: input_id=image_1_current_observation; "
            "role=current_observation; "
            "usage=Use as the current visual evidence for this decision."
        ),
    }
    assert content[2] == {"type": "image_url", "image_url": {"url": image_data}}


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


def test_model_action_planner_rejects_legacy_action_prompt_override():
    with pytest.raises(ModelActionPlannerConfigError, match="coupled action/phase prompt only"):
        ModelActionPlanner(ModelConfig(model_name="stub", api_key="test-key"), prompt_path="legacy.txt")


def test_model_action_planner_combines_coupled_prompt_fragments_in_order():
    planner = ModelActionPlanner(ModelConfig(model_name="stub", api_key="test-key"))
    prompt = planner.coupled_action_phase_system_prompt

    overview_index = prompt.index("experimental coupled action-and-phase planner")
    phase_index = prompt.index("--- Phase Decision Contract ---")
    action_index = prompt.index("--- Action Planning Contract ---")

    assert overview_index < phase_index < action_index
    assert "Phase Decision Guide" in prompt
    assert "Allowed Tools For Continue Decisions" in prompt
    assert "coupled_action_phase_planner_system.txt" not in prompt


def test_model_action_planner_requires_api_key():
    planner = ModelActionPlanner(ModelConfig(model_name="stub", api_key=""))

    with pytest.raises(ModelActionPlannerConfigError, match="API_KEY"):
        planner.plan_next_with_phase_decision(make_state(), {"current_phase": {"name": "orient_leftward"}})


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

    plan = make_tool_plan(planner, state, make_context())

    assert plan.phase == "orient_leftward"
    assert plan.steps[0].tool == "uav.turn_relative"
    assert plan.steps[0].args["yaw_deg"] == -15
    assert len(plan.steps) == 1
    assert "phase_transition" not in plan.to_dict()
    assert plan.agent_messages[0].type == "progress"


def test_short_horizon_plan_direct_constructor_has_no_planner_side_transition():
    plan = ShortHorizonPlan("orient_leftward", [])

    assert "phase_transition" not in plan.to_dict()
    assert plan.action_window.scope == "current_phase_only"
    assert plan.to_dict()["action_window"]["intent"]


def test_model_action_planner_rejects_missing_action_window():
    planner = StubModelActionPlanner(
        {
            "_omit_action_window": True,
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="requires action_window"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_missing_agent_messages():
    planner = StubModelActionPlanner(
        {
            "_omit_agent_messages": True,
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="requires agent_messages"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_invalid_action_window_scope():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "action_window": make_action_window(scope="whole_task"),
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="current_phase_only"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


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
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_unknown_tool():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.fake", "args": {}, "reason": "bad"}],
        }
    )

    with pytest.raises(ValueError, match="not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


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
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_phase_mismatch():
    planner = StubModelActionPlanner(
        {
            "phase": "different_phase",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="phase mismatch"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


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
        planner.plan_next_with_phase_decision(make_state(), make_context())


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
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_hover_with_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {"duration": 1.0}, "reason": "bad"}],
        }
    )

    with pytest.raises(ValueError, match="uav.hover args must be"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


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
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_model_facing_move_body_direction_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "backward", "duration": 1.5},
                    "reason": "legacy named motion",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="direction shorthand is not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_accepts_numeric_move_for_5m_nominal_scale():
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
                    "args": {"vx": 2.0, "vy": 0.0, "vz": 0.0, "duration": 2.5},
                    "reason": "translate about 5 meters using nominal 2 m/s scale",
                }
            ],
        }
    )

    plan = make_tool_plan(planner, state, context)

    assert plan.steps[0].args == {"vx": 2.0, "vy": 0.0, "vz": 0.0, "duration": 2.5}


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

    plan = make_tool_plan(planner, state, make_context())

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

    with pytest.raises(ValueError, match="direction shorthand is not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


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

    with pytest.raises(ValueError, match="direction shorthand is not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


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

    with pytest.raises(ValueError, match="direction shorthand is not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_repairs_contract_violation_on_first_attempt():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "repaired hold"}],
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

    plan = make_tool_plan(planner, make_state(), make_context())

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
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "observe", "modalities": ["rgb"], "reason": "bad"}],
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "finally safe"}],
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

    plan = make_tool_plan(planner, make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"
    assert len(planner.last_repair_attempts) == 3
    assert planner.last_repair_result["successful_attempt"] == 3


def test_model_action_planner_repair_failure_exposes_structured_error():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 1.0}, "reason": "bad"}],
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 2.0}, "reason": "bad"}],
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 3.0}, "reason": "bad"}],
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
        planner.plan_next_with_phase_decision(make_state(), make_context())

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
    }
    repair_planner = StubRepairPlanner([bad_plan, bad_plan])
    planner = StubModelActionPlanner(
        bad_plan,
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    with pytest.raises(ModelRepairPlannerError, match="repeated"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


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
        planner.plan_next_with_phase_decision(make_state(), make_context())

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
        planner.plan_next_with_phase_decision(make_state(), make_context())

    assert exc_info.value.violations[-1]["code"] == "repair_plan_parse_error"
    assert "requires agent_messages" in exc_info.value.violations[-1]["message"]


def test_model_action_planner_has_no_independent_phase_judge_method():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    assert not hasattr(planner, "judge_phase_transition")


def test_coupled_action_phase_plan_accepts_continue_with_tool_steps():
    plan = CoupledActionPhasePlan.from_dict(
        {
            "phase": "orient_leftward",
            "phase_decision": {"decision": "continue_current_phase", "reason": "target not aligned yet"},
            "action_window": make_action_window(),
            "agent_messages": make_agent_messages(),
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        },
        require_agent_messages=True,
    )

    short_plan = plan.to_short_horizon_plan()

    assert plan.phase_decision.decision == "continue_current_phase"
    assert short_plan.steps[0].tool == "uav.hover"
    assert "phase_decision" in plan.to_dict()
    assert "phase_transition" not in plan.to_dict()


def test_coupled_action_phase_plan_accepts_advance_with_empty_steps():
    plan = CoupledActionPhasePlan.from_dict(
        {
            "phase": "orient_leftward",
            "phase_decision": {"decision": ADVANCE_TO_NEXT_PHASE, "reason": "phase already complete"},
            "agent_messages": make_agent_messages(),
            "steps": [],
        }
    )

    assert plan.phase_decision.decision == ADVANCE_TO_NEXT_PHASE
    assert plan.steps == []


def test_coupled_action_phase_plan_rejects_transition_with_steps():
    with pytest.raises(ValueError, match="empty steps"):
        CoupledActionPhasePlan.from_dict(
            {
                "phase": "orient_leftward",
                "phase_decision": {"decision": COMPLETE_TASK, "reason": "done"},
                "agent_messages": make_agent_messages(),
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "bad"}],
            }
        )


def test_coupled_action_phase_plan_rejects_continue_without_steps():
    with pytest.raises(ValueError, match="1-3 tool steps"):
        CoupledActionPhasePlan.from_dict(
            {
                "phase": "orient_leftward",
                "phase_decision": {"decision": "continue_current_phase", "reason": "not done"},
                "action_window": make_action_window(),
                "agent_messages": make_agent_messages(),
                "steps": [],
            }
        )


def test_coupled_action_phase_plan_rejects_invalid_decision():
    with pytest.raises(ValueError, match="Unsupported phase_decision decision"):
        CoupledActionPhasePlan.from_dict(
            {
                "phase": "orient_leftward",
                "phase_decision": {"decision": "invalid_transition_request", "reason": "old design"},
                "agent_messages": make_agent_messages(),
                "steps": [],
            }
        )


def test_model_action_planner_coupled_continue_uses_tool_contract_validator():
    planner = StubModelActionPlanner(
        {
            "phase": "unused",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "unused"}],
        }
    )
    planner._call_coupled_action_phase_openai_compatible = lambda context: {
        "phase": "orient_leftward",
        "phase_decision": {"decision": "continue_current_phase", "reason": "need one more action"},
        "action_window": make_action_window(),
        "agent_messages": make_agent_messages(),
        "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
    }

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.phase_decision.decision == "continue_current_phase"
    assert plan.steps[0].tool == "uav.hover"


def test_model_action_planner_coupled_continue_repairs_tool_contract_violation():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "action_window": make_action_window(
                    intent="repair to one safe observation window",
                    requires_intermediate_observation=True,
                    expected_effect="collect one new observation after holding",
                ),
                "agent_messages": make_agent_messages(content="我会改用一个安全的短动作继续确认。"),
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "single repaired step"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "unused",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "unused"}],
        },
        max_repair_attempts=1,
        repair_planner=repair_planner,
    )
    planner._call_coupled_action_phase_openai_compatible = lambda context: {
        "phase": "orient_leftward",
        "phase_decision": {"decision": "continue_current_phase", "reason": "need fresh evidence"},
        "action_window": make_action_window(
            requires_intermediate_observation=True,
            expected_effect="bad multi-step plan asks for a mid-window observation",
        ),
        "agent_messages": make_agent_messages(),
        "steps": [
            {"kind": "tool", "tool": "uav.turn_relative", "args": {"yaw_deg": -5}, "reason": "first"},
            {"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "second"},
        ],
    }

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.phase_decision.decision == "continue_current_phase"
    assert plan.phase_decision.reason == "need fresh evidence"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "uav.hover"
    assert plan.action_window.requires_intermediate_observation is True
    assert planner.last_repair_result["status"] == "repaired"
    assert repair_planner.contexts[0].violation["code"] == "action_window_intermediate_observation_requires_single_step"


def test_model_action_planner_coupled_transition_with_steps_is_not_tool_repaired():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "should not be used"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "unused",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "unused"}],
        },
        max_repair_attempts=1,
        repair_planner=repair_planner,
    )
    planner._call_coupled_action_phase_openai_compatible = lambda context: {
        "phase": "orient_leftward",
        "phase_decision": {"decision": ADVANCE_TO_NEXT_PHASE, "reason": "done"},
        "agent_messages": make_agent_messages(),
        "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "bad transition action"}],
    }

    with pytest.raises(ValueError, match="empty steps"):
        planner.plan_next_with_phase_decision(make_state(), make_context())

    assert repair_planner.contexts == []


def test_model_action_planner_coupled_rejects_phase_mismatch_without_tool_steps():
    planner = StubModelActionPlanner(
        {
            "phase": "unused",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "unused"}],
        }
    )
    planner._call_coupled_action_phase_openai_compatible = lambda context: {
        "phase": "wrong_phase",
        "phase_decision": {"decision": ADVANCE_TO_NEXT_PHASE, "reason": "bad phase"},
        "agent_messages": make_agent_messages(),
        "steps": [],
    }

    with pytest.raises(ValueError, match="phase mismatch"):
        planner.plan_next_with_phase_decision(make_state(), make_context())
