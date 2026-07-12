"""Test task parsing into lightweight semantic plans."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.semantic import SemanticPlanValidationError, semantic_plan_from_dict
from src.agent.semantic.task_parser import TaskParser, TaskParserConfigError
from src.agent.semantic.schemas import SemanticPhase
from src.runtime.configs.model_config import ModelConfig


def test_rule_based_parser_parses_black_street_lamp_task():
    parser = TaskParser(mode="rule_based")

    result = parser.parse("前面右转会有一个黑色的路灯，右转面向路灯后停下")

    plan = result.plan
    assert plan.intent == "navigate_by_visual_landmark"
    assert plan.task_goal.target == "black street lamp"
    assert plan.task_goal.target_attributes == {"color": "black", "category": "street_lamp"}
    assert plan.task_goal.final_state == "uav_facing_target_and_hovering"
    assert [phase.name for phase in plan.phases] == [
        "orient_right",
        "search_landmark",
        "align_to_landmark",
        "stop",
    ]
    assert [phase.index for phase in plan.phases] == [1, 2, 3, 4]
    assert [phase.total for phase in plan.phases] == [4, 4, 4, 4]
    assert plan.phases[2].done_when == ["street lamp remains visible", "view is roughly aligned with the target"]
    assert "target_identity_requires_observation" in plan.uncertainties


def test_rule_based_parser_parses_red_vehicle_task():
    parser = TaskParser(mode="rule_based")

    result = parser.parse("前方寻找红色车辆，靠近但保持安全距离")

    plan = result.plan
    assert plan.intent == "navigate_by_visual_landmark"
    assert plan.task_goal.target == "red vehicle"
    assert plan.task_goal.target_attributes == {"color": "red", "category": "vehicle"}
    assert plan.task_goal.final_state == "uav_near_target_with_safe_distance"
    assert [phase.name for phase in plan.phases] == ["search_landmark", "approach_landmark", "stop"]
    assert [phase.index for phase in plan.phases] == [1, 2, 3]
    assert [phase.total for phase in plan.phases] == [3, 3, 3]
    assert plan.phases[1].observe == ["latest_observation", "phase_start_reference", "phase_execution_summary"]
    assert "safe_distance_requires_depth_or_user_constraint" in plan.uncertainties


def test_rule_based_parser_parses_building_entrance_task():
    parser = TaskParser(mode="rule_based")

    result = parser.parse("左转寻找建筑入口并停在入口前")

    plan = result.plan
    assert plan.task_goal.target == "building entrance"
    assert plan.task_goal.target_attributes == {"category": "building_entrance"}
    assert plan.task_goal.final_state == "uav_hovering_in_front_of_target"
    assert [phase.name for phase in plan.phases] == [
        "orient_left",
        "search_landmark",
        "approach_landmark",
        "stop",
    ]
    assert [phase.index for phase in plan.phases] == [1, 2, 3, 4]
    assert "front_of_target_requires_observation_and_depth" in plan.uncertainties


def test_rule_based_parser_parses_observe_and_clarify_task():
    parser = TaskParser(mode="rule_based")

    result = parser.parse("观察右侧区域，不确定目标时停下询问")

    plan = result.plan
    assert plan.intent == "observe_and_clarify"
    assert plan.task_goal.target == "right-side area target"
    assert plan.task_goal.target_attributes == {}
    assert plan.task_goal.final_state == "uav_hovering_or_waiting_for_clarification"
    assert [phase.name for phase in plan.phases] == [
        "orient_right",
        "observe_area",
        "clarify_if_uncertain",
    ]
    assert [phase.total for phase in plan.phases] == [3, 3, 3]
    assert "user_clarification_may_be_required" in plan.uncertainties


def test_model_parser_requires_api_key():
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(
            task_parser_mode="model",
            provider="openai_compatible",
            model_name="test-model",
            api_key="",
            base_url="https://example.test/v1",
        ),
    )

    with pytest.raises(TaskParserConfigError, match="FLYAGENT_MODEL_API_KEY"):
        parser.parse("前面右转会有一个黑色的路灯，右转面向路灯后停下")


def test_semantic_phase_accepts_metrics():
    phase = SemanticPhase.from_dict(
        {
            "index": 1,
            "total": 1,
            "name": "move_forward",
            "goal": "move forward five meters",
            "done_when": ["forward distance reaches about five meters"],
            "observe": ["phase_execution_summary"],
            "continue_when": ["forward progress is below target"],
            "metrics": [
                {
                    "name": "forward_distance",
                    "target": 5.0,
                    "unit": "m",
                    "tolerance": 1.0,
                    "correction_allowed": True,
                }
            ],
        }
    )

    assert phase.metrics == [
        {
            "name": "forward_distance",
            "target": 5.0,
            "unit": "m",
            "tolerance": 1.0,
            "correction_allowed": True,
        }
    ]
    assert phase.to_dict()["metrics"][0]["name"] == "forward_distance"


def test_model_parser_prompt_requires_explicit_numeric_metrics():
    parser = TaskParser(mode="rule_based")

    prompt = parser.system_prompt

    assert "The preceding system message is the authoritative FlyAgent identity" in prompt
    assert "Numeric provenance is strict" in prompt
    assert "every numeric target or quantity" in prompt
    assert "An explicit number alone is not sufficient" in prompt
    assert "target-relative numbers such as orbit radius" in prompt
    assert "Do not convert them into planar_distance" in prompt
    assert "do not generate planar_distance=2*pi*N" in prompt
    assert "do not make exact radius confirmation a hard blocker" in prompt
    assert "Do not describe fixed tool sequences or exact paths" in prompt
    assert "negative target" in prompt
    assert '"direction":' not in prompt
    assert "rearward displacement" in prompt
    assert "backward movement" not in prompt
    assert "Numeric angle role rules" in prompt
    assert "A numeric angle is not automatically a yaw angle" in prompt
    assert "orientation_role" in prompt
    assert "horizontal_displacement_role" in prompt
    assert "vertical_path_role" in prompt
    assert "forward_distance = D*cos(A)" in prompt
    assert "avoid defaulting to yaw_delta" in prompt
    assert "move to the trees" in prompt
    assert "Do not invent a distance metric." in prompt


def test_model_parser_builds_flyagent_and_contract_as_separate_system_messages():
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(
            task_parser_mode="model",
            provider="openai_compatible",
            model_name="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        ),
    )

    messages = parser._build_model_messages("Progress past the visual landmark from the right side")
    expected_rules = Path("memory/FLYAGENT.md").read_text(encoding="utf-8-sig").strip()

    assert [message["role"] for message in messages] == ["system", "system", "user"]
    assert messages[0]["content"] == expected_rules
    assert messages[1]["content"] == parser.system_prompt
    assert messages[2]["content"] == "Progress past the visual landmark from the right side"
    assert "你是 FlyAgent" in messages[0]["content"]
    assert "Required JSON shape" in messages[1]["content"]
    assert "你是 FlyAgent" not in messages[1]["content"]


def test_model_parser_custom_contract_keeps_default_flyagent_rules(tmp_path):
    custom_prompt = tmp_path / "custom_task_parser.txt"
    custom_prompt.write_text("Return one custom semantic JSON object.", encoding="utf-8")
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(
            task_parser_mode="model",
            provider="openai_compatible",
            model_name="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
            task_parser_prompt_path=str(custom_prompt),
        ),
    )

    messages = parser._build_model_messages("test instruction")

    assert messages[0]["content"] == Path("memory/FLYAGENT.md").read_text(encoding="utf-8-sig").strip()
    assert messages[1]["content"] == "Return one custom semantic JSON object."


@pytest.mark.parametrize(
    ("file_content", "error"),
    [(None, "not found"), ("", "is empty")],
)
def test_model_parser_requires_nonempty_flyagent_rules(tmp_path, monkeypatch, file_content, error):
    rules_path = tmp_path / "FLYAGENT.md"
    if file_content is not None:
        rules_path.write_text(file_content, encoding="utf-8")
    monkeypatch.setattr("src.agent.semantic.task_parser.DEFAULT_AGENT_RULES_PATH", rules_path)

    with pytest.raises(TaskParserConfigError, match=error):
        TaskParser(
            mode="model",
            model_config=ModelConfig(
                task_parser_mode="model",
                provider="openai_compatible",
                model_name="test-model",
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
        )


def test_rule_based_parser_does_not_load_flyagent_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.agent.semantic.task_parser.DEFAULT_AGENT_RULES_PATH",
        tmp_path / "missing" / "FLYAGENT.md",
    )

    parser = TaskParser(mode="rule_based")
    result = parser.parse("前面右转会有一个黑色的路灯，右转面向路灯后停下")

    assert parser.agent_rules == ""
    assert result.plan.task_goal.target == "black street lamp"


def test_semantic_validator_allows_natural_uav_sentence_but_rejects_tool_namespace():
    semantic_plan_from_dict(
        {
            "intent": "navigate_by_visual_landmark",
            "task_goal": {
                "target": "front trees",
                "target_attributes": {},
                "final_state": "uav_near_front_trees",
            },
            "phases": [
                {
                    "index": 1,
                    "total": 1,
                    "name": "approach_front_trees",
                    "goal": "Move the UAV toward the trees on the front side using visual evidence.",
                    "done_when": ["trees are identifiable", "approach progress is visible"],
                    "observe": ["latest_observation"],
                    "continue_when": ["trees are unclear"],
                    "metrics": [],
                }
            ],
            "uncertainties": [],
        }
    )
    semantic_plan_from_dict(
        {
            "intent": "movement",
            "task_goal": {
                "target": "land four meters lower",
                "target_attributes": {"distance_m": 4.0},
                "final_state": "uav_stable_after_landing_descent",
            },
            "phases": [
                {
                    "index": 1,
                    "total": 1,
                    "name": "controlled_landing_descent",
                    "goal": "Land by descending four meters and stabilizing.",
                    "done_when": ["the UAV has descended four meters", "the UAV is stable"],
                    "observe": ["phase_execution_summary"],
                    "continue_when": ["the requested descent is incomplete"],
                    "metrics": [{"name": "up_distance", "target": -4.0, "unit": "m", "tolerance": 0.5}],
                }
            ],
            "uncertainties": [],
        }
    )

    with pytest.raises(SemanticPlanValidationError, match="uav.move_body"):
        semantic_plan_from_dict(
            {
                "intent": "navigate_by_visual_landmark",
                "task_goal": {
                    "target": "front trees",
                    "target_attributes": {},
                    "final_state": "uav_near_front_trees",
                },
                "phases": [
                    {
                        "index": 1,
                        "total": 1,
                        "name": "approach_front_trees",
                        "goal": "Call uav.move_body toward the trees.",
                        "done_when": ["uav.move_body has been called"],
                        "observe": ["latest_observation"],
                        "continue_when": [],
                        "metrics": [],
                    }
                ],
                "uncertainties": [],
            }
        )
