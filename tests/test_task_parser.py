"""Test M3 task parsing into semantic plans."""

from __future__ import annotations

import pytest

from src.agent.semantic.task_parser import TaskParser, TaskParserConfigError
from src.agent.semantic.schemas import SemanticPhase
from src.runtime.configs.model_config import ModelConfig


def test_rule_based_parser_parses_black_street_lamp_task():
    parser = TaskParser(mode="rule_based")

    result = parser.parse("前面右转会有一个黑色的路灯，右转面向路灯后停下")

    plan = result.plan
    assert plan.intent == "navigate_by_visual_landmark"
    assert plan.goal.target.description == "黑色路灯"
    assert plan.goal.target.attributes == {"color": "black", "category": "street_lamp"}
    assert plan.goal.final_state == "uav_facing_target_and_hovering"
    assert [phase.name for phase in plan.phases] == [
        "orient_right",
        "search_landmark",
        "align_to_landmark",
        "stop",
    ]
    assert [phase.index for phase in plan.phases] == [1, 2, 3, 4]
    assert [phase.total for phase in plan.phases] == [4, 4, 4, 4]
    assert "target_identity_requires_observation" in plan.uncertainties


def test_rule_based_parser_parses_red_vehicle_task():
    parser = TaskParser(mode="rule_based")

    result = parser.parse("前方寻找红色车辆，靠近但保持安全距离")

    plan = result.plan
    assert plan.intent == "navigate_by_visual_landmark"
    assert plan.goal.target.description == "红色车辆"
    assert plan.goal.target.attributes == {"color": "red", "category": "vehicle"}
    assert plan.goal.final_state == "uav_near_target_with_safe_distance"
    assert [phase.name for phase in plan.phases] == ["search_landmark", "approach_landmark", "stop"]
    assert [phase.index for phase in plan.phases] == [1, 2, 3]
    assert [phase.total for phase in plan.phases] == [3, 3, 3]
    assert "safe_distance_requires_depth_or_user_constraint" in plan.uncertainties


def test_rule_based_parser_parses_building_entrance_task():
    parser = TaskParser(mode="rule_based")

    result = parser.parse("左转寻找建筑入口并停在入口前")

    plan = result.plan
    assert plan.goal.target.description == "建筑入口"
    assert plan.goal.target.attributes == {"category": "building_entrance"}
    assert plan.goal.final_state == "uav_hovering_in_front_of_target"
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
    assert plan.goal.target.description == "右侧区域中的目标"
    assert plan.goal.target.attributes == {}
    assert plan.goal.final_state == "uav_hovering_or_waiting_for_clarification"
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


def test_semantic_phase_accepts_completion_criteria():
    phase = SemanticPhase.from_dict(
        {
            "index": 1,
            "total": 1,
            "name": "move_forward",
            "meaning": "move forward five meters",
            "completion_criteria": {
                "kind": "translation",
                "axis": "forward",
                "target": 5.0,
                "unit": "m",
                "tolerance": 1.0,
                "correction_allowed": True,
            },
        }
    )

    assert phase.completion_criteria == [
        {
            "kind": "translation",
            "axis": "forward",
            "target": 5.0,
            "unit": "m",
            "tolerance": 1.0,
            "correction_allowed": True,
        }
    ]
    assert phase.to_dict()["completion_criteria"][0]["axis"] == "forward"
