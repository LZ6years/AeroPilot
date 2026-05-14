"""Test semantic plan validation."""

from __future__ import annotations

import pytest

from src.agent.semantic.validator import SemanticPlanValidationError, semantic_plan_from_dict


def test_semantic_plan_accepts_valid_plan():
    plan = semantic_plan_from_dict(
        {
            "intent": "navigate_by_visual_landmark",
            "goal": {
                "target": {"description": "黑色路灯", "attributes": {"color": "black"}},
                "final_state": "uav_facing_target_and_hovering",
            },
            "phases": [{"name": "search_landmark", "meaning": "寻找黑色路灯"}],
            "uncertainties": ["target_identity_requires_observation"],
        }
    )

    assert plan.intent == "navigate_by_visual_landmark"
    assert plan.phases[0].index == 1
    assert plan.phases[0].total == 1


def test_semantic_plan_accepts_free_model_phase_names():
    plan = semantic_plan_from_dict(
        {
            "intent": "navigate_by_visual_landmark",
            "goal": {
                "target": {"description": "black streetlight"},
                "final_state": "facing_target_and_hovering",
            },
            "phases": [
                {
                    "index": 1,
                    "total": 2,
                    "name": "turn_to_search_possible_lamp_area",
                    "meaning": "Use the user's right-turn clue as semantic context for later action planning",
                },
                {
                    "index": 2,
                    "total": 2,
                    "name": "visually_identify_black_streetlight",
                    "meaning": "Use observation context to identify candidate black streetlights",
                },
            ],
        }
    )

    assert [phase.index for phase in plan.phases] == [1, 2]
    assert [phase.total for phase in plan.phases] == [2, 2]
    assert plan.phases[0].name == "turn_to_search_possible_lamp_area"


def test_semantic_plan_rejects_wrong_phase_total():
    with pytest.raises(SemanticPlanValidationError, match="total"):
        semantic_plan_from_dict(
            {
                "intent": "navigate_by_visual_landmark",
                "goal": {
                    "target": {"description": "black streetlight"},
                    "final_state": "facing_target_and_hovering",
                },
                "phases": [
                    {"index": 1, "total": 9, "name": "free_phase", "meaning": "semantic stage"}
                ],
            }
        )


def test_semantic_plan_rejects_non_consecutive_phase_index():
    with pytest.raises(SemanticPlanValidationError, match="index"):
        semantic_plan_from_dict(
            {
                "intent": "navigate_by_visual_landmark",
                "goal": {
                    "target": {"description": "black streetlight"},
                    "final_state": "facing_target_and_hovering",
                },
                "phases": [
                    {"index": 2, "total": 1, "name": "free_phase", "meaning": "semantic stage"}
                ],
            }
        )


def test_semantic_plan_rejects_missing_target():
    with pytest.raises(SemanticPlanValidationError, match="target.description"):
        semantic_plan_from_dict(
            {
                "intent": "navigate_by_visual_landmark",
                "goal": {"target": {}, "final_state": "uav_facing_target_and_hovering"},
                "phases": [{"name": "search_landmark", "meaning": "寻找目标"}],
            }
        )


def test_semantic_plan_rejects_missing_final_state():
    with pytest.raises(SemanticPlanValidationError, match="final_state"):
        semantic_plan_from_dict(
            {
                "intent": "navigate_by_visual_landmark",
                "goal": {"target": {"description": "黑色路灯"}, "final_state": ""},
                "phases": [{"name": "search_landmark", "meaning": "寻找黑色路灯"}],
            }
        )


def test_semantic_plan_rejects_empty_phases():
    with pytest.raises(SemanticPlanValidationError, match="phases"):
        semantic_plan_from_dict(
            {
                "intent": "navigate_by_visual_landmark",
                "goal": {
                    "target": {"description": "黑色路灯"},
                    "final_state": "uav_facing_target_and_hovering",
                },
                "phases": [],
            }
        )


def test_semantic_plan_rejects_primitive_action_tokens():
    with pytest.raises(SemanticPlanValidationError, match="primitive action"):
        semantic_plan_from_dict(
            {
                "intent": "navigate_by_visual_landmark",
                "goal": {
                    "target": {"description": "黑色路灯"},
                    "final_state": "uav_facing_target_and_hovering",
                },
                "phases": [{"name": "uav.turn_relative", "meaning": "turn_relative 30 degrees"}],
            }
        )


def test_semantic_plan_rejects_env_observe_token():
    with pytest.raises(SemanticPlanValidationError, match="primitive action"):
        semantic_plan_from_dict(
            {
                "intent": "observe_and_clarify",
                "goal": {
                    "target": {"description": "右侧区域中的目标"},
                    "final_state": "uav_hovering_or_waiting_for_clarification",
                },
                "phases": [{"name": "observe_area", "meaning": "call env.observe"}],
            }
        )
