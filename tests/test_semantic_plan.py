"""Test lightweight semantic plan validation."""

from __future__ import annotations

import pytest

from src.agent.semantic.validator import SemanticPlanValidationError, semantic_plan_from_dict


def make_plan(**overrides):
    data = {
        "intent": "navigate_by_visual_landmark",
        "task_goal": {
            "target": "black street lamp",
            "target_attributes": {"color": "black"},
            "final_state": "uav_facing_target_and_hovering",
        },
        "phases": [
            {
                "name": "search_landmark",
                "goal": "identify the black street lamp candidate",
                "done_when": ["black street lamp or strong candidate is visible"],
                "observe": ["latest_observation"],
                "continue_when": ["target identity is unclear"],
                "metrics": [],
            }
        ],
        "uncertainties": ["target_identity_requires_observation"],
    }
    data.update(overrides)
    return data


def test_semantic_plan_accepts_valid_plan():
    plan = semantic_plan_from_dict(make_plan())

    assert plan.intent == "navigate_by_visual_landmark"
    assert plan.task_goal.target == "black street lamp"
    assert plan.phases[0].index == 1
    assert plan.phases[0].total == 1
    assert plan.phases[0].done_when == ["black street lamp or strong candidate is visible"]
    assert "action_bias" not in plan.phases[0].to_dict()


def test_semantic_plan_ignores_legacy_action_bias_input():
    data = make_plan()
    data["phases"][0]["action_bias"] = "legacy action preference should not survive"

    plan = semantic_plan_from_dict(data)

    assert "action_bias" not in plan.to_dict()["phases"][0]


def test_semantic_plan_accepts_free_model_phase_names():
    plan = semantic_plan_from_dict(
        make_plan(
            phases=[
                {
                    "index": 1,
                    "total": 2,
                    "name": "turn_to_search_possible_lamp_area",
                    "goal": "Use the user's right-turn clue as semantic context for later action planning.",
                    "done_when": ["right-side search area is visible"],
                    "observe": ["latest_observation"],
                    "continue_when": ["right-side search area is not in view"],
                    "metrics": [],
                },
                {
                    "index": 2,
                    "total": 2,
                    "name": "visually_identify_black_streetlight",
                    "goal": "Use observation context to identify candidate black streetlights.",
                    "done_when": ["black streetlight candidate is visible"],
                    "observe": ["latest_observation"],
                    "continue_when": ["candidate identity is unclear"],
                    "metrics": [],
                },
            ]
        )
    )

    assert [phase.index for phase in plan.phases] == [1, 2]
    assert [phase.total for phase in plan.phases] == [2, 2]
    assert plan.phases[0].name == "turn_to_search_possible_lamp_area"


def test_semantic_plan_canonicalizes_model_phase_total():
    plan = semantic_plan_from_dict(
        make_plan(
            phases=[
                {
                    "index": 1,
                    "total": 9,
                    "name": "free_phase",
                    "goal": "semantic stage",
                    "done_when": ["semantic stage is done"],
                    "observe": ["latest_observation"],
                    "continue_when": ["semantic stage is not done"],
                    "metrics": [],
                }
            ]
        )
    )

    assert [phase.total for phase in plan.phases] == [1]


def test_semantic_plan_canonicalizes_model_phase_index():
    plan = semantic_plan_from_dict(
        make_plan(
            phases=[
                {
                    "index": 2,
                    "total": 1,
                    "name": "free_phase",
                    "goal": "semantic stage",
                    "done_when": ["semantic stage is done"],
                    "observe": ["latest_observation"],
                    "continue_when": ["semantic stage is not done"],
                    "metrics": [],
                }
            ]
        )
    )

    assert [phase.index for phase in plan.phases] == [1]


def test_semantic_plan_rejects_missing_target():
    with pytest.raises(SemanticPlanValidationError, match="task_goal.target"):
        semantic_plan_from_dict(make_plan(task_goal={"target": "", "final_state": "uav_facing_target_and_hovering"}))


def test_semantic_plan_rejects_missing_final_state():
    with pytest.raises(SemanticPlanValidationError, match="final_state"):
        semantic_plan_from_dict(make_plan(task_goal={"target": "black street lamp", "final_state": ""}))


def test_semantic_plan_rejects_empty_phases():
    with pytest.raises(SemanticPlanValidationError, match="phases"):
        semantic_plan_from_dict(make_plan(phases=[]))


def test_semantic_plan_rejects_phase_without_goal():
    data = make_plan()
    data["phases"][0]["goal"] = ""
    with pytest.raises(SemanticPlanValidationError, match="goal is required"):
        semantic_plan_from_dict(data)


def test_semantic_plan_rejects_phase_without_done_when():
    data = make_plan()
    data["phases"][0]["done_when"] = []
    with pytest.raises(SemanticPlanValidationError, match="done_when"):
        semantic_plan_from_dict(data)


def test_semantic_plan_rejects_primitive_action_tokens():
    with pytest.raises(SemanticPlanValidationError, match="primitive action"):
        semantic_plan_from_dict(
            make_plan(
                phases=[
                    {
                        "name": "uav.turn_relative",
                        "goal": "turn_relative 30 degrees",
                        "done_when": ["uav.turn_relative finishes"],
                        "observe": ["phase_execution_summary"],
                        "continue_when": [],
                        "metrics": [],
                    }
                ]
            )
        )


def test_semantic_plan_rejects_env_observe_token():
    with pytest.raises(SemanticPlanValidationError, match="primitive action"):
        semantic_plan_from_dict(
            make_plan(
                intent="observe_and_clarify",
                task_goal={
                    "target": "right-side area target",
                    "target_attributes": {},
                    "final_state": "uav_hovering_or_waiting_for_clarification",
                },
                phases=[
                    {
                        "name": "observe_area",
                        "goal": "call env.observe",
                        "done_when": ["env.observe returns"],
                        "observe": ["latest_observation"],
                        "continue_when": [],
                        "metrics": [],
                    }
                ],
            )
        )
