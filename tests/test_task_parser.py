"""Test task parsing into lightweight semantic plans."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.semantic import SemanticPlanValidationError, semantic_plan_from_dict
from src.agent.semantic.parser_contracts import (
    SemanticPlanStructure,
    TaskParserContractError,
    merge_semantic_plan,
    validate_phase_criteria,
)
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
    assert "target identity is unclear" in plan.phases[2].continue_when
    assert "uncertainties" not in plan.to_dict()


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
    assert "safe approach relation is not yet plausible" in plan.phases[1].continue_when
    assert "uncertainties" not in plan.to_dict()


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
    assert "entrance identity is unclear" in plan.phases[2].continue_when
    assert "uncertainties" not in plan.to_dict()


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
    assert "uncertainty is acknowledged and the UAV is stable" in plan.phases[2].done_when
    assert "uncertainties" not in plan.to_dict()


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


def test_model_parser_prompts_split_structure_from_phase_criteria():
    structure_prompt = Path(
        "src/agent/semantic/prompts/task_parser_structure_system.txt"
    ).read_text(encoding="utf-8")
    criteria_prompt = Path(
        "src/agent/semantic/prompts/task_parser_phase_criteria_system.txt"
    ).read_text(encoding="utf-8")

    assert "Output only intent, task_goal, and phase index/total/name/goal" in structure_prompt
    assert "Do not output done_when, observe, continue_when, metrics" in structure_prompt
    assert "skill-informed stage" in structure_prompt
    assert "done_when" not in structure_prompt.split("Boundaries:", 1)[0]
    assert "Metric provenance" in criteria_prompt
    assert "target-relative" in criteria_prompt
    assert "not phase-start planar_distance" in criteria_prompt
    assert "roughly aligned" in criteria_prompt
    assert "exact center" in criteria_prompt
    assert "Do not output or modify intent, task_goal, total, goal" in criteria_prompt


def test_model_parser_builds_skill_free_structure_messages():
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

    messages = parser._build_structure_messages("Progress past the visual landmark from the right side")
    expected_rules = Path("memory/FLYAGENT.md").read_text(encoding="utf-8-sig").strip()

    assert [message["role"] for message in messages] == ["system", "system", "user"]
    assert messages[0]["content"] == expected_rules
    assert messages[1]["content"] == parser.structure_prompt
    assert messages[2]["content"] == "Progress past the visual landmark from the right side"
    assert "你是 FlyAgent" in messages[0]["content"]
    assert "semantic task-structure parser" in messages[1]["content"]
    assert "skill_text" not in messages[1]["content"]
    assert "你是 FlyAgent" not in messages[1]["content"]


def test_model_parser_custom_contracts_keep_default_flyagent_rules(tmp_path):
    structure_prompt = tmp_path / "custom_structure.txt"
    criteria_prompt = tmp_path / "custom_criteria.txt"
    structure_prompt.write_text("Return custom immutable structure JSON.", encoding="utf-8")
    criteria_prompt.write_text("Return custom criteria patch JSON.", encoding="utf-8")
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(
            task_parser_mode="model",
            provider="openai_compatible",
            model_name="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
            task_parser_structure_prompt_path=str(structure_prompt),
            task_parser_criteria_prompt_path=str(criteria_prompt),
        ),
    )

    structure_messages = parser._build_structure_messages("test instruction")
    structure = SemanticPlanStructure.from_dict(
        {
            "intent": "face_target",
            "task_goal": {
                "target": "person",
                "target_attributes": {},
                "final_state": "facing_person",
            },
            "phases": [
                {"index": 1, "total": 1, "name": "align_to_person", "goal": "Face the person"}
            ],
        }
    )
    criteria_messages = parser._build_phase_criteria_messages(
        "test instruction",
        structure,
        structure.phases[0],
        [{"name": "visual", "description": "visual guidance", "skill_text": "rough alignment"}],
    )

    expected_rules = Path("memory/FLYAGENT.md").read_text(encoding="utf-8-sig").strip()
    assert structure_messages[0]["content"] == expected_rules
    assert structure_messages[1]["content"] == "Return custom immutable structure JSON."
    assert criteria_messages[0]["content"] == expected_rules
    assert criteria_messages[-2]["content"] == "Return custom criteria patch JSON."
    assert "rough alignment" in criteria_messages[1]["content"]


def test_model_parser_legacy_prompt_path_maps_to_criteria_only(tmp_path):
    legacy_prompt = tmp_path / "legacy_criteria.txt"
    legacy_prompt.write_text("Legacy criteria contract.", encoding="utf-8")
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(
            task_parser_mode="model",
            model_name="test-model",
            api_key="test-key",
            task_parser_prompt_path=str(legacy_prompt),
        ),
    )

    assert parser.criteria_prompt == "Legacy criteria contract."
    assert "semantic task-structure parser" in parser.structure_prompt
    assert parser.config_warnings == ["legacy_task_parser_prompt_path_used_for_criteria"]


def test_new_criteria_prompt_path_wins_over_legacy_path(tmp_path):
    new_prompt = tmp_path / "new.txt"
    legacy_prompt = tmp_path / "legacy.txt"
    new_prompt.write_text("New criteria contract.", encoding="utf-8")
    legacy_prompt.write_text("Legacy criteria contract.", encoding="utf-8")
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(
            task_parser_mode="model",
            model_name="test-model",
            api_key="test-key",
            task_parser_prompt_path=str(legacy_prompt),
            task_parser_criteria_prompt_path=str(new_prompt),
        ),
    )

    assert parser.criteria_prompt == "New criteria contract."
    assert parser.config_warnings == ["legacy_task_parser_prompt_path_ignored"]


def test_model_parser_runs_structure_then_independent_phase_criteria(monkeypatch):
    class RecordingResolver:
        def __init__(self):
            self.calls = []

        def resolve(self, instruction, semantic_goal, phase):
            self.calls.append((instruction, semantic_goal, dict(phase)))
            return SimpleNamespace(
                selected_skills=[
                    {
                        "name": f"skill_for_{phase['name']}",
                        "description": "phase knowledge",
                        "skill_text": "Use sufficient observable completion.",
                    }
                ],
                trace=lambda: {
                    "selection_status": "selected",
                    "selected_skills": [f"skill_for_{phase['name']}"],
                    "warnings": [],
                },
            )

    resolver = RecordingResolver()
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(
            task_parser_mode="model",
            model_name="test-model",
            api_key="test-key",
        ),
        phase_skill_resolver=resolver,
    )
    responses = iter(
        [
            '{"intent":"face_target","task_goal":{"target":"person","target_attributes":{},'
            '"final_state":"facing_person"},"phases":['
            '{"index":1,"total":2,"name":"locate_person","goal":"Locate the person"},'
            '{"index":2,"total":2,"name":"align_to_person","goal":"Face the person"}]}',
            '{"index":1,"name":"locate_person","done_when":["person is identifiable"],'
            '"observe":["latest_observation"],"continue_when":["person is absent or ambiguous"],'
            '"metrics":[]}',
            '{"index":2,"name":"align_to_person","done_when":['
            '"person is clearly visible and roughly aligned"],"observe":["latest_observation"],'
            '"continue_when":["person is clearly outside the forward view"],"metrics":[]}',
        ]
    )
    requests = []

    def fake_request(messages):
        requests.append(messages)
        return next(responses)

    monkeypatch.setattr(parser, "_request_model_content", fake_request)

    result = parser.parse("Face the person")

    assert [phase.name for phase in result.plan.phases] == ["locate_person", "align_to_person"]
    assert result.plan.phases[1].goal == "Face the person"
    assert result.plan.phases[1].done_when == ["person is clearly visible and roughly aligned"]
    assert len(resolver.calls) == 2
    assert [call[2]["name"] for call in resolver.calls] == ["locate_person", "align_to_person"]
    assert len(requests) == 3
    assert [message["role"] for message in requests[0]] == ["system", "system", "user"]
    assert "skill_for_" not in str(requests[0])
    assert "skill_for_locate_person" in str(requests[1])
    assert "skill_for_align_to_person" in str(requests[2])
    assert [item["selected_skills"] for item in parser.last_parse_trace["phases"]] == [
        ["skill_for_locate_person"],
        ["skill_for_align_to_person"],
    ]


def test_model_parser_continues_without_skill_when_phase_selector_fails(monkeypatch):
    class FailingResolver:
        def resolve(self, instruction, semantic_goal, phase):
            raise RuntimeError("selector unavailable")

    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(task_parser_mode="model", model_name="test-model", api_key="test-key"),
        phase_skill_resolver=FailingResolver(),
    )
    responses = iter(
        [
            '{"intent":"face_target","task_goal":{"target":"person","target_attributes":{},'
            '"final_state":"facing_person"},"phases":['
            '{"index":1,"total":1,"name":"align_to_person","goal":"Face the person"}]}',
            '{"index":1,"name":"align_to_person","done_when":["person is roughly aligned"],'
            '"observe":["latest_observation"],"continue_when":["person is not aligned"],'
            '"metrics":[]}',
        ]
    )
    requests = []

    def fake_request(messages):
        requests.append(messages)
        return next(responses)

    monkeypatch.setattr(parser, "_request_model_content", fake_request)

    result = parser.parse("Face the person")

    assert result.plan.phases[0].done_when == ["person is roughly aligned"]
    assert [message["role"] for message in requests[1]] == ["system", "system", "user"]
    assert parser.last_parse_trace["phases"][0]["selection_status"] == "resolver_error"


def test_model_parser_repairs_invalid_phase_criteria_once(monkeypatch):
    resolver = SimpleNamespace(
        resolve=lambda instruction, semantic_goal, phase: SimpleNamespace(
            selected_skills=[],
            trace=lambda: {"selection_status": "none_selected", "selected_skills": [], "warnings": []},
        )
    )
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(task_parser_mode="model", model_name="test-model", api_key="test-key"),
        phase_skill_resolver=resolver,
    )
    responses = iter(
        [
            '{"intent":"face_target","task_goal":{"target":"person","target_attributes":{},'
            '"final_state":"facing_person"},"phases":['
            '{"index":1,"total":1,"name":"align_to_person","goal":"Face the person"}]}',
            '{"index":1,"name":"align_to_person","goal":"illegally rewritten",'
            '"done_when":["person is centered"],"observe":[],"continue_when":[],"metrics":[]}',
            '{"index":1,"name":"align_to_person","done_when":["person is roughly aligned"],'
            '"observe":["latest_observation"],"continue_when":["alignment is insufficient"],'
            '"metrics":[]}',
        ]
    )
    requests = []

    def fake_request(messages):
        requests.append(messages)
        return next(responses)

    monkeypatch.setattr(parser, "_request_model_content", fake_request)

    result = parser.parse("Face the person")

    assert result.plan.phases[0].goal == "Face the person"
    assert len(requests) == 3
    assert "violated the phase criteria" in requests[-1][-1]["content"]


def test_phase_criteria_cannot_rewrite_structure_and_merge_preserves_goal():
    structure = SemanticPlanStructure.from_dict(
        {
            "intent": "face_target",
            "task_goal": {
                "target": "person",
                "target_attributes": {},
                "final_state": "facing_person",
            },
            "phases": [
                {"index": 1, "total": 1, "name": "align_to_person", "goal": "Face the person"}
            ],
        }
    )
    criteria = {
        "index": 1,
        "name": "align_to_person",
        "done_when": ["person is visible and roughly aligned"],
        "observe": ["latest_observation"],
        "continue_when": ["person is absent or clearly misaligned"],
        "metrics": [],
    }

    with pytest.raises(TaskParserContractError, match="unsupported fields"):
        validate_phase_criteria({**criteria, "goal": "Rewrite the goal"}, structure.phases[0])
    with pytest.raises(TaskParserContractError, match="does not match"):
        validate_phase_criteria({**criteria, "name": "different_phase"}, structure.phases[0])

    plan = merge_semantic_plan(structure, [criteria])
    assert plan.phases[0].goal == "Face the person"


def test_phase_criteria_numeric_provenance_rejects_invented_thresholds_and_target_distance_metric():
    phase = {"index": 1, "total": 1, "name": "hold_distance", "goal": "Hold target distance"}
    base = {
        "index": 1,
        "name": "hold_distance",
        "done_when": ["distance to the person is about 3.5 m"],
        "observe": ["latest_observation"],
        "continue_when": ["requested distance is not yet supported"],
        "metrics": [],
    }
    task_goal = {
        "target": "person",
        "target_attributes": {"distance_m": 3.5},
        "final_state": "holding_distance",
    }

    with pytest.raises(TaskParserContractError, match="numbers absent"):
        validate_phase_criteria(
            {**base, "continue_when": ["distance is outside 2.5 to 4.5 m"]},
            phase,
            instruction="Move to a position 3.5 meters from the person",
            task_goal=task_goal,
        )
    with pytest.raises(TaskParserContractError, match="target-relative"):
        validate_phase_criteria(
            {
                **base,
                "metrics": [
                    {
                        "name": "planar_distance",
                        "target": 3.5,
                        "unit": "m",
                        "tolerance": 0.5,
                        "correction_allowed": True,
                    }
                ],
            },
            phase,
            instruction="Move to a position 3.5 meters from the person",
            task_goal=task_goal,
        )


def test_phase_criteria_numeric_provenance_allows_metric_tolerance_and_bearing_decomposition():
    phase = {"index": 1, "total": 1, "name": "diagonal_move", "goal": "Move on a bearing"}
    criteria = {
        "index": 1,
        "name": "diagonal_move",
        "done_when": ["the requested 5 meter movement at 30 degrees is complete"],
        "observe": ["phase_execution_summary"],
        "continue_when": ["the requested movement remains incomplete"],
        "metrics": [
            {
                "name": "forward_distance",
                "target": 4.330127,
                "unit": "m",
                "tolerance": 1.0,
                "correction_allowed": True,
            },
            {
                "name": "right_distance",
                "target": 2.5,
                "unit": "m",
                "tolerance": 1.0,
                "correction_allowed": True,
            },
        ],
    }

    result = validate_phase_criteria(
        criteria,
        phase,
        instruction="Move 5 meters at 30 degrees to the right",
        task_goal={"target": "requested endpoint", "target_attributes": {}, "final_state": "moved"},
    )

    assert len(result["metrics"]) == 2


def test_phase_criteria_repairs_unrequested_visual_centering_precision():
    phase = {"index": 1, "total": 1, "name": "orient_to_target", "goal": "Face the target"}
    criteria = {
        "index": 1,
        "name": "orient_to_target",
        "done_when": ["the target is clearly visible and centered in the camera frame"],
        "observe": ["latest_observation"],
        "continue_when": ["the target is not visible or clearly misaligned"],
        "metrics": [],
    }

    with pytest.raises(TaskParserContractError, match="centering precision"):
        validate_phase_criteria(criteria, phase, instruction="Turn toward the target", task_goal={})

    repaired = validate_phase_criteria(
        {**criteria, "done_when": ["the target is clearly visible and roughly centered"]},
        phase,
        instruction="Turn toward the target",
        task_goal={},
    )
    explicit = validate_phase_criteria(
        criteria,
        phase,
        instruction="Center the target in the camera frame",
        task_goal={},
    )

    assert "roughly centered" in repaired["done_when"][0]
    assert "centered" in explicit["done_when"][0]


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
            }
        )
