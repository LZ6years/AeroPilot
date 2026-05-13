"""Test model-ready observation context building."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.schemas import SemanticPhase
from src.agent.task_parser import TaskParser
from src.perception.context_builder import ObservationContextBuilder


def make_state() -> MissionState:
    result = TaskParser(mode="rule_based").parse("左转寻找建筑入口并停在入口前")
    result.plan.phases[0] = SemanticPhase(
        index=1,
        total=len(result.plan.phases),
        name="orient_leftward",
        meaning="向左调整视角，寻找入口可能区域",
    )
    return MissionState("run_test", result.instruction, result.plan)


def test_context_builder_includes_mission_phase_observation_actions_and_safety():
    state = make_state()
    state.record_action("uav.hover", {}, {"ok": True})
    state.record_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["rgb", "depth", "pose"],
            "pose": {"x": 0.0},
            "frames": {"rgb": {"mock": True}},
            "timestamp": "t0",
            "source": "phase_step",
        }
    )

    context = ObservationContextBuilder().build(state)

    assert context["active_instruction"] == state.user_instruction
    assert context["semantic_goal"] == state.semantic_plan.goal.to_dict()
    assert context["current_phase"]["name"] == "orient_leftward"
    assert context["current_phase"]["index"] == 1
    assert context["safety_constraints"] == {"hover_required": False}
    assert "uav.connect" in context["allowed_action_constraints"]["lifecycle_tools_excluded"]
    assert "uav.hover" in context["allowed_action_constraints"]["planner_tools"]
    fragments = {fragment["name"]: fragment for fragment in context["fragments"]}
    assert fragments["image"]["content"]["recent_observation_sources"][-1] == "phase_step"
    assert fragments["action"]["content"]["recent_actions"][-1]["tool"] == "uav.hover"


def test_context_builder_handles_missing_observation():
    state = make_state()

    context = ObservationContextBuilder().build(state)

    fragments = {fragment["name"]: fragment for fragment in context["fragments"]}
    assert fragments["image"]["warnings"] == ["no_recent_image_frames"]
    assert fragments["action"]["content"]["recent_actions"] == []
    assert context["current_phase"]["name"] == "orient_leftward"
