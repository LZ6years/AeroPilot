"""Tests for runtime phase-window context construction."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.semantic.schemas import SemanticPhase, SemanticPlan, TaskGoal
from src.runtime.context import PhaseWindowContextBuilder


def make_state() -> MissionState:
    plan = SemanticPlan(
        intent="test_phase_window_context",
        task_goal=TaskGoal(target="target", final_state="done"),
        phases=[
            SemanticPhase(
                index=1,
                total=1,
                name="orient_to_target",
                goal="turn toward the target",
                done_when=[
                    "target is visible",
                    "UAV has moved meaningfully closer to target",
                ],
                continue_when=["target is not aligned"],
                metrics=[{"name": "yaw_delta", "target": 30.0, "unit": "deg", "tolerance": 5.0}],
            )
        ],
    )
    return MissionState("run-context", "turn toward target", plan)


def test_phase_window_context_adds_runtime_evidence_and_model_context():
    state = make_state()
    state.current_phase_progress = {
        "measurements": {"yaw_delta_deg": 31.0, "forward_m": 0.0, "right_m": 0.0, "up_m": 0.0, "planar_m": 0.0},
        "tool_facts": {
            "turn_yaw_success_sum_deg": 30.0,
            "move_duration_success_sum_sec": 0.0,
            "tools_executed": {"uav.turn_relative": 1},
        },
    }
    observation_before = {
        "source": "preflight",
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": -3.0},
            "velocity": {"x": 9.0, "y": 0.0, "z": 0.0},
        },
    }

    result = PhaseWindowContextBuilder().build(
        state,
        {"active_instruction": "turn toward target", "fragments": []},
        observation_before,
    )

    assert result.planner_payload["phase_metrics"] == result.phase_metrics
    assert result.planner_payload["phase_completion_evaluation"] == result.phase_completion_evaluation
    assert result.planner_payload["phase_progress_evidence"] == result.phase_progress_evidence
    assert result.planner_payload["latest_observation_summary"] == result.latest_observation_summary
    assert result.phase_completion_evaluation["all_within_tolerance"] is True
    assert result.latest_observation_summary["pose"]["position"]["x"] == 1.0
    assert "velocity" not in result.latest_observation_summary["pose"]
    assert "velocity" not in result.model_context["latest_observation_summary"]["pose"]
    assert "runtime_control" not in result.planner_payload
    assert "runtime_control" not in result.model_context
    assert "phase_progress_evidence" not in result.model_context
    assert "phase_execution_summary" not in result.model_context


def test_phase_window_context_adds_compact_runtime_control_when_available():
    state = make_state()

    result = PhaseWindowContextBuilder().build(
        state,
        {"active_instruction": "turn toward target", "fragments": []},
        {"pose": {}},
        runtime_control_context={
            "available": ["semantic_plan_update"],
            "contracts": {
                "semantic_plan_update": "Request after multiple cycles when semantic-contract confidence is low.",
                "ask_user": "hidden because ask_user is not available",
            },
            "status": {"semantic_plan_update": "pending", "ask_user": "hidden"},
            "recommendations": {
                "semantic_plan_update": "review_now: reassess the current semantic contract",
                "ask_user": "hidden because ask_user is not available",
            },
            "debug": {"job_id": "hidden"},
        },
    )

    assert result.planner_payload["runtime_control"] == {
        "available": ["semantic_plan_update"],
        "contracts": {
            "semantic_plan_update": "Request after multiple cycles when semantic-contract confidence is low.",
        },
        "status": {"semantic_plan_update": "pending"},
        "recommendations": {
            "semantic_plan_update": "review_now: reassess the current semantic contract",
        },
    }
    assert result.model_context["runtime_control"] == result.planner_payload["runtime_control"]
    assert list(result.planner_payload).index("runtime_control") < list(result.planner_payload).index("phase_metrics")
    assert "phase_progress_evidence" not in result.model_context
    assert "phase_execution_summary" not in result.model_context


def test_phase_window_context_marks_yaw_only_progress():
    state = make_state()
    state.current_phase_progress = {
        "measurements": {"yaw_delta_deg": 18.0, "forward_m": 0.0, "right_m": 0.0, "up_m": 0.0, "planar_m": 0.0},
        "tool_facts": {
            "turn_yaw_success_sum_deg": 18.0,
            "move_duration_success_sum_sec": 0.0,
            "tools_executed": {"uav.turn_relative": 2},
        },
    }

    result = PhaseWindowContextBuilder().build(state, {"fragments": []}, {"pose": {}})
    progress = result.phase_progress_evidence["motion_progress"]

    assert progress["measurement_scope"] == "phase_start_relative_uav_motion"
    assert "not distance-to-target" in progress["interpretation_note"]
    assert progress["yaw_only_so_far"] is True
    assert progress["has_translation"] is False
    assert "done_when_support" not in result.phase_progress_evidence
    assert "visual_progress" not in result.phase_progress_evidence
    assert "model_guidance" not in result.phase_progress_evidence


def test_phase_window_context_model_context_uses_neutral_no_metric_completion():
    plan = SemanticPlan(
        intent="test_no_metric_phase",
        task_goal=TaskGoal(target="target", final_state="done"),
        phases=[
            SemanticPhase(
                index=1,
                total=1,
                name="visual_phase",
                goal="judge a visual phase",
                done_when=["visual evidence is sufficient"],
                continue_when=["visual evidence is insufficient"],
                metrics=[],
            )
        ],
    )
    state = MissionState("run-context", "judge visual target", plan)

    base_context = {
        "current_phase": state.semantic_plan.phases[0].to_dict(),
        "fragments": [],
    }
    result = PhaseWindowContextBuilder().build(state, base_context, {"pose": {}})

    assert result.planner_payload["phase_completion_evaluation"]["all_within_tolerance"] is False
    assert result.planner_payload["current_phase"]["continue_when"] == ["visual evidence is insufficient"]
    assert result.model_context["current_phase"]["continue_when"] == ["visual evidence is insufficient"]
    assert "phase_completion_evaluation" not in result.model_context
