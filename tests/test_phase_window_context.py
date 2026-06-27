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
                observe=["latest_observation", "phase_execution_summary"],
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
    support = result.phase_progress_evidence["done_when_support"]

    assert progress["yaw_only_so_far"] is True
    assert progress["has_translation"] is False
    assert any(item["requires_translation"] and item["supported_by_motion"] is False for item in support)
