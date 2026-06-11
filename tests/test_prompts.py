"""Test prompt files used by model-backed planning."""

from __future__ import annotations

from pathlib import Path


def test_task_parser_prompt_contains_phase_position_and_safety_boundaries():
    prompt = (Path("src") / "agent" / "semantic" / "prompts" / "task_parser_system.txt").read_text(encoding="utf-8")

    assert '"index"' in prompt
    assert '"total"' in prompt
    assert "not a fixed program dispatch key" in prompt
    assert "uav.turn_relative" in prompt
    assert "env.observe" in prompt
    assert "uav.<tool_name>" in prompt
    assert "向上飞行" in prompt
    assert "altitude change" in prompt
    assert "distance_m" in prompt
    assert "completion_criteria" in prompt
    assert "correction_allowed" in prompt
    assert "hover_after_movement" in prompt
    assert "Do not turn simple movement-only tasks into visual landmark search tasks" in prompt
    assert "Seeing or facing the target" in prompt
    assert "visual confirmation check" in prompt


def test_action_planner_prompt_contains_allowed_tools_contract():
    prompt = (Path("src") / "agent" / "planning" / "prompts" / "action_planner_system.txt").read_text(encoding="utf-8")

    assert "uav.hover" in prompt
    assert "uav.turn_relative" in prompt
    assert "uav.move_body" in prompt
    assert "forbidden_tools" not in prompt
    assert "observation_policy" not in prompt
    assert "phase_transition" in prompt
    assert "current_phase.completion_criteria" in prompt
    assert "action_window" in prompt
    assert "current_phase_only" in prompt
    assert "requires_intermediate_observation" in prompt
    assert "Action Window" in prompt
    assert "agent_messages" in prompt
    assert "user_preferences" in prompt
    assert "raw thinking" in prompt
    assert "Do not output phase_transition" in prompt
    assert "Do not output kind=\"observe\"" in prompt
    assert "action_schemas" in prompt
    assert "Do not invent alternate argument names" in prompt
    assert "nominal_action_scale" in prompt
    assert "duration = distance_m / nominal_action_scale.linear_velocity_mps" in prompt
    assert '{"direction": "forward", "duration": 2.5}' in prompt
    assert "Always include duration for uav.move_body" in prompt
    assert "Do not mechanically reuse 0.5 or 1.0 seconds" in prompt
    assert "model_inputs" in prompt
    assert "actual image input is attached" in prompt
    assert "target is left of image center" in prompt
    assert "do not chase pixel-perfect centering" in prompt
    assert "do not use runtime safety budgets as a search strategy" in prompt
    assert "safety_constraints yaw scan budget" not in prompt
    assert "outside the required JSON shape" in prompt
    assert "selected search skill guidance" in prompt
    assert "short-step profile" not in prompt
    assert "recommended_next_observation" not in prompt


def test_model_repair_prompt_requires_contract_valid_tool_only_plan():
    prompt = (Path("src") / "agent" / "planning" / "prompts" / "model_repair_system.txt").read_text(encoding="utf-8")

    assert "Return exactly one corrected ShortHorizonPlan" in prompt
    assert "Use only kind=\"tool\" steps" in prompt
    assert "Never output kind=\"observe\"" in prompt
    assert "Do not repeat the previous invalid plan" in prompt
    assert "Repair only tool-contract issues" in prompt
    assert "action_window" in prompt
    assert "agent_messages" in prompt
    assert "raw thinking" in prompt
    assert "current_phase_only" in prompt
    assert "requires_intermediate_observation" in prompt
    assert "phase_transition" not in prompt


def test_phase_transition_judge_prompt_requires_agent_messages():
    prompt = (Path("src") / "agent" / "planning" / "prompts" / "phase_transition_judge_system.txt").read_text(encoding="utf-8")

    assert "agent_messages" in prompt
    assert "final" in prompt
    assert "raw thinking" in prompt
    assert "Do not say a visual target was found" in prompt
    assert "actual image input is attached" in prompt
    assert "executed_steps[].result" in prompt
    assert "unconfirmed model-judged criteria" in prompt
    assert "do not require pixel-perfect centering" in prompt
    assert "alternating small yaw corrections" in prompt
    assert "short-step benchmark profiles" not in prompt
    assert "recommended_next_observation" not in prompt
