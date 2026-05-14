"""Test prompt files used by model-backed planning."""

from __future__ import annotations

from pathlib import Path


def test_task_parser_prompt_contains_phase_position_and_safety_boundaries():
    prompt = (
        Path("src") / "agent" / "semantic" / "prompts" / "task_parser_system.txt"
    ).read_text(encoding="utf-8")

    assert '"index"' in prompt
    assert '"total"' in prompt
    assert "not fixed program dispatch keys" in prompt
    assert "uav.turn_relative" in prompt
    assert "env.observe" in prompt
    assert "向上飞行" in prompt
    assert "altitude change" in prompt


def test_action_planner_prompt_contains_allowed_and_forbidden_tools():
    prompt = (
        Path("src") / "agent" / "planning" / "prompts" / "action_planner_system.txt"
    ).read_text(encoding="utf-8")

    assert "uav.hover" in prompt
    assert "uav.turn_relative" in prompt
    assert "uav.move_body" in prompt
    assert "forbidden_tools" in prompt
    assert "observation_policy" in prompt
    assert "Do not output kind=\"observe\"" in prompt
    assert "action_schemas" in prompt
    assert "Do not invent alternate argument names" in prompt
