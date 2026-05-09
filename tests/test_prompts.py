"""Test prompt files used by model-backed planning."""

from __future__ import annotations

from pathlib import Path


def test_task_parser_prompt_contains_phase_position_and_safety_boundaries():
    prompt = (
        Path("src")
        / "agent"
        / "prompts"
        / "task_parser_system.txt"
    ).read_text(encoding="utf-8")

    assert '"index"' in prompt
    assert '"total"' in prompt
    assert "not fixed program dispatch keys" in prompt
    assert "uav.turn_relative" in prompt
    assert "env.observe" in prompt
