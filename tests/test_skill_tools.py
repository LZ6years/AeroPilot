"""Tests for JSON-discovered callable skill capabilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.agent.planning.tool_contract_validator import (
    ToolContractValidationError,
    ToolContractValidator,
)
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.plugins.contracts.skill import SkillCapabilityResult
from src.plugins.skills import dispatcher as dispatcher_module
from src.plugins.skills.dispatcher import SkillCapabilityDispatcher
from src.plugins.skills.markdown import SkillMarkdownError, parse_skill_markdown
from src.tools.registry import default_registry


def _capability_registry() -> PluginRegistry:
    return PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/capabilities"],
                "enabled": ["capabilities.visual_grounding"],
            }
        }
    )


def _context_for_skill_tool() -> dict:
    constraints = ObservationContextBuilder(
        plugin_registry=_capability_registry()
    )._allowed_action_constraints()
    return {
        "current_phase": {"name": "confirm_visual_reference"},
        "allowed_action_constraints": constraints,
    }


def test_callable_skill_is_discovered_from_json_and_exposes_schema():
    registry = _capability_registry()
    capability = registry.skill_capability_catalog.capabilities[0]
    listings = registry.skill_capability_catalog.listings()

    assert [item["call_name"] for item in listings] == [
        "skill.visual_grounding.locate"
    ]
    assert listings[0]["plugin_id"] == "capabilities.visual_grounding"
    assert capability.settings["grounding_backend"] == "grounding_dino"
    assert listings[0]["skill_name"] == "visual_grounding_locate"
    assert listings[0]["wire_name"] == "skill_visual_grounding_locate"
    assert listings[0]["input_schema"]["required"] == [
        "observation_id",
        "query",
        "include_distance",
    ]
    assert listings[0]["runtime"] == {
        "effect": "read_only",
        "result_timing": "immediate",
        "batching": "parallel_safe",
        "model_exposed": True,
        "max_calls_per_observation": 2,
    }
    assert "settings" not in listings[0]
    assert "grounding_model" not in str(listings[0])


def test_provider_tool_schema_does_not_expose_settings_or_policy_extensions():
    tool = default_registry(_capability_registry()).get(
        "skill.visual_grounding.locate"
    ).to_provider_tool()
    parameters = tool["function"]["parameters"]

    assert not any(key.startswith("x-flyagent-") for key in parameters)
    assert "settings" not in parameters
    assert "IDEA-Research" not in str(tool)


def test_context_constraints_merge_builtin_tools_and_skill_calls():
    constraints = _context_for_skill_tool()["allowed_action_constraints"]

    assert constraints["planner_tools"][:3] == [
        "uav.hover",
        "uav.turn_relative",
        "uav.move_body",
    ]
    assert "skill.visual_grounding.locate" in constraints["planner_tools"]
    assert (
        constraints["action_schemas"]["skill.visual_grounding.locate"]["source"]
        == "skill_capability"
    )


def test_skill_tool_validator_accepts_schema_valid_args():
    plan = ShortHorizonPlan(
        "confirm_visual_reference",
        [
            PlannedStep.tool_step(
                "skill.visual_grounding.locate",
                {
                    "observation_id": "obs_0001",
                    "query": "the person to orbit around",
                    "include_distance": False,
                },
                "confirm target visibility",
            )
        ],
    )

    validated = ToolContractValidator().validate(
        plan, "confirm_visual_reference", _context_for_skill_tool()
    )

    assert validated.steps[0].tool == "skill.visual_grounding.locate"


def test_skill_tool_validator_rejects_schema_invalid_args():
    plan = ShortHorizonPlan(
        "confirm_visual_reference",
        [
            PlannedStep.tool_step(
                "skill.visual_grounding.locate",
                {
                    "observation_id": "obs_0001",
                    "query": "person",
                    "include_distance": False,
                    "extra": True,
                },
                "bad args",
            )
        ],
    )

    with pytest.raises(ToolContractValidationError):
        ToolContractValidator().validate(
            plan, "confirm_visual_reference", _context_for_skill_tool()
        )


def test_dispatcher_passes_private_settings_only_to_handler(monkeypatch):
    dispatcher = SkillCapabilityDispatcher(_capability_registry())
    captured: dict = {}

    def handle(payload: dict) -> SkillCapabilityResult:
        captured.update(payload)
        return SkillCapabilityResult(message="ok", data={"candidates": []})

    monkeypatch.setattr(
        dispatcher_module,
        "_load_handler_module",
        lambda handler_path: type("HandlerModule", (), {"handle": staticmethod(handle)})(),
    )
    result = dispatcher.call(
        "scene_1",
        "uav_1",
        "skill.visual_grounding.locate",
        {
            "observation_id": "obs_0001",
            "query": "路口右侧的人",
            "include_distance": False,
        },
        runtime_context={"latest_observation": {"observation_id": "obs_0001"}},
    )

    assert result.ok is True
    assert captured["settings"]["grounding_model"].endswith("grounding-dino-base")
    assert "settings" not in captured["capability"]
    captured["settings"]["box_threshold"] = 99
    assert (
        _capability_registry().skill_capability_catalog.capabilities[0].settings[
            "box_threshold"
        ]
        == 0.4
    )


def test_capability_settings_must_be_a_json_object(tmp_path):
    skill_dir = tmp_path / "skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill.\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    (scripts_dir / "handler.py").write_text(
        "def handle(payload):\n    return {}\n", encoding="utf-8"
    )
    (skill_dir / "capability.json").write_text(
        """{
          "call_name": "skill.demo.call",
          "wire_name": "skill_demo_call",
          "description": "Demo.",
          "input_schema": {"type": "object"},
          "runtime": {
            "effect": "read_only",
            "result_timing": "immediate",
            "batching": "single",
            "model_exposed": true
          },
          "settings": ["not", "an", "object"],
          "handler": "scripts/handler.py"
        }""",
        encoding="utf-8",
    )

    with pytest.raises(SkillMarkdownError, match="settings must be a JSON object"):
        parse_skill_markdown(skill_dir / "SKILL.md", "capabilities.demo")


def test_core_registration_and_runtime_do_not_hardcode_visual_grounding_id():
    core_files = [
        Path("src/plugins/core/discovery.py"),
        Path("src/plugins/core/registry.py"),
        Path("src/plugins/skills/dispatcher.py"),
        Path("src/runtime/runners/task_runtime.py"),
        Path("src/tools/registry.py"),
    ]

    for path in core_files:
        text = path.read_text(encoding="utf-8")
        assert "capabilities.visual_grounding" not in text, path
        assert "skill.visual_grounding.locate" not in text, path
