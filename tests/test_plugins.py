"""Tests for UAV-Agent plugin discovery and skill loading."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.semantic.task_parser import TaskParser
from src.memory.context.modules.skills import DefaultSkillsContextModule
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.flight_memory import FlightMemory
from src.plugins.core.discovery import PluginDiscovery
from src.plugins.core.registry import PluginRegistry
from src.plugins.skills import SkillContentLoader
from src.plugins.skills.selection import ModelSkillSelector, SkillSelectionResult
from src.runtime.configs.model_config import ModelConfig


EXPECTED_UAV_FLIGHT_SKILLS = {
    "visual_target_task_reasoning",
    "search_under_uncertainty",
    "approach_and_standoff",
    "obstacle_and_safety_awareness",
    "task_completion_evidence",
}

REMOVED_UAV_FLIGHT_SKILLS = {
    "visual_target_alignment",
    "search_and_confirm_target",
    "standoff_distance",
    "approach_landmark",
    "motion_progress_assessment",
}


def make_stage_context(result) -> StageContextInput:
    return StageContextInput(
        active_instruction=result.instruction,
        semantic_plan=result.plan.to_dict(),
        current_phase=result.plan.phases[0].to_dict(),
        semantic_goal=result.plan.goal.to_dict(),
    )


def _uav_flight_registry() -> PluginRegistry:
    return PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/knowledge"],
                "enabled": ["knowledge.uav_flight"],
            }
        }
    )


def test_plugin_discovery_finds_only_current_uav_flight_skills():
    result = PluginDiscovery(
        roots=["plugins/knowledge"],
        enabled=["knowledge.uav_flight"],
    ).discover()

    names = {skill.name for skill in result.contributions.skills}

    assert names == EXPECTED_UAV_FLIGHT_SKILLS
    assert not REMOVED_UAV_FLIGHT_SKILLS & names
    assert {skill.plugin_id for skill in result.contributions.skills} == {"knowledge.uav_flight"}
    assert result.warnings == []


def test_skill_catalog_validates_selected_names_and_loader_reads_skill_resources():
    catalog = _uav_flight_registry().skill_catalog

    accepted, rejected = catalog.validate_selection(["visual_target_task_reasoning", "missing_skill"])

    assert [skill.name for skill in accepted] == ["visual_target_task_reasoning"]
    assert rejected == ["missing_skill"]
    summary = SkillContentLoader().load_summary(accepted[0])
    assert summary["name"] == "visual_target_task_reasoning"
    assert "Visual Target Task Reasoning" in summary["skill_text"]
    assert "Visual-target tasks usually need three checks" in summary["skill_text"]
    assert summary["references"] == []
    assert summary["available_references"] == []
    assert summary["assets"]["scripts"] == []


def test_skill_loader_lists_references_without_executing_scripts():
    catalog = _uav_flight_registry().skill_catalog

    accepted, _ = catalog.validate_selection(["search_under_uncertainty", "obstacle_and_safety_awareness"])
    summaries = {skill.name: SkillContentLoader().load_summary(skill) for skill in accepted}

    search_refs = summaries["search_under_uncertainty"]["available_references"]
    safety_refs = summaries["obstacle_and_safety_awareness"]["available_references"]
    assert search_refs == ["references\\search_patterns.md"] or search_refs == ["references/search_patterns.md"]
    assert safety_refs == ["references\\safety_checks.md"] or safety_refs == ["references/safety_checks.md"]
    assert summaries["search_under_uncertainty"]["assets"]["scripts"] == []
    assert summaries["obstacle_and_safety_awareness"]["assets"]["scripts"] == []


def test_skill_catalog_accepts_unique_qualified_skill_names():
    catalog = _uav_flight_registry().skill_catalog

    accepted, rejected = catalog.validate_selection(
        [
            "knowledge.uav_flight.visual_target_task_reasoning",
            "visual_target_task_reasoning",
            "missing.plugin.skill",
        ]
    )

    assert [skill.name for skill in accepted] == ["visual_target_task_reasoning"]
    assert rejected == ["missing.plugin.skill"]


def test_skill_selection_fixture_references_only_current_skill_names():
    fixture_path = Path("tests/fixtures/uav_knowledge_skill_selection_cases.json")
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert len(cases) >= 20
    for case in cases:
        assert case["instruction"].strip()
        assert set(case.get("expected_any", [])) <= EXPECTED_UAV_FLIGHT_SKILLS
        assert set(case.get("forbidden", [])) <= EXPECTED_UAV_FLIGHT_SKILLS
        assert not set(case.get("expected_any", [])) & set(case.get("forbidden", []))
        assert int(case.get("max_selected", 0)) >= 0


def test_skills_module_handles_no_model_selection():
    result = TaskParser(mode="rule_based").parse("观察右侧区域，不确定时问我。")
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", active_instruction=result.instruction)
    module = DefaultSkillsContextModule(
        plugin_registry=_uav_flight_registry(),
        skill_selector=ModelSkillSelector(enabled=False),
    )

    fragment = module.build(MemorySourceBundle(memory), make_stage_context(result)).to_dict()

    assert fragment["name"] == "skills"
    assert fragment["content"]["selected_skills"] == []
    assert "available_skills" not in fragment["content"]
    assert fragment["metadata"]["available_skill_names"]
    assert fragment["metadata"]["selection"]["status"] == "disabled"
    assert fragment["facts"] == []


def test_model_skill_selector_retries_invalid_json_once():
    class RetrySelector(ModelSkillSelector):
        def __init__(self):
            super().__init__(
                ModelConfig(model_name="test-model", api_key="secret", base_url="https://example.test/v1")
            )
            self.calls = 0

        def _request_content(self, selection_context):
            self.calls += 1
            if self.calls == 1:
                return '{"selected_skills": ["visual_target_task_reasoning"'
            return '{"selected_skills": ["knowledge.uav_flight.visual_target_task_reasoning"]}'

    registry = _uav_flight_registry()

    result = RetrySelector().select({"available_skills": registry.skill_catalog.listings()}, registry.skill_catalog)

    assert result.status == "selected"
    assert result.accepted_skill_names == ["visual_target_task_reasoning"]
    assert result.rejected_skill_names == []


def test_model_skill_selector_falls_back_to_empty_selection_after_invalid_json_retry():
    class InvalidSelector(ModelSkillSelector):
        def __init__(self):
            super().__init__(
                ModelConfig(model_name="test-model", api_key="secret", base_url="https://example.test/v1")
            )

        def _request_content(self, selection_context):
            return '{"selected_skills": ["unknown_skill"'

    registry = _uav_flight_registry()

    result = InvalidSelector().select({"available_skills": registry.skill_catalog.listings()}, registry.skill_catalog)

    assert result.status == "none_selected"
    assert result.accepted_skill_names == []
    assert any(warning.startswith("skill_selection_invalid_json_fallback:") for warning in result.warnings)


def test_model_skill_selector_recovers_skill_names_from_invalid_json_text():
    class InvalidSelector(ModelSkillSelector):
        def __init__(self):
            super().__init__(
                ModelConfig(model_name="test-model", api_key="secret", base_url="https://example.test/v1")
            )

        def _request_content(self, selection_context):
            return "selected_skills should include visual_target_task_reasoning and search_under_uncertainty"

    registry = _uav_flight_registry()

    result = InvalidSelector().select({"available_skills": registry.skill_catalog.listings()}, registry.skill_catalog)

    assert result.status == "selected"
    assert set(result.accepted_skill_names) == {"visual_target_task_reasoning", "search_under_uncertainty"}
    assert any(warning.startswith("skill_selection_invalid_json_recovered:") for warning in result.warnings)


def test_skills_module_loads_model_selected_skill():
    class FixedSelector:
        def select(self, selection_context, catalog):
            return SkillSelectionResult(
                requested_skill_names=["visual_target_task_reasoning", "missing_skill"],
                accepted_skill_names=["visual_target_task_reasoning"],
                rejected_skill_names=["missing_skill"],
                status="selected",
                warnings=["skill_selection_rejected:missing_skill"],
            )

    result = TaskParser(mode="rule_based").parse("观察右侧区域，不确定时问我。")
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", active_instruction=result.instruction)
    module = DefaultSkillsContextModule(plugin_registry=_uav_flight_registry(), skill_selector=FixedSelector())

    fragment = module.build(MemorySourceBundle(memory), make_stage_context(result)).to_dict()

    assert fragment["metadata"]["selection"]["accepted_skill_names"] == ["visual_target_task_reasoning"]
    assert fragment["metadata"]["selection"]["rejected_skill_names"] == ["missing_skill"]
    assert fragment["content"]["selected_skills"][0]["name"] == "visual_target_task_reasoning"
    assert "skill_text" in fragment["content"]["selected_skills"][0]
    assert "plugin_id" not in fragment["content"]["selected_skills"][0]
    assert "source_path" not in fragment["content"]["selected_skills"][0]
    assert "available_references" not in fragment["content"]["selected_skills"][0]
    assert fragment["facts"] == []
    assert fragment["warnings"].count("skill_selection_rejected:missing_skill") >= 1
