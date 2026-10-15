"""Tests for UAV-Agent plugin discovery and skill loading."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import re

import pytest

from src.memory.context.modules.skills import DefaultSkillsContextModule
from src.memory.context.contracts import MemorySourceBundle, TaskContextInput
from src.memory.flight_memory import FlightMemory
from src.plugins.core.discovery import PluginDiscovery
from src.plugins.core.registry import PluginRegistry
from src.plugins.skills import SkillContentLoader
from src.agent.config import ModelConfig


EXPECTED_UAV_FLIGHT_SKILLS = {
    "visual_target_task_reasoning",
    "approach_and_standoff",
    "basic_flight_control",
    "abnormal_situation_recovery",
    "route_traversal_and_passage",
    "orbit_and_surround",
}

REMOVED_UAV_FLIGHT_SKILLS = {
    "vertical_landing_and_altitude_control",
    "visual_target_alignment",
    "search_and_confirm_target",
    "search_under_uncertainty",
    "standoff_distance",
    "approach_landmark",
    "motion_progress_assessment",
    "obstacle_and_safety_awareness",
    "task_completion_evidence",
}

KNOWLEDGE_SKILL_SECTIONS = [
    "Application Boundary",
    "Evidence Model",
    "Decision Procedure",
    "Completion And Stop",
    "Recovery",
]


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
    assert all(skill.summary.strip() for skill in result.contributions.skills)
    assert result.warnings == []


def test_uav_knowledge_skills_follow_five_section_contract():
    catalog = _uav_flight_registry().skill_catalog

    for skill in catalog.skills:
        headings = [
            line.removeprefix("## ").strip()
            for line in skill.body.splitlines()
            if line.startswith("## ")
        ]
        assert headings == KNOWLEDGE_SKILL_SECTIONS, skill.name
        for internal_term in ("done_when", "continue_when", "SemanticPlan", "output schema", "list item"):
            assert internal_term not in skill.body, f"{skill.name}:{internal_term}"
        for index, heading in enumerate(KNOWLEDGE_SKILL_SECTIONS):
            marker = f"## {heading}"
            start = skill.body.index(marker) + len(marker)
            end = (
                skill.body.index(f"## {KNOWLEDGE_SKILL_SECTIONS[index + 1]}")
                if index + 1 < len(KNOWLEDGE_SKILL_SECTIONS)
                else len(skill.body)
            )
            assert skill.body[start:end].strip(), f"{skill.name}:{heading}"


def test_teaching_references_are_valid_optional_reads_without_benchmark_or_batch_guidance():
    from src.plugins.skills.knowledge_service import KnowledgeService
    catalog = _uav_flight_registry().skill_catalog
    service = KnowledgeService(catalog)
    assert service.loaded_index() == []
    for skill in catalog.skills:
        loaded = service.load(skill.name)
        assert loaded["body"] == skill.body.strip() and loaded["references"] == []
        paths = {item["path"] for item in loaded["reference_manifest"]}
        linked = set(re.findall(r"\]\((references/[^)]+)\)", skill.body))
        assert linked <= paths, skill.name
        detail = service.load(skill.name, sorted(paths))
        assert not detail["body_included"] and "body" not in detail
        assert {item["path"] for item in detail["references"]} == paths
        for text in [skill.body, skill.description, *(item["text"] for item in detail["references"])]:
            assert "Multiple actions are" not in text
            for private in ("AerialVLN", "CURRENT TASK STATE", "distance_to_goal", "near_boundary_stop"):
                assert private not in text, (skill.name, private)
    # Loading all chapters above is a test, not an automatic startup policy.
    assert KnowledgeService(catalog).loaded_index() == []


def test_skill_catalog_validates_selected_names_and_loader_reads_skill_resources():
    catalog = _uav_flight_registry().skill_catalog
    accepted, rejected = catalog.validate_selection(["visual_target_task_reasoning", "missing_skill"])
    assert [skill.name for skill in accepted] == ["visual_target_task_reasoning"]
    assert rejected == ["missing_skill"]
    skill = accepted[0]
    index = skill.index_listing()
    assert index["name"] == skill.name and index["summary"]
    assert "source_path" not in index and "skill_text" not in index
    assert {ref["path"] for ref in index["references"]} == {
        "references/search_under_uncertainty.md", "references/viewpoint_examples.md"}
    summary = SkillContentLoader().load_summary(skill)
    # All essential teaching sections fit the bounded legacy loader as well as
    # the complete live KnowledgeService. References remain explicit reads.
    assert summary["skill_text"] == skill.body.strip()
    assert summary["references"] == [] and summary["assets"]["scripts"] == []


def test_skill_loader_lists_references_without_executing_scripts():
    skill = _uav_flight_registry().skill_catalog.validate_selection(["visual_target_task_reasoning"])[0][0]
    loader = SkillContentLoader()
    guidance = loader.load_planner_guidance(skill)
    assert guidance["skill_text"] == skill.body.strip()
    for private in ("summary", "references", "source_path", "skill_dir", "available_references"):
        assert private not in guidance
    expanded = loader.load_planner_guidance(skill, requested_references=["references/search_under_uncertainty.md"])
    assert len(expanded["requested_references"]) == 1
    assert expanded["requested_references"][0]["path"] == "references/search_under_uncertainty.md"
    assert "view overlap" in expanded["requested_references"][0]["text"]


def test_search_under_uncertainty_keeps_broad_search_out_of_candidate_confirmation():
    root = Path("plugins/knowledge/uav_flight/visual_target_task_reasoning")
    core = (root / "SKILL.md").read_text(encoding="utf-8")
    reference = (root / "references/search_under_uncertainty.md").read_text(encoding="utf-8")
    assert "tentative candidate" in core and "Contradicted attributes" in core
    assert "view overlap" in reference and "Reorient when evidence changes" in reference
    assert "30-45" not in reference and "5-15" not in reference


def test_vertical_handling_and_landing_use_separate_cores_and_explicit_formula_reference():
    from src.plugins.skills.knowledge_service import KnowledgeService
    service = KnowledgeService(_uav_flight_registry().skill_catalog)
    handling = service.load("basic_flight_control")
    landing = service.load("approach_and_standoff")
    assert "Height" in handling["body"] and "angled" in handling["body"]
    assert "landing" in landing["body"] and "touchdown" in landing["body"]
    formula = "angle = atan(sqrt(vx^2 + vy^2) / abs(vz))"
    assert formula not in handling["body"] and handling["references"] == []
    detail = service.load("basic_flight_control", ["references/angled_vertical_motion.md"])
    assert not detail["body_included"] and "body" not in detail
    assert formula in detail["references"][0]["text"]
    with pytest.raises(ValueError):
        service.load("vertical_landing_and_altitude_control")


def test_route_traversal_skill_keeps_completion_core_and_optional_examples():
    from src.plugins.skills.knowledge_service import KnowledgeService
    service = KnowledgeService(_uav_flight_registry().skill_catalog)
    core = service.load("route_traversal_and_passage")
    assert "## Completion And Stop" in core["body"]
    assert core["references"] == []
    detail = service.load("route_traversal_and_passage", ["references/route_progress_examples.md"])
    assert not detail["body_included"]
    reference = detail["references"][0]["text"]
    assert "Passing A Landmark On One Side" in reference
    assert "Crossing A Gap Or Opening" in reference


def test_approach_skill_stops_before_overshoot_and_excludes_route_references():
    from src.plugins.skills.knowledge_service import KnowledgeService
    service = KnowledgeService(_uav_flight_registry().skill_catalog)
    result = service.load("approach_and_standoff")
    paths = {item["path"] for item in result["reference_manifest"]}
    assert "references/completion_examples.md" in paths
    assert "references/route_progress_examples.md" not in paths
    with pytest.raises(ValueError):
        service.load("approach_and_standoff", ["references/route_progress_examples.md"])
    assert service.loaded_index()[0]["reference_paths"] == []


def test_skill_frontmatter_summaries_are_selection_scoped_without_planner_details():
    catalog = _uav_flight_registry().skill_catalog
    forbidden = [
        "atan(",
        "vx",
        "vy",
        "vz",
        "yaw",
        "hover",
        "phase_decision",
        "phase_progress_evidence",
        "current_phase",
        "UAV-Flow",
        "UnrealCV",
    ]

    for skill in catalog.skills:
        summary = skill.summary
        assert summary
        assert summary.startswith("Select")
        for term in forbidden:
            assert term not in summary


def test_orbit_and_surround_skill_contains_orbit_guidance_without_runtime_internals():
    skill = _uav_flight_registry().skill_catalog.validate_selection(["orbit_and_surround"])[0][0]
    assert "Yaw alone is not orbit progress" in skill.body
    assert "tangential" in skill.body and "Worked example:" in skill.body
    for private in ("phase_decision", "phase_progress_evidence", "current_phase", "UAV-Flow", "hidden benchmark"):
        assert private not in skill.body


def test_flyagent_contains_visual_direction_basics_without_case_specific_terms():
    rules = Path("memory/FLYAGENT.md").read_text(encoding="utf-8")

    assert "## Identity And View" in rules
    assert "## Spatial Consistency" in rules
    assert "Turn toward a visible cue" in rules
    assert "negative yaw means turn left" not in rules
    assert "positive yaw means turn right" not in rules
    assert "turning away" in rules
    assert "does not prove physical travel" in rules
    forbidden_terms = [
        "UAV-Flow",
        "case id",
        "coupled planner",
        "phase judge",
        "model_inputs",
        "phase_progress_evidence",
        "current_phase",
    ]
    for term in forbidden_terms:
        assert term not in rules


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
    catalog = _uav_flight_registry().skill_catalog
    by_name = {skill.name: skill for skill in catalog.skills}

    assert len(cases) >= 20
    for case in cases:
        assert case["instruction"].strip()
        assert set(case.get("expected_any", [])) <= EXPECTED_UAV_FLIGHT_SKILLS
        assert set(case.get("forbidden", [])) <= EXPECTED_UAV_FLIGHT_SKILLS
        assert not set(case.get("expected_any", [])) & set(case.get("forbidden", []))
        expected_refs = case.get("expected_references", {})
        forbidden_refs = case.get("forbidden_references", {})
        assert set(expected_refs) <= EXPECTED_UAV_FLIGHT_SKILLS
        assert set(forbidden_refs) <= EXPECTED_UAV_FLIGHT_SKILLS
        for skill_name, paths in {**expected_refs, **forbidden_refs}.items():
            available_paths = {item["path"] for item in by_name[skill_name].index_listing().get("references", [])}
            assert set(paths) <= available_paths
        assert int(case.get("max_selected", 0)) >= 0


def test_skill_catalog_rejects_unknown_reference_paths():
    catalog = _uav_flight_registry().skill_catalog
    accepted, _ = catalog.validate_selection(["visual_target_task_reasoning"])

    valid, rejected = catalog.validate_reference_selection(
        accepted,
        {
            "visual_target_task_reasoning": [
                "references/search_under_uncertainty.md",
                "references/missing.md",
                "../outside.md",
            ]
        },
    )

    assert valid == {"visual_target_task_reasoning": ["references/search_under_uncertainty.md"]}
    assert rejected == {"visual_target_task_reasoning": ["references/missing.md", "../outside.md"]}



def test_skills_context_is_directory_only_and_load_index_is_isolated():
    registry = _uav_flight_registry()
    module = DefaultSkillsContextModule(warnings=registry.warnings)
    state = TaskContextInput("Inspect route")
    result = module.build(MemorySourceBundle(None), state)
    assert result.fragment.content == {}
    assert "skill_text" not in str(result.fragment.content)
    assert "selected_skills" not in result.fragment.content
    assert "knowledge_catalog" not in result.fragment.content
    assert not hasattr(state, "loaded_knowledge")
