"""Tests for UAV-Agent plugin discovery and skill loading."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from src.memory.context.modules.skills import DefaultSkillsContextModule
from src.memory.context.contracts import MemorySourceBundle, TaskContextInput
from src.memory.flight_memory import FlightMemory
from src.plugins.core.discovery import PluginDiscovery
from src.plugins.core.registry import PluginRegistry
from src.plugins.skills import SkillContentLoader
from src.plugins.skills.loader import DEFAULT_MAX_SKILL_CHARS
from src.agent.config import ModelConfig


EXPECTED_UAV_FLIGHT_SKILLS = {
    "visual_target_task_reasoning",
    "approach_and_standoff",
    "vertical_landing_and_altitude_control",
    "route_traversal_and_passage",
    "orbit_and_surround",
}

REMOVED_UAV_FLIGHT_SKILLS = {
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


def test_uav_knowledge_skills_follow_compact_five_section_contract():
    catalog = _uav_flight_registry().skill_catalog

    for skill in catalog.skills:
        headings = [
            line.removeprefix("## ").strip()
            for line in skill.body.splitlines()
            if line.startswith("## ")
        ]
        assert headings == KNOWLEDGE_SKILL_SECTIONS, skill.name
        assert len(skill.body.strip()) <= DEFAULT_MAX_SKILL_CHARS, skill.name
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


def test_skill_catalog_validates_selected_names_and_loader_reads_skill_resources():
    catalog = _uav_flight_registry().skill_catalog

    accepted, rejected = catalog.validate_selection(["visual_target_task_reasoning", "missing_skill"])

    assert [skill.name for skill in accepted] == ["visual_target_task_reasoning"]
    assert rejected == ["missing_skill"]
    index = accepted[0].index_listing()
    assert index["name"] == "visual_target_task_reasoning"
    assert "summary" in index
    assert "source_path" not in index
    assert "skill_text" not in index
    assert index["references"] == [
        {
            "path": "references/search_under_uncertainty.md",
            "description": "Optional reference for search under uncertainty.",
        }
    ]
    summary = SkillContentLoader().load_summary(accepted[0])
    assert summary["name"] == "visual_target_task_reasoning"
    assert summary["summary"]
    assert summary["available_references"] == ["references/search_under_uncertainty.md"]
    assert "Visual Target Task Reasoning" in summary["skill_text"]
    normalized_summary = " ".join(summary["skill_text"].split())
    assert "## Evidence Model" in summary["skill_text"]
    assert "candidate outside the current view" in normalized_summary
    assert "tentative candidate" in normalized_summary
    assert "directional cue" in normalized_summary
    assert "usable bearing" in normalized_summary
    assert "Current bearing supports alignment" in normalized_summary
    assert "a separate geometric heading measurement is not required" not in normalized_summary
    assert "Moderate left or right offset is explicitly acceptable" in normalized_summary
    assert "exact centering needs an explicit request" in normalized_summary
    assert "Do not correct merely for lateral offset" in normalized_summary
    assert "correct only when the cue is near the frame boundary" not in normalized_summary
    assert "broad usable forward region" not in normalized_summary
    assert "an explicit request" in normalized_summary
    assert "Contradicted attributes" in normalized_summary
    assert "alternate around an already usable bearing" in normalized_summary
    assert "left/negative yaw" not in normalized_summary
    assert "right/positive yaw" not in normalized_summary
    assert "current phase decision" not in summary["skill_text"]
    assert "good enough to advance" not in summary["skill_text"]
    assert "motion_history_summary" not in summary["skill_text"]
    assert "phase_action_summary" not in summary["skill_text"]
    assert summary["references"] == []
    assert summary["assets"]["scripts"] == []


def test_skill_loader_lists_references_without_executing_scripts():
    catalog = _uav_flight_registry().skill_catalog

    accepted, _ = catalog.validate_selection(["visual_target_task_reasoning"])
    summaries = {skill.name: SkillContentLoader().load_summary(skill) for skill in accepted}

    visual_refs = summaries["visual_target_task_reasoning"]["available_references"]
    assert visual_refs == ["references/search_under_uncertainty.md"]
    assert "candidate outside the current view" in summaries["visual_target_task_reasoning"]["skill_text"]
    assert "tentative candidate" in summaries["visual_target_task_reasoning"]["skill_text"]
    assert "30-45 degrees" not in summaries["visual_target_task_reasoning"]["skill_text"]
    assert summaries["visual_target_task_reasoning"]["assets"]["scripts"] == []
    planner_guidance = SkillContentLoader().load_planner_guidance(accepted[0])
    assert planner_guidance["name"] == "visual_target_task_reasoning"
    assert "summary" not in planner_guidance
    assert "skill_text" in planner_guidance
    assert "Visual Target Task Reasoning" in planner_guidance["skill_text"]
    assert "skill_text_truncated" not in planner_guidance
    assert planner_guidance["skill_text"] == accepted[0].body.strip()
    assert "references" not in planner_guidance
    assert "requested_references" not in planner_guidance
    assert "source_path" not in planner_guidance
    assert "skill_dir" not in planner_guidance
    assert "available_references" not in planner_guidance
    planner_guidance_with_ref = SkillContentLoader().load_planner_guidance(
        accepted[0],
        requested_references=["references/search_under_uncertainty.md"],
    )
    assert planner_guidance_with_ref["requested_references"][0]["path"] == "references/search_under_uncertainty.md"
    assert "view overlap" in planner_guidance_with_ref["requested_references"][0]["text"]


def test_search_under_uncertainty_keeps_broad_search_out_of_candidate_confirmation():
    skill_text = Path("plugins/knowledge/uav_flight/visual_target_task_reasoning/SKILL.md").read_text(encoding="utf-8")
    reference_text = Path(
        "plugins/knowledge/uav_flight/visual_target_task_reasoning/references/search_under_uncertainty.md"
    ).read_text(encoding="utf-8")
    normalized_skill_text = " ".join(skill_text.split())
    normalized_reference_text = " ".join(reference_text.split())

    assert "candidate outside the current view" in normalized_skill_text
    assert "tentative candidate" in normalized_skill_text
    assert "directional cue" in normalized_skill_text
    assert "30-45 degrees" not in normalized_skill_text
    assert "current phase decision" not in normalized_skill_text
    assert "artifact paths" not in normalized_skill_text
    assert "hidden benchmark" not in normalized_skill_text
    assert "Inspection and arrival remain separate outcomes" in normalized_skill_text
    assert "Contradicted attributes" in normalized_skill_text
    assert "view overlap" in normalized_reference_text
    assert "30-45" not in reference_text and "5-15" not in reference_text
    assert "Reorient when evidence changes" in normalized_reference_text


def test_vertical_landing_skill_keeps_core_completion_and_moves_formulas_to_reference():
    skill_text = Path(
        "plugins/knowledge/uav_flight/vertical_landing_and_altitude_control/SKILL.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(skill_text.split())

    reference_text = Path(
        "plugins/knowledge/uav_flight/vertical_landing_and_altitude_control/references/angled_vertical_motion.md"
    ).read_text(encoding="utf-8")
    normalized_reference = " ".join(reference_text.split())

    assert "landing" in normalized
    assert "angled ascent/descent" in normalized
    assert "Stop descending" in normalized
    assert "at the requested drop" in normalized
    assert "near the landing surface" in normalized
    assert "shorten adjustments" in normalized
    assert "meaningfully different viewpoint or clearance" in normalized
    assert "landing region" in normalized
    assert "height change, progress, clearance and surface cues" in normalized
    assert "Near a surface" in normalized
    assert "risk or error" in normalized
    assert "Angled paths deviate from the vertical axis" in normalized
    assert "angle = atan(sqrt(vx^2 + vy^2) / abs(vz))" not in normalized
    assert "sqrt(vx^2 + vy^2) = abs(vz) * tan(angle)" not in normalized
    assert "angle = atan(sqrt(vx^2 + vy^2) / abs(vz))" in normalized_reference
    assert "sqrt(vx^2 + vy^2) = abs(vz) * tan(angle)" in normalized_reference
    assert "smaller angle is closer to straight vertical motion" in normalized_reference
    assert "non-zero horizontal component" in normalized_reference
    assert "relative to the horizontal plane" not in normalized
    assert "abs(vertical_speed) / horizontal_speed = tan(angle)" not in normalized
    assert "yaw, heading, pitch or camera angle" in normalized
    assert "effective bounded progress" not in normalized
    assert "reasonable tolerance" in normalized
    assert "not decimal correction" in normalized
    assert "contact sensor" not in normalized
    assert "hidden benchmark" not in normalized
    forbidden_terms = [
        "UAV-Flow",
        "UnrealCV",
        "phase_progress_evidence",
        "current_phase",
        "context",
        "case id",
    ]
    for term in forbidden_terms:
        assert term not in skill_text


def test_route_traversal_skill_keeps_completion_core_and_optional_examples():
    skill_text = Path(
        "plugins/knowledge/uav_flight/route_traversal_and_passage/SKILL.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(skill_text.split())
    reference_text = Path(
        "plugins/knowledge/uav_flight/route_traversal_and_passage/references/route_progress_examples.md"
    ).read_text(encoding="utf-8")
    normalized_reference = " ".join(reference_text.split())

    assert "route progress" in normalized
    assert "pass through" in normalized
    assert "gap" in normalized
    assert "translate through or past it" in normalized
    assert "first useful route view with the latest view" in normalized
    assert "cue is not necessarily a destination" in normalized
    assert "exact unseen boundary" in normalized
    assert "Choose the next observation point" in normalized
    assert "Yaw can inspect a route or change image bearing" in normalized
    assert "Cue absence alone is ambiguous" in normalized
    assert "Passage completion requires a terminal route relation" in normalized
    assert "cue moving sideways, rearward" in normalized
    assert "cue leaves view" in normalized
    assert "Do not require every source, reacquisition, continued visibility" in normalized
    assert "done_when" not in normalized
    assert "continue_when" not in normalized
    assert "schema" not in normalized
    assert "list item" not in normalized
    assert "Passing A Landmark On One Side" in normalized_reference
    assert "Crossing A Gap Or Opening" in normalized_reference
    assert "camera reorientation rather than traversal" in normalized_reference
    assert "next semantic step" not in normalized
    assert "good enough to continue" not in normalized
    forbidden_terms = [
        "UAV-Flow",
        "UnrealCV",
        "phase_progress_evidence",
        "current_phase",
        "context",
        "case id",
    ]
    for term in forbidden_terms:
        assert term not in skill_text


def test_approach_skill_stops_before_overshoot_and_excludes_route_references():
    skill_text = Path(
        "plugins/knowledge/uav_flight/approach_and_standoff/SKILL.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(skill_text.split())

    assert "Traversal belongs to route guidance" in normalized
    assert "Observation-only approach does not establish identity" in normalized
    assert "UAV displacement is not target distance" in normalized
    assert "before correcting distance" in normalized
    assert "Numeric standoff needs valid target-relative measurement" in normalized
    assert "no repeated measurement is imposed" in normalized
    assert "Turn, rise, shift or retrace" in normalized
    forbidden_terms = [
        "UAV-Flow",
        "UnrealCV",
        "phase_progress_evidence",
        "current_phase",
        "context",
        "case id",
    ]
    for term in forbidden_terms:
        assert term not in skill_text


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
    skill_text = Path(
        "plugins/knowledge/uav_flight/orbit_and_surround/SKILL.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(skill_text.split())

    assert "lateral or tangential translation" in normalized
    assert "Yaw alone is not orbit progress" in normalized
    assert "not displacement from the UAV's local origin" in normalized
    assert "brief radial repair without replacing circulation" in normalized
    assert "Orbit direction controls the shift direction, not the yaw sign" in normalized
    assert "Stop orbit movement" in normalized
    assert "repeated radial corrections" in normalized
    assert "choose a different circulation viewpoint" in normalized
    assert "semantic phase" not in normalized
    assert "phase_decision" not in normalized
    assert "phase_progress_evidence" not in normalized
    assert "current_phase" not in normalized
    assert "UAV-Flow" not in normalized
    assert "hidden benchmark" not in normalized
    assert "raw file names" not in normalized
    assert "reference paths" not in normalized


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
