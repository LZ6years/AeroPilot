"""Tests for UAV-Agent plugin discovery and skill loading."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from src.agent.semantic.task_parser import TaskParser
from src.memory.context.modules.skills import DefaultSkillsContextModule
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.flight_memory import FlightMemory
from src.plugins.core.discovery import PluginDiscovery
from src.plugins.core.registry import PluginRegistry
from src.plugins.skills import SkillContentLoader
from src.plugins.skills.loader import DEFAULT_MAX_SKILL_CHARS
from src.plugins.skills.semantic_criteria import SemanticCriteriaSkillResolver
from src.plugins.skills.selection import ModelSkillSelector, SkillSelectionResult
from src.runtime.configs.model_config import ModelConfig


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


def make_stage_context(result) -> StageContextInput:
    return StageContextInput(
        active_instruction=result.instruction,
        semantic_plan=result.plan.to_dict(),
        current_phase=result.plan.phases[0].to_dict(),
        semantic_goal=result.plan.task_goal.to_dict(),
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
    assert "target absent" in normalized_summary
    assert "weak candidate" in normalized_summary
    assert "directional target cue" in normalized_summary
    assert "inspection-ready" in normalized_summary
    assert "usable image bearing" in normalized_summary
    assert "Usable current image bearing is operational evidence for visual alignment" in normalized_summary
    assert "a separate geometric heading measurement is not required" not in normalized_summary
    assert "Moderate left or right offset is explicitly acceptable" in normalized_summary
    assert "exact centering is not required unless explicitly requested" in normalized_summary
    assert "Do not correct merely for lateral offset" in normalized_summary
    assert "correct only when the cue is near the frame boundary" not in normalized_summary
    assert "broad usable forward region" not in normalized_summary
    assert "unless explicitly requested" in normalized_summary
    assert "Do not restart or reverse broad search" in normalized_summary
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
    assert "target absent" in summaries["visual_target_task_reasoning"]["skill_text"]
    assert "weak candidate" in summaries["visual_target_task_reasoning"]["skill_text"]
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
    assert "30-45 degrees" in planner_guidance_with_ref["requested_references"][0]["text"]


def test_semantic_criteria_skill_resolver_keeps_all_valid_skills_without_internal_resources():
    class InspectingSelector:
        def __init__(self):
            self.context = None

        def select(self, selection_context, catalog):
            self.context = selection_context
            names = [skill.name for skill in catalog.skills]
            return SkillSelectionResult(
                requested_skill_names=[*names, names[0], "disabled_or_unknown_skill"],
                accepted_skill_names=[*names, names[0], "disabled_or_unknown_skill"],
                requested_references={names[0]: ["references/search_under_uncertainty.md"]},
                status="selected",
            )

    selector = InspectingSelector()
    resolver = SemanticCriteriaSkillResolver(
        ModelConfig(model_name="test-model", api_key="secret"),
        plugin_registry=_uav_flight_registry(),
        selector=selector,
        content_loader=SkillContentLoader(max_skill_chars=80),
    )

    result = resolver.resolve(
        "Face and then pass the landmark while preserving safe distance.",
        {"target": "landmark", "target_attributes": {}, "final_state": "past_landmark"},
        {"index": 2, "total": 2, "name": "pass_landmark", "goal": "Pass the landmark"},
    )

    assert result.selected_skill_names == [skill.name for skill in _uav_flight_registry().skill_catalog.skills]
    assert len(result.selected_skill_names) == len(EXPECTED_UAV_FLIGHT_SKILLS)
    assert "skill_selection_rejected:disabled_or_unknown_skill" in result.warnings
    assert "semantic_criteria_skill_references_ignored" in result.warnings
    assert all("semantic_criteria_skill_truncated:" in warning for warning in result.warnings[-5:])
    assert selector.context["current_phase"] == {
        "index": 2,
        "name": "pass_landmark",
        "goal": "Pass the landmark",
    }
    for listing in selector.context["available_skills"]:
        assert set(listing) == {"name", "description", "summary"}
    for guidance in result.selected_skills:
        assert set(guidance) == {"name", "description", "skill_text"}
        assert len(guidance["skill_text"]) <= 80


def test_semantic_skill_guidance_preserves_completion_tail_within_bound():
    catalog = _uav_flight_registry().skill_catalog
    accepted, _ = catalog.validate_selection(["route_traversal_and_passage"])

    guidance = SkillContentLoader(max_skill_chars=800).load_semantic_guidance(accepted[0])

    assert len(guidance["skill_text"]) <= 800
    assert "bounded middle omitted" in guidance["skill_text"]
    assert "## Application Boundary" in guidance["skill_text"]
    assert "translation evidence with repeated yaw" in guidance["skill_text"]
    assert guidance["truncated"] is True


def test_search_under_uncertainty_keeps_broad_search_out_of_candidate_confirmation():
    skill_text = Path("plugins/knowledge/uav_flight/visual_target_task_reasoning/SKILL.md").read_text(encoding="utf-8")
    reference_text = Path(
        "plugins/knowledge/uav_flight/visual_target_task_reasoning/references/search_under_uncertainty.md"
    ).read_text(encoding="utf-8")
    normalized_skill_text = " ".join(skill_text.split())
    normalized_reference_text = " ".join(reference_text.split())

    assert "target absent" in normalized_skill_text
    assert "weak candidate" in normalized_skill_text
    assert "directional target cue" in normalized_skill_text
    assert "inspection-ready" in normalized_skill_text
    assert "30-45 degrees" not in normalized_skill_text
    assert "current phase decision" not in normalized_skill_text
    assert "artifact paths" not in normalized_skill_text
    assert "hidden benchmark" not in normalized_skill_text
    assert "sweep consistently into new views" in normalized_skill_text
    assert "Absence alone does not justify reversal" in normalized_skill_text
    assert "stop broad search" in normalized_skill_text
    assert "30-45 degrees is broad search, not a small/local confirmation turn" in normalized_reference_text
    assert "switch from broad sweep to hover or a small 5-15 degree yaw" in normalized_reference_text
    assert "continue a 30-45 degree broad sweep as if it were confirmation" in normalized_reference_text
    assert "Reverse a broad sweep only when" in normalized_reference_text


def test_vertical_landing_skill_keeps_core_completion_and_moves_formulas_to_reference():
    skill_text = Path(
        "plugins/knowledge/uav_flight/vertical_landing_and_altitude_control/SKILL.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(skill_text.split())

    reference_text = Path(
        "plugins/knowledge/uav_flight/vertical_landing_and_altitude_control/references/angled_vertical_motion.md"
    ).read_text(encoding="utf-8")
    normalized_reference = " ".join(reference_text.split())

    assert "controlled landing" in normalized
    assert "angled ascent/descent" in normalized
    assert "Stop descending" in normalized
    assert "requested drop is reached" in normalized
    assert "landing surface is near enough" in normalized
    assert "shorten adjustments" in normalized
    assert "hover or use a smaller adjustment" in normalized
    assert "landing region" in normalized
    assert "vertical progress" in normalized
    assert "available clearance or surface cues" in normalized
    assert "near a surface" in normalized
    assert "vertical error" in normalized
    assert "deviation from the vertical ascent/descent axis" in normalized
    assert "angle = atan(sqrt(vx^2 + vy^2) / abs(vz))" not in normalized
    assert "sqrt(vx^2 + vy^2) = abs(vz) * tan(angle)" not in normalized
    assert "angle = atan(sqrt(vx^2 + vy^2) / abs(vz))" in normalized_reference
    assert "sqrt(vx^2 + vy^2) = abs(vz) * tan(angle)" in normalized_reference
    assert "smaller angle is closer to straight vertical motion" in normalized_reference
    assert "non-zero horizontal component" in normalized_reference
    assert "relative to the horizontal plane" not in normalized
    assert "abs(vertical_speed) / horizontal_speed = tan(angle)" not in normalized
    assert "yaw, heading, vehicle pitch, or camera angle" in normalized
    assert "effective bounded progress" in normalized
    assert "reasonable tolerance" in normalized
    assert "decimal-level correction" in normalized
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
    assert "bounded translation" in normalized
    assert "first useful route view with the latest view" in normalized
    assert "cue as a destination" in normalized
    assert "exact geometric boundary" in normalized
    assert "not a fixed maneuver sequence" in normalized
    assert "Yaw can inspect a route or change image bearing" in normalized
    assert "Cue absence alone is ambiguous" in normalized
    assert "Passage completion is a terminal route relation" in normalized
    assert "cue moves from ahead toward a side, rearward" in normalized
    assert "cue leaves the forward view" in normalized
    assert "Do not demand every evidence form, reacquisition, continued visibility" in normalized
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

    assert "Exclude routes ending through, past, beyond, across, or around a reference" in normalized
    assert "final held front/back/side position" in normalized
    assert "A side relation belongs here only when the UAV must stop or hold there" in normalized
    assert "target visibility can conflict with traversal" in normalized
    assert "Target identity and target-relative range are independently assessable facts" not in normalized
    assert "For qualitative requests, form a credible rough range" in normalized
    assert "Assess identity and range separately" not in normalized
    assert "tight representative interior region" in normalized
    assert "Yaw cannot reach a distance or side relation" in normalized
    assert "familiar scale, ground contact" not in normalized
    assert "clearly place the UAV too far" in normalized
    assert "clearly place it too near" in normalized
    assert "before correcting distance" in normalized
    assert "Completion requires credible target identity. Separately" not in normalized
    assert "broadly near the requested relation" in normalized
    assert "explicit metric standoff additionally requires a fresh small-ROI measurement" in normalized
    assert "Do not infer a decimal distance from the color bar" in normalized
    assert "Reverse only when new evidence clearly places the relation on the opposite side" in normalized
    assert "contains the requested value" not in normalized
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
    assert "do not enforce a fixed maneuver sequence" in normalized
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

    assert "## Who You Are" in rules
    assert "## Spatial And Action Consistency" in rules
    assert "When a visible cue is on the left" in rules
    assert "Apply the same rule in reverse" in rules
    assert "negative yaw means turn left" not in rules
    assert "positive yaw means turn right" not in rules
    assert "Do not turn away from a visible cue" in rules
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


def test_manual_selector_probe_builds_task_relevant_semantic_context():
    probe_path = Path("tests/manual/plugin_system_real_probe.py")
    spec = importlib.util.spec_from_file_location("plugin_system_real_probe", probe_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    land = module._semantic_plan_for_instruction("Land on the right side of the sunshade")
    assert land.intent == "vertical_landing"
    assert land.task_goal.target == "sunshade"
    assert land.task_goal.target_attributes["relation"] == "right_side"
    assert land.phases[0].name == "controlled_landing_near_requested_region"
    assert "altitude or surface progress is insufficient" in land.phases[0].continue_when
    assert "metrics" not in land.phases[0].to_dict()

    route = module._semantic_plan_for_instruction("Please navigate through the sun umbrella from the left side")
    assert route.intent == "route_traversal"
    assert route.task_goal.target == "sun umbrella"
    assert route.task_goal.target_attributes["relation"] == "left_side"
    assert route.phases[0].name == "traverse_requested_route"

    descend = module._semantic_plan_for_instruction("Descend 9.0 meters at 30 degrees")
    assert descend.intent == "altitude_control"
    assert "metrics" not in descend.phases[0].to_dict()

    fallback = module._semantic_plan_for_instruction("Hold this area conservatively")
    assert fallback.intent == "general_visual_navigation"
    assert fallback.task_goal.target != "person"

    expected_refs = module._expected_reference_map(
        {"visual_target_task_reasoning": ["references\\search_under_uncertainty.md"]}
    )
    assert expected_refs == {
        "visual_target_task_reasoning": ["references/search_under_uncertainty.md"]
    }
    assert module._missing_expected_references(
        expected_refs,
        {"visual_target_task_reasoning": ["references/search_under_uncertainty.md"]},
    ) == {}
    assert module._missing_expected_references(expected_refs, {}) == expected_refs
    assert module._forbidden_reference_hits(
        expected_refs,
        {"visual_target_task_reasoning": ["references/search_under_uncertainty.md"]},
    ) == expected_refs
    assert module._forbidden_reference_hits(expected_refs, {}) == {}


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


def test_enabled_model_skill_selector_requires_explicit_config():
    with pytest.raises(ValueError, match="explicit ModelConfig"):
        ModelSkillSelector()

    config = ModelConfig(
        model_name="test-model",
        api_key="secret",
        base_url="https://example.test/v1",
    )
    selector = ModelSkillSelector(config)

    assert selector.model_config is config


def test_skills_module_does_not_cache_transient_selector_errors():
    class TransientSelector:
        def __init__(self):
            self.calls = 0

        def select(self, selection_context, catalog):
            self.calls += 1
            if self.calls == 1:
                return SkillSelectionResult(
                    status="model_error",
                    warnings=["skill_selection_failed:temporary"],
                )
            return SkillSelectionResult(
                requested_skill_names=["visual_target_task_reasoning"],
                accepted_skill_names=["visual_target_task_reasoning"],
                status="selected",
            )

    result = TaskParser(mode="rule_based").parse("观察右侧区域，不确定时问我。")
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", active_instruction=result.instruction)
    selector = TransientSelector()
    module = DefaultSkillsContextModule(
        plugin_registry=_uav_flight_registry(),
        skill_selector=selector,
    )
    stage_context = make_stage_context(result)

    first = module.build(MemorySourceBundle(memory), stage_context).to_dict()
    second = module.build(MemorySourceBundle(memory), stage_context).to_dict()
    third = module.build(MemorySourceBundle(memory), stage_context).to_dict()

    assert selector.calls == 2
    assert first["metadata"]["selection"]["status"] == "model_error"
    assert first["metadata"]["selection_source"] == "fresh"
    assert second["metadata"]["selection"]["status"] == "selected"
    assert second["metadata"]["selection_source"] == "fresh"
    assert third["metadata"]["selection_source"] == "phase_cache"


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


def test_model_skill_selector_accepts_requested_references_for_selected_skills():
    class ReferenceSelector(ModelSkillSelector):
        def __init__(self):
            super().__init__(
                ModelConfig(model_name="test-model", api_key="secret", base_url="https://example.test/v1")
            )

        def _request_content(self, selection_context):
            return json.dumps(
                {
                    "selected_skills": ["visual_target_task_reasoning"],
                    "requested_references": {
                        "visual_target_task_reasoning": ["references/not_selected.md"],
                    },
                }
            )

    registry = _uav_flight_registry()

    result = ReferenceSelector().select(
        {"available_skills": registry.skill_catalog.index_listings()},
        registry.skill_catalog,
    )

    assert result.status == "selected"
    assert result.accepted_skill_names == ["visual_target_task_reasoning"]
    assert result.accepted_references == {}
    assert result.rejected_references == {"visual_target_task_reasoning": ["references/not_selected.md"]}
    assert "skill_reference_rejected:visual_target_task_reasoning:references/not_selected.md" in result.warnings


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
            return "selected_skills should include visual_target_task_reasoning"

    registry = _uav_flight_registry()

    result = InvalidSelector().select({"available_skills": registry.skill_catalog.listings()}, registry.skill_catalog)

    assert result.status == "selected"
    assert result.accepted_skill_names == ["visual_target_task_reasoning"]
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
    assert "summary" not in fragment["content"]["selected_skills"][0]
    assert "skill_text" in fragment["content"]["selected_skills"][0]
    assert "Visual Target Task Reasoning" in fragment["content"]["selected_skills"][0]["skill_text"]
    assert "skill_text_truncated" not in fragment["content"]["selected_skills"][0]
    assert "plugin_id" not in fragment["content"]["selected_skills"][0]
    assert "source_path" not in fragment["content"]["selected_skills"][0]
    assert "available_references" not in fragment["content"]["selected_skills"][0]
    assert fragment["facts"] == []
    assert fragment["warnings"].count("skill_selection_rejected:missing_skill") >= 1


def test_skills_module_loads_only_requested_references_for_selected_skill():
    class FixedSelector:
        def select(self, selection_context, catalog):
            assert "semantic_plan" not in selection_context
            assert "metadata" not in selection_context
            assert "phase_progress_brief" in selection_context
            available_skill = next(
                item for item in selection_context["available_skills"] if item["name"] == "visual_target_task_reasoning"
            )
            assert "source_path" not in available_skill
            assert "skill_text" not in available_skill
            assert available_skill["references"][0]["path"] == "references/search_under_uncertainty.md"
            return SkillSelectionResult(
                requested_skill_names=["visual_target_task_reasoning"],
                accepted_skill_names=["visual_target_task_reasoning"],
                requested_references={"visual_target_task_reasoning": ["references/search_under_uncertainty.md"]},
                accepted_references={"visual_target_task_reasoning": ["references/search_under_uncertainty.md"]},
                status="selected",
            )

    instruction = "Search for the person near the street."
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", active_instruction=instruction)
    module = DefaultSkillsContextModule(plugin_registry=_uav_flight_registry(), skill_selector=FixedSelector())

    fragment = module.build(
        MemorySourceBundle(memory),
        StageContextInput(
            active_instruction=instruction,
            semantic_plan={},
            current_phase={
                "name": "search_person",
                "goal": "find the person candidate",
                "done_when": ["person candidate is visible"],
                "continue_when": ["person is not visible or ambiguous"],
            },
            semantic_goal={"target": "person"},
            phase_execution_summary={
                "window_count": 1,
                "measurements": {"yaw_delta_deg": 30.0},
                "tool_facts": {"tools_executed": {"uav.turn_relative": 1}},
            },
        ),
    ).to_dict()

    selected = fragment["content"]["selected_skills"][0]
    assert selected["name"] == "visual_target_task_reasoning"
    assert "summary" not in selected
    assert "skill_text" in selected
    assert "Visual Target Task Reasoning" in selected["skill_text"]
    assert "No-candidate broad search" not in selected["skill_text"]
    assert selected["requested_references"][0]["path"] == "references/search_under_uncertainty.md"
    assert "30-45 degrees" in selected["requested_references"][0]["text"]
    assert fragment["metadata"]["available_skill_references"]["visual_target_task_reasoning"] == [
        "references/search_under_uncertainty.md"
    ]


def test_skills_module_selection_context_keeps_phase_only_policy_in_prompt_not_context_shape():
    class InspectingSelector:
        def select(self, selection_context, catalog):
            assert set(selection_context) == {
                "active_instruction",
                "current_phase",
                "semantic_goal",
                "phase_progress_brief",
                "available_skills",
                "instruction",
            }
            assert "selection_scope" not in selection_context
            assert "final_goal_is_context_only" not in selection_context
            assert selection_context["current_phase"]["name"] == "locate_and_approach_landmark"
            assert "landing" in selection_context["semantic_goal"]["final_state"]
            assert selection_context["instruction"].startswith("Select zero or more skill names")
            return SkillSelectionResult(status="none_selected")

    instruction = "Land on the right side of the landmark."
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", active_instruction=instruction)
    module = DefaultSkillsContextModule(plugin_registry=_uav_flight_registry(), skill_selector=InspectingSelector())

    fragment = module.build(
        MemorySourceBundle(memory),
        StageContextInput(
            active_instruction=instruction,
            semantic_plan={},
            current_phase={
                "name": "locate_and_approach_landmark",
                "goal": "Identify the landmark and approach the requested side while maintaining visual contact",
                "done_when": ["landmark is visible and the requested side relation is plausible"],
                "continue_when": ["landmark is not identified or the requested side is not yet plausible"],
            },
            semantic_goal={"target": "landmark", "final_state": "landing_on_requested_side"},
            phase_execution_summary={"window_count": 0, "measurements": {}, "tool_facts": {}},
        ),
    ).to_dict()

    assert fragment["content"]["selected_skills"] == []
    assert fragment["metadata"]["selection"]["status"] == "none_selected"


def test_skills_module_reuses_selection_within_phase_and_refreshes_on_phase_change():
    class CountingSelector:
        def __init__(self):
            self.calls = 0

        def select(self, selection_context, catalog):
            self.calls += 1
            return SkillSelectionResult(
                requested_skill_names=["visual_target_task_reasoning"],
                accepted_skill_names=["visual_target_task_reasoning"],
                status="selected",
            )

    result = TaskParser(mode="rule_based").parse("观察右侧区域，不确定时问我。")
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", active_instruction=result.instruction)
    selector = CountingSelector()
    module = DefaultSkillsContextModule(plugin_registry=_uav_flight_registry(), skill_selector=selector)
    phase_one = make_stage_context(result)
    phase_one = StageContextInput(
        **{**phase_one.to_dict(), "metadata": {"run_id": "run_1", "current_phase_key": "1:observe_area"}}
    )
    phase_two = StageContextInput(
        **{**phase_one.to_dict(), "metadata": {"run_id": "run_1", "current_phase_key": "2:clarify_if_uncertain"}}
    )

    first = module.build(MemorySourceBundle(memory), phase_one).to_dict()
    second = module.build(MemorySourceBundle(memory), phase_one).to_dict()
    third = module.build(MemorySourceBundle(memory), phase_two).to_dict()

    assert selector.calls == 2
    assert first["metadata"]["selection_source"] == "fresh"
    assert second["metadata"]["selection_source"] == "phase_cache"
    assert third["metadata"]["selection_source"] == "fresh"
    assert second["content"]["selected_skills"][0]["name"] == "visual_target_task_reasoning"
