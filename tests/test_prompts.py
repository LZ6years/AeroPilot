"""Test prompt files used by model-backed planning."""

from __future__ import annotations

from pathlib import Path

from src.plugins.skills.semantic_criteria import SEMANTIC_CRITERIA_SKILL_SELECTION_PROMPT
from src.plugins.skills.selection import SKILL_SELECTION_SYSTEM_PROMPT


def test_native_prompt_prefers_direct_vision_without_forcing_detection_only():
    prompt = Path("src/agent/planning/prompts/agent_tool_loop_system.txt").read_text(encoding="utf-8")
    assert "decision-relevant visual ambiguity" in prompt
    assert "rough\nlocation is the only question" not in prompt
    assert "If every done_when item is supported" in prompt
    assert "runtime_report_blocked" in prompt


def test_task_parser_prompts_keep_structure_and_criteria_contracts_separate():
    prompt_dir = Path("src") / "agent" / "semantic" / "prompts"
    structure = (prompt_dir / "task_parser_structure_system.txt").read_text(encoding="utf-8")
    criteria = (prompt_dir / "task_parser_phase_criteria_system.txt").read_text(encoding="utf-8")
    done_when = (prompt_dir / "done_when_generation_system.txt").read_text(encoding="utf-8")
    structure_shape = structure.split("Boundaries:", 1)[0]
    criteria_shape = criteria.split("Assembly contract:", 1)[0]

    assert not (prompt_dir / "task_parser_system.txt").exists()
    assert '"intent"' in structure_shape
    assert '"task_goal"' in structure_shape
    assert '"target_attributes"' in structure_shape
    assert '"done_when"' not in structure_shape
    assert '"observe"' not in structure_shape
    assert '"continue_when"' not in structure_shape
    assert '"metrics"' not in structure_shape
    assert "One phase is one local route leg" in structure
    assert "It is not one primitive action" in structure
    assert "Do not split only because movement mode or punctuation changes" in structure
    assert "Never weaken a result into preparing" in structure
    assert "Create an orientation result only when" in structure
    assert '"done_when"' in criteria_shape
    assert '"observe"' not in criteria_shape
    assert '"continue_when"' in criteria_shape
    assert '"metrics"' not in criteria_shape
    assert '"intent"' not in criteria_shape
    assert '"task_goal"' not in criteria_shape
    assert "Preserve user-provided numeric requirements in natural language" in criteria
    assert "configured evidence capabilities only to avoid an inherently" in criteria
    assert "terminal obligations" in criteria
    assert "continue_when is a concise missing or contradicted user result" in criteria
    assert "Keep target-relative distance distinct from UAV self-motion" in criteria
    assert "minimum sufficient user result" in done_when
    assert "Never split by evidence source" in done_when
    assert "Split only when" in done_when
    assert "invented axes" in done_when
    assert "Add a final heading only when" in done_when
    assert '"as judged by"' in done_when
    assert "Evidence is not part of the terminal state itself" in done_when
    for task_specific in ("Approach", "Retreat", "Pass", "Land", "Orbit", "Surround"):
        assert task_specific not in structure + criteria + done_when
    assert '"action_bias"' not in structure + criteria
    assert '"direction":' not in structure + criteria
    assert '"uncertainties"' not in structure + criteria


def test_legacy_action_and_phase_judge_prompts_removed():
    prompt_dir = Path("src") / "agent" / "planning" / "prompts"

    assert not (prompt_dir / "action_planner_system.txt").exists()
    assert not (prompt_dir / "phase_transition_judge_system.txt").exists()


def test_agent_boundaries_document_prompt_and_skill_writing_rules():
    guide = Path("AGENT.md").read_text(encoding="utf-8")
    compact_guide = " ".join(guide.split())

    assert "Prompt And Skill Writing Rules" in guide
    assert "Reusable skills are flight-domain knowledge" in guide
    assert "`current_phase`" in guide
    assert "`phase_decision`" in guide
    assert "`phase_progress_evidence`" in guide
    assert "`model_inputs`" in guide
    assert "Do not restore model-facing `direction` shorthand" in guide
    assert "natural-language completion conditions" in guide
    assert "hidden benchmark truth" in guide
    assert "`memory/FLYAGENT.md` owns only the shared embodied Agent identity" in guide
    assert "`memory/USER.md` owns ordinary human spatial" in guide
    assert "`src/agent/semantic/prompts/done_when_generation_system.txt` owns the general" in guide
    assert "must not duplicate generic" in guide
    assert "must not duplicate generic done_when rules or contain visual" in guide
    assert "must not teach `done_when`" in guide
    assert "reuses the triggering planner window's selected core skill guidance" in guide
    assert "declare a short `semantic_evidence` capability list" in guide
    assert "The structure parser must not receive them" in guide
    assert "does not add fields to `SemanticPlan`" in guide
    assert "must not infer AND/OR semantics" in guide
    assert "rather than exact long prompt sentences" in guide
    assert "condition quality, and grouping" in compact_guide
    assert "`src/agent/transcript.py` owns chronological model intent" in guide
    assert "Context modules must not duplicate that history" in guide
    assert "`runtime.commit_phase_turn` carries the completion assessment" in compact_guide
    assert "provider request through `tools`" in guide
    assert "The runtime uses two loops" in guide
    assert "inner loop may execute immediate read-only tools" in guide
    assert "outer phase loop sends an ordered batch" in compact_guide
    assert "Phase strategy is mutable Runtime state" in guide
    assert "not part of `SemanticPlan`" in guide
    assert "intermediate strategy stage may advance while phase completion remains unknown" in compact_guide
    assert "planning-time refresh with no new window measurement preserves that snapshot" in compact_guide
    assert "explicitly measured zero-motion window replaces it with zeros" in compact_guide
    assert "canonical `ToolSpec` schemas" in compact_guide


def test_agent_boundaries_require_benchmark_generalization():
    guide = Path("AGENT.md").read_text(encoding="utf-8")
    compact_guide = " ".join(guide.split())

    assert "Benchmark Generalization Boundary" in guide
    assert "Benchmarks are diagnostic probes, not product specifications" in guide
    assert "must not directly decide parser, planner, runtime, context, safety, skill, or prompt behavior" in compact_guide
    assert "benchmark case names" in guide
    assert "benchmark classes" in guide
    assert "reference paths" in guide
    assert "SR judge" in guide
    assert "nDTW scores" in guide
    assert "hidden truth" in guide
    assert "Evaluation-only facts must stay in `evaluation/*`" in guide
    assert "restate the issue as a general UAV capability gap" in compact_guide
    assert "target-relative spatial relations" in compact_guide
    assert "around-target progress evidence" in compact_guide
    assert "If a proposed change only helps one benchmark" in compact_guide


def test_model_facing_prompts_and_reusable_skills_do_not_leak_benchmark_repairs():
    model_facing_paths = [
        *Path("src/agent/planning/prompts").glob("*.txt"),
        *Path("src/agent/semantic/prompts").glob("*.txt"),
        *Path("plugins/knowledge").rglob("SKILL.md"),
    ]
    forbidden_terms = [
        "UAV-Flow",
        "nDTW",
        "SR judge",
        "case id",
        "benchmark class",
        "reference_path",
        "reference_path_preprocessed",
        "target_pos",
        "classified_instr",
        "leaderboard",
        "2025-03-30",
    ]

    for path in model_facing_paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in text, f"{path} leaks benchmark-specific term {term!r}"


def test_task_parser_keeps_general_spatial_roles_and_optional_agl_evidence():
    structure = Path(
        "src/agent/semantic/prompts/task_parser_structure_system.txt"
    ).read_text(encoding="utf-8")
    criteria = Path(
        "src/agent/semantic/prompts/task_parser_phase_criteria_system.txt"
    ).read_text(encoding="utf-8")
    compact_criteria = " ".join(criteria.split())

    assert "Image bearing, target-relative position, route relation, and UAV" in structure
    assert "Create an orientation result only when" in structure
    assert "side or route relation alone" in structure
    assert "does not create one" in structure
    assert "configured evidence capabilities only to avoid an inherently" in compact_criteria
    assert "Capability names and anticipated observations must not appear" in compact_criteria
    assert "Do not output extra fields" in criteria
    assert "completion confidence" not in structure + criteria
    assert "single-frame" not in structure + criteria


def test_flyagent_rules_contain_general_visual_action_alignment_without_internal_terms():
    rules = (Path("memory") / "FLYAGENT.md").read_text(encoding="utf-8")

    assert "## Who You Are" in rules
    assert "## Spatial And Action Consistency" in rules
    assert "When a visible cue is on the left" in rules
    assert "Apply the same rule in reverse" in rules
    assert "negative yaw means turn left" not in rules
    assert "positive yaw means turn right" not in rules
    assert "Do not turn away from a visible cue" in rules
    assert "does not prove physical travel" in rules
    assert '"Pass the target on its right"' not in rules
    assert "A semantic plan must not translate" not in rules
    assert "A side-passage phase" not in rules
    assert "move forward, backward, left, right, up, or down in sequence" not in rules

    parser_prompt = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("src/agent/semantic/prompts/task_parser_structure_system.txt"),
            Path("src/agent/semantic/prompts/task_parser_phase_criteria_system.txt"),
        ]
    )
    duplicated_route_rules = ["Who You Are", "How You Work", "Decision Style"]
    for rule in duplicated_route_rules:
        assert rule not in parser_prompt

    forbidden_terms = [
        "UAV-Flow",
        "sunshade",
        "case id",
        "coupled planner",
        "phase judge",
        "model_inputs",
        "context",
        "当前选中的任务技能",
    ]
    for term in forbidden_terms:
        assert term not in rules

    for implementation_term in (
        "PAST",
        "NOW",
        "Depth",
        "AGL",
        "Pose",
        "phase window",
        "overlay",
        "AirSim",
        "UnrealCV",
    ):
        assert implementation_term not in rules

    user_preferences = Path("memory/USER.md").read_text(encoding="utf-8")
    assert "entity's own facing direction" in user_preferences
    assert "observer-relative position" in user_preferences
    assert "route relation" in user_preferences
    assert "positive evidence for an ordinary front-side route or position" in user_preferences
    assert "Only an explicit request" in user_preferences
    assert "to face, orient toward, or turn" in user_preferences
    for implementation_term in ("Runtime", "Backend", "Schema", "PAST", "NOW", "overlay"):
        assert implementation_term not in user_preferences


def test_semantic_plan_update_prompt_is_limited_to_current_phase_patch():
    prompt = (Path("src") / "agent" / "semantic" / "prompts" / "semantic_plan_update_system.txt").read_text(
        encoding="utf-8"
    )

    assert "limited patch" in prompt
    assert '"phase"' in prompt
    assert '"append_recovery_phase"' in prompt
    assert '"goal"' in prompt
    assert '"done_when"' in prompt
    assert '"continue_when"' in prompt
    assert '"metrics"' not in prompt
    assert '"observe"' not in prompt
    assert "Do not rewrite the instruction, task goal, completed phases" in prompt
    assert "potential evidence" in prompt
    assert "Neither is current observed fact" in prompt
    assert "Do not output primitive actions" in prompt
    assert "continue_when remains non-exhaustive clarification" in prompt
    assert "Apply the preceding general completion-condition rules" in prompt
    assert "selected domain knowledge" in prompt
    assert "route-reference passage" not in prompt
    assert "visual facing or orientation" not in prompt
    assert "numeric target-relative standoff" not in prompt
    assert 'Never begin a done_when list item with "or" or "alternatively"' not in prompt
    assert "keep target confirmation and a required target-relative relation in separate items" not in prompt
    assert "Treat the planner's request as a hypothesis" in prompt
    assert "reference path" not in prompt


def test_native_agent_tool_loop_prompt_has_clear_boundaries():
    prompt_dir = Path("src") / "agent" / "planning" / "prompts"
    prompt = (prompt_dir / "agent_tool_loop_system.txt").read_text(encoding="utf-8")

    assert not (prompt_dir / "coupled_action_phase_planner_system.txt").exists()
    assert not (prompt_dir / "coupled_action_planner_system.txt").exists()
    assert not (prompt_dir / "coupled_phase_decision_system.txt").exists()
    assert not (prompt_dir / "coupled_phase_strategy_system.txt").exists()
    assert "chronological transcript" in prompt
    assert "runtime_commit_phase_turn" in prompt
    assert "completion_assessment" in prompt
    assert "supported, unknown, or contradicted" in prompt
    assert "continue_when" in prompt
    assert "one to three" in prompt
    assert "ordered action window" in prompt
    assert "observes once" in prompt
    assert "read-only" in prompt
    assert "observation_id and frame_id" in prompt
    assert "tool_call_id" in prompt
    assert "hidden coordinates" in " ".join(prompt.split())
    assert "previous intent" in prompt
    assert "one to four short ordered maneuver stages" in prompt
    assert "control_actions" not in prompt
    assert "allowed_action_constraints" not in prompt
    assert "response_format" not in prompt


def test_orbit_skill_stays_compact_and_action_oriented():
    skill = (Path("plugins") / "knowledge" / "uav_flight" / "orbit_and_surround" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(skill.lower().split())

    assert "yaw alone is not orbit progress" in normalized
    assert "lateral or tangential translation" in normalized
    assert "brief radial repair without replacing circulation" in normalized
    assert "stop orbit movement" in normalized
    assert "repeated radial corrections" in normalized
    assert "do not enforce a fixed maneuver sequence" in normalized
    assert "Estimate radius" not in skill
    forbidden_internal_terms = [
        "phase_progress_evidence",
        "model_inputs",
        "current_phase",
        "phase_decision",
        "debug metadata",
    ]
    for term in forbidden_internal_terms:
        assert term not in skill


def test_skill_selection_prompt_keeps_compound_locate_and_approach_phases_covered():
    prompt = " ".join(SKILL_SELECTION_SYSTEM_PROMPT.split())

    assert "name, description, summary, and reference manifest" in prompt
    assert "active_instruction, semantic_goal, current_phase, phase_progress_brief" in prompt
    assert "minimal useful set" in prompt
    assert "current semantic phase" in prompt
    assert "Select skills for the current semantic phase only" in prompt
    assert "current_phase.goal, current_phase.done_when, and current_phase.continue_when" in prompt
    assert "active_instruction and semantic_goal only to understand the target, user intent, and final relation" in prompt
    assert "do not preload skills for future phases" in prompt
    assert "primary concern" in prompt
    assert "at most one primary skill" in prompt
    assert "auxiliary skill only when it covers a separate concern" in prompt
    assert "Default to 0-1 skills" in prompt
    assert "Select 2 only when two independent concerns" in prompt
    assert "Select 3 only for clearly complex phases" in prompt
    assert "Do not add search guidance just because a landmark exists" in prompt
    assert "Do not add visual attention guidance just because an object is visible" in prompt
    assert "Do not add obstacle or safety guidance just because a route exists" in prompt
    assert "Treat side/front/back relations as modifiers of the primary task" in prompt
    assert "current phase itself makes the side relation part of route traversal" in prompt
    assert "current phase's main concern" in prompt
    assert "unless traversal is the active current concern" in prompt
    assert "description clearly covers" in prompt
    assert "Request reference files only when the selected skill summary is not enough" in prompt
    assert "Do not request references as insurance" in prompt
    assert "Selecting a skill does not mean its references are needed" in prompt
    assert "absent target evidence, weak candidate evidence, hidden target evidence" in prompt
    assert "explicit search/find/locate/identify-candidate task" in prompt
    assert "target visibility has not yet been confirmed" in prompt
    assert "Do not request a search reference for ordinary face, center, keep-in-view" in prompt
    assert "decide-if-current-observation-is-sufficient" in prompt
    assert "requested_references" in prompt
    assert "important current-phase nuance beyond the primitive itself" in prompt
    assert "controlled vertical motion" not in prompt
    assert "stopping near a surface" not in prompt
    assert "translate N meters" in prompt
    assert "move forward N meters" not in prompt
    assert "selection_scope" not in prompt
    assert "final_goal_is_context_only" not in prompt
    hardcoded_skill_names = [
        "approach_and_standoff",
        "visual_target_task_reasoning",
        "vertical_landing_and_altitude_control",
        "route_traversal_and_passage",
        "orbit_and_surround",
    ]
    for name in hardcoded_skill_names:
        assert name not in prompt
    assert "prefer search_under_uncertainty over" not in prompt


def test_semantic_criteria_skill_selection_has_no_hard_count_or_reference_requests():
    prompt = " ".join(SEMANTIC_CRITERIA_SKILL_SELECTION_PROMPT.split())

    assert "There is no numeric selection limit" in prompt
    assert "smallest genuinely useful set" in prompt
    assert "completion conditions, continuation conditions, or evidence interpretation" in prompt
    assert "phase name, goal, order, or task meaning" in prompt
    assert "requested_references" not in prompt
    assert "at most" not in prompt
    assert "Default to 0-1" not in prompt
