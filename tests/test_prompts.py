"""Test prompt files used by model-backed planning."""

from __future__ import annotations

from pathlib import Path

from src.plugins.skills.semantic_criteria import SEMANTIC_CRITERIA_SKILL_SELECTION_PROMPT
from src.plugins.skills.selection import SKILL_SELECTION_SYSTEM_PROMPT


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
    assert "phase-decision prompt owns only completion-evidence" in guide
    assert "classification. The phase-strategy prompt owns" in guide
    assert "ordered maneuver stages and active stage" in compact_guide
    assert "action prompt owns tool choice, signed argument use" in compact_guide
    assert "Phase strategy is mutable Runtime state" in guide
    assert "not part of `SemanticPlan`" in guide
    assert "intermediate strategy stage may advance while phase completion remains unknown" in compact_guide
    assert "planning-time refresh with no new window measurement preserves that snapshot" in compact_guide
    assert "explicitly measured zero-motion window replaces it with zeros" in compact_guide
    assert "canonical UAV tool contract" in guide


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


def test_coupled_prompt_fragments_have_clear_boundaries():
    prompt_dir = Path("src") / "agent" / "planning" / "prompts"
    phase_prompt = (prompt_dir / "coupled_phase_decision_system.txt").read_text(encoding="utf-8")
    strategy_prompt = (prompt_dir / "coupled_phase_strategy_system.txt").read_text(encoding="utf-8")
    action_prompt = (prompt_dir / "coupled_action_planner_system.txt").read_text(encoding="utf-8")

    assert not (prompt_dir / "coupled_action_phase_planner_system.txt").exists()
    assert "Output only completion_assessment" in phase_prompt
    assert "completion_assessment" in phase_prompt
    assert len(phase_prompt) <= 3000
    assert "control_actions" not in phase_prompt
    assert "Runtime Control Boundary" not in phase_prompt
    assert "Action Boundary" not in phase_prompt
    assert "Ask for semantic_plan_update" not in phase_prompt
    assert "Ask for ask_user" not in phase_prompt
    assert "phase_completion_evaluation is not_measurable" not in phase_prompt
    assert "phase_execution_summary.window_count is already high" not in phase_prompt
    assert "repeated same-kind actions have produced meaningful phase_progress_evidence" not in phase_prompt
    assert "endless small corrective actions" not in phase_prompt
    assert "Judgment:" in phase_prompt
    assert "Judge every supplied completion condition" in phase_prompt
    assert "Missing ideal measurements" in phase_prompt
    assert "unavailable sensors are not" in phase_prompt
    assert "explicit either/or inside one done_when item" not in phase_prompt
    assert "Continuation descriptions are" in phase_prompt
    assert "not additional gates" in phase_prompt
    assert "Do not use unknown to request a better view" in phase_prompt
    assert "Lack of a centroid, exact pixel, direct measurement, or tolerance" in phase_prompt
    assert "no calibrated measurement" not in phase_prompt
    assert "target-distance condition" not in phase_prompt
    assert "familiar target scale" not in phase_prompt
    assert "Phase Decision Guide" not in phase_prompt
    assert "visual_alignment" not in phase_prompt
    assert "route_traversal" not in phase_prompt
    assert "spatial_relation" not in phase_prompt
    assert "circulation_or_orbit" not in phase_prompt
    assert "hidden coordinates" in phase_prompt
    assert "phase_progress_evidence" not in phase_prompt
    assert "phase_execution_summary" not in phase_prompt
    assert "action_trend" not in phase_prompt
    assert "yaw_only_so_far" not in phase_prompt
    assert "phase_metrics" not in phase_prompt
    assert "phase_completion_evaluation" not in phase_prompt
    assert "vertical flight-path angle" not in phase_prompt
    assert "yaw, heading, vehicle pitch" not in phase_prompt
    assert "exactly one object for every" in phase_prompt
    assert "status is exactly supported, unknown, or contradicted" in phase_prompt
    assert "evidence briefly states" in phase_prompt
    assert "If continue_when is still true or evidence is uncertain" not in phase_prompt

    assert "current-phase maneuver strategy planner" in strategy_prompt
    assert "only stages, active_stage, and reason" in strategy_prompt
    assert "1-4 short ordered stages" in strategy_prompt
    assert "maneuver intent plus observable terminal state" in strategy_prompt
    assert "Never put UAV tool names or argument fields" in strategy_prompt
    assert "independently reassess the previous active stage" in strategy_prompt
    assert "previous_reason only explains why the prior window" in strategy_prompt
    assert "It is never evidence" in strategy_prompt
    assert "Phase completion may remain unknown while an intermediate stage is satisfied" in strategy_prompt
    assert "keep is not the default update" in strategy_prompt
    assert "ideal geometric proof is not required" in strategy_prompt
    assert "last-window UAV motion" in strategy_prompt
    assert "indefinitely repeating the same action" in strategy_prompt
    assert "Keep the ordered stages unchanged when the route remains valid" in strategy_prompt
    assert "Rewrite only the active and future stages" in strategy_prompt
    assert "target is no longer visible" in strategy_prompt
    assert "Do not copy previous_reason as proof" in strategy_prompt
    assert "phase_strategy does not determine phase completion" in strategy_prompt
    assert "sunshade" not in strategy_prompt
    assert "trees" not in strategy_prompt
    assert "control_actions" not in strategy_prompt

    assert "Do not output phase_decision" in action_prompt
    assert "completion_assessment" in action_prompt
    assert "a request for about 4.5 m with a credible target" in action_prompt
    assert "is supported, not unknown" in action_prompt
    assert "control_actions" in action_prompt
    assert "Runtime Control Actions" in action_prompt
    assert "Never put ask_user or semantic_plan_update in steps" in action_prompt
    assert "runtime_control.available" in action_prompt
    assert "semantic_plan_update may be requested alongside 1-3 safe UAV tool steps" in action_prompt
    assert "semantic_plan_update payload must include reason" in action_prompt
    assert "goal, done_when, or continue_when" in action_prompt
    assert "steps must be []" in action_prompt
    assert "use exactly one step" in action_prompt
    assert "kind=\"observe\"" in action_prompt
    assert "uav.hover" in action_prompt
    assert "uav.turn_relative" in action_prompt
    assert "uav.move_body" in action_prompt
    assert "action_schemas[*].argument_semantics" in action_prompt
    assert "args, action_window.expected_effect, and reason" in action_prompt
    assert "must not knowingly move the state away from the phase goal" in action_prompt
    assert "instead of guessing a signed movement" in action_prompt
    assert "Positive moves forward" not in action_prompt
    assert "negative turns left" not in action_prompt
    assert '"direction":' not in action_prompt
    assert "agent_messages" in action_prompt
    assert '"phase_strategy"' in action_prompt
    assert "updated phase_strategy.stages[phase_strategy.active_stage]" in action_prompt
    assert "do not repeat an earlier stage after the strategy has advanced" in action_prompt
    assert "Phase Decision Guide" not in action_prompt
    assert "route progress evidence" not in action_prompt
    coupled_prompts = phase_prompt + strategy_prompt + action_prompt
    assert "phase_transition_judge_system" not in coupled_prompts
    assert '"blocking"' not in coupled_prompts
    assert '"async"' not in coupled_prompts
    assert '"use_when"' not in coupled_prompts
    assert '"avoid_when"' not in coupled_prompts


def test_phase_strategy_prompt_separates_stage_progress_from_phase_completion():
    prompt_dir = Path("src") / "agent" / "planning" / "prompts"
    strategy_prompt = (prompt_dir / "coupled_phase_strategy_system.txt").read_text(encoding="utf-8")
    action_prompt = (prompt_dir / "coupled_action_planner_system.txt").read_text(encoding="utf-8")

    assert "independently reassess the previous active stage" in strategy_prompt
    assert "previous_reason only explains why the prior window" in strategy_prompt
    assert "Phase completion may remain unknown while an intermediate stage is satisfied" in strategy_prompt
    assert "keep is not the default update" in strategy_prompt
    assert "ideal geometric proof is not required" in strategy_prompt
    assert "last-window UAV motion" in strategy_prompt
    assert "phase-start-relative cumulative motion" in strategy_prompt
    assert "indefinitely repeating the same action" in strategy_prompt
    assert "target is no longer visible" in strategy_prompt
    assert "Do not copy previous_reason as proof" in strategy_prompt
    assert "Never put UAV tool names or argument fields" in strategy_prompt
    assert "sunshade" not in strategy_prompt
    assert "trees" not in strategy_prompt
    assert "updated phase_strategy.stages[phase_strategy.active_stage]" in action_prompt
    assert "do not repeat an earlier stage after the strategy has advanced" in action_prompt


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
