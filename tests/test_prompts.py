"""Test prompt files used by model-backed planning."""

from __future__ import annotations

from pathlib import Path

from src.plugins.skills.semantic_criteria import SEMANTIC_CRITERIA_SKILL_SELECTION_PROMPT
from src.plugins.skills.selection import SKILL_SELECTION_SYSTEM_PROMPT


def test_task_parser_prompts_keep_structure_and_criteria_contracts_separate():
    prompt_dir = Path("src") / "agent" / "semantic" / "prompts"
    structure = (prompt_dir / "task_parser_structure_system.txt").read_text(encoding="utf-8")
    criteria = (prompt_dir / "task_parser_phase_criteria_system.txt").read_text(encoding="utf-8")
    structure_shape = structure.split("Boundaries:", 1)[0]
    criteria_shape = criteria.split("Immutable structure:", 1)[0]

    assert not (prompt_dir / "task_parser_system.txt").exists()
    assert '"intent"' in structure_shape
    assert '"task_goal"' in structure_shape
    assert '"target_attributes"' in structure_shape
    assert '"done_when"' not in structure_shape
    assert '"observe"' not in structure_shape
    assert '"continue_when"' not in structure_shape
    assert '"metrics"' not in structure_shape
    assert "phase count" in structure
    assert "Phase structure comes from user intent" in structure
    assert "does not decide execution feasibility or downgrade the request for safety" in structure
    assert "Cover the complete requested outcome" in structure
    assert "Every explicit terminal result" in structure
    assert "phase name and goal" in structure
    assert '"prepare for X"' in structure
    assert "controlled descent and a landed/stable result" in structure
    assert "check every requested outcome verb" in structure
    assert "Never add exact or exactly" in structure
    assert "Do not rewrite them as centered in the image or camera" in structure
    assert '"done_when"' in criteria_shape
    assert '"observe"' not in criteria_shape
    assert '"continue_when"' in criteria_shape
    assert '"metrics"' in criteria_shape
    assert '"intent"' not in criteria_shape
    assert '"task_goal"' not in criteria_shape
    assert '"forward_distance | right_distance | up_distance | planar_distance | yaw_delta | hover_stability"' in criteria
    assert "target-relative" in criteria
    assert "not phase-start planar_distance" in criteria
    assert "user instruction is the only numeric source" in criteria
    assert "current metric schema has no target-distance metric" in criteria
    assert "Never add yaw_delta=0" in criteria
    assert "Criteria must cover the immutable phase goal through completion" in criteria
    assert "terminal result has already occurred" in criteria
    assert "controlled descent or altitude/surface progress" in criteria
    assert "exception never permits invented numbers" in criteria
    assert '"Move to a position N meters from a target"' in criteria
    assert "Do not infer yaw_delta=360" in criteria
    assert "scan every number" in criteria
    assert 'Do not add parenthetical examples, "e.g." expansions' in criteria
    assert "invented lower/upper bounds" in criteria
    assert "Every done_when item is a required core condition" in criteria
    assert "independently assessable required facts in separate done_when items" in criteria
    assert "target confirmation, a requested spatial relation, and stabilization" in criteria
    assert "alternatives that can each support the same core condition remain together" in criteria
    assert "inside the same item" in criteria
    assert 'Never begin a done_when list item with "or" or "alternatively"' in criteria
    assert "minimum sufficient completed state" in criteria
    assert "identifiable target cue is clearly inside the forward view" in criteria
    assert "Modest lateral offset, incidental scene occlusion" in criteria
    assert "Do not require exact centering, full-body or full-object visibility" in criteria
    assert "State the terminal task relation directly" in criteria
    assert "continue_when gives non-exhaustive clarification" in criteria
    assert "must not treat a state that can occur on a valid done_when path as unfinished by itself" in criteria
    assert "route reference being absent is not sufficient continuation evidence" in criteria
    assert "two alternative sufficient paths" in criteria
    assert "Do not require both paths, reacquisition, or continued reference visibility" in criteria
    assert 'never write bare reference absence, "reference not visible", or "reference not yet visible"' in criteria
    assert "preferred visual manifestation or extra confirmation" in criteria
    assert "numeric target-relative standoff without a deterministic target-distance metric" in criteria
    assert "visual scale and available perspective, depth, scene, or observation-change cues" in criteria
    assert "no clear evidence that the target is substantially nearer or farther" in criteria
    assert "Do not phrase it as a demand to verify, measure, or confirm the exact number" in criteria
    assert "Do not add hover, hold, or stabilization" in criteria
    assert "uav.turn_relative" in structure
    assert "uav.move_body" in criteria
    assert '"action_bias"' not in structure + criteria
    assert '"direction":' not in structure + criteria
    assert '"uncertainties"' not in structure + criteria


def test_legacy_action_and_phase_judge_prompts_removed():
    prompt_dir = Path("src") / "agent" / "planning" / "prompts"

    assert not (prompt_dir / "action_planner_system.txt").exists()
    assert not (prompt_dir / "phase_transition_judge_system.txt").exists()


def test_agent_boundaries_document_prompt_and_skill_writing_rules():
    guide = Path("AGENT.md").read_text(encoding="utf-8")

    assert "Prompt And Skill Writing Rules" in guide
    assert "Reusable skills are flight-domain knowledge" in guide
    assert "`current_phase`" in guide
    assert "`phase_decision`" in guide
    assert "`phase_progress_evidence`" in guide
    assert "`model_inputs`" in guide
    assert "Do not restore model-facing `direction` shorthand" in guide
    assert "`relation` plus measurable metrics" in guide
    assert "hidden benchmark truth" in guide
    assert "`memory/FLYAGENT.md` is the shared Agent identity" in guide
    assert "both model planning and model semantic task parsing" in guide
    assert "must not duplicate" in guide


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
        "reference path",
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


def test_flyagent_rules_contain_general_visual_action_alignment_without_internal_terms():
    rules = (Path("memory") / "FLYAGENT.md").read_text(encoding="utf-8")

    assert "## Visual Direction And Action Consistency" in rules
    assert "A cue on the left side of the image calls for a leftward alignment adjustment" in rules
    assert "a cue on the right calls for a rightward alignment adjustment" in rules
    assert "negative yaw means turn left" not in rules
    assert "positive yaw means turn right" not in rules
    assert "make a small turn toward the same side" in rules
    assert "A turn toward the opposite side" in rules
    assert "not centering the currently visible cue" in rules
    assert "it must not also say that turning right will center it" in rules
    assert "Basic flight knowledge always applies" in rules
    assert "## Spatial And Route Relations" in rules
    assert "Image-side position can change from yaw alone" in rules
    assert "cannot by itself prove that the UAV has reached a physical side of a target" in rules
    assert '"Pass the target on its right" means the UAV route lies on the target\'s right' in rules
    assert "it does not mean the target should appear on the right side of the image" in rules
    assert "the target will usually appear on the left" in rules
    assert 'A semantic plan must not translate "the target is on the left or right of the image"' in rules
    assert "A search phase only confirms the target and a plausible route" in rules
    assert "A side-passage phase must rely on translation or route progress" in rules
    assert "Yaw alone can support search or observation" in rules
    assert "cannot prove reaching a target side or completing passage" in rules
    assert "move forward, backward, left, right, up, or down in sequence" not in rules

    parser_prompt = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("src/agent/semantic/prompts/task_parser_structure_system.txt"),
            Path("src/agent/semantic/prompts/task_parser_phase_criteria_system.txt"),
        ]
    )
    duplicated_route_rules = [
        "Image-side position can change from yaw alone",
        'A semantic plan must not translate "the target is on the left or right of the image"',
        "A search phase only confirms the target and a plausible route",
        "A side-passage phase must rely on translation or route progress",
        "Yaw alone can support search or observation",
    ]
    for rule in duplicated_route_rules:
        assert rule not in parser_prompt

    forbidden_terms = [
        "UAV-Flow",
        "sunshade",
        "person",
        "case id",
        "coupled planner",
        "phase judge",
        "model_inputs",
        "context",
        "当前选中的任务技能",
    ]
    for term in forbidden_terms:
        assert term not in rules


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

    assert "limited semantic patch" in prompt
    assert '"phase"' in prompt
    assert '"append_recovery_phase"' in prompt
    assert '"goal"' in prompt
    assert '"done_when"' in prompt
    assert '"continue_when"' in prompt
    assert '"metrics"' in prompt
    assert '"observe"' not in prompt
    assert "Do not rewrite task_goal, instruction, full phases" in prompt
    assert "Do not output UAV tool names" in prompt
    assert "one exact camera manifestation" in prompt
    assert "Treat continue_when as non-exhaustive clarification" in prompt
    assert "allowed by a valid done_when path" in prompt
    assert "Every done_when list item is a required core condition" in prompt
    assert "independently assessable required facts in separate done_when items" in prompt
    assert "Do not strengthen an approximate target-distance condition" in prompt
    assert "keep target confirmation separate from a directly judgeable qualitative distance condition" in prompt
    assert "inside the same item with explicit either/or wording" in prompt
    assert 'Never begin a done_when list item with "or" or "alternatively"' in prompt
    assert "minimum sufficient completed state" in prompt
    assert "independently sufficient paths" in prompt
    assert "Do not require both paths, reacquisition, or continued visibility" in prompt
    assert "Never use bare route-reference absence as continue_when" in prompt
    assert "preferred visual manifestation or extra confirmation" in prompt
    assert "uav.turn_relative" in prompt
    assert "uav.move_body" in prompt
    assert "evaluation sample" in prompt
    assert "Treat planner_request as a hypothesis, not authority" in prompt
    assert "planner_request.payload.focus" in prompt
    assert "reference path" not in prompt


def test_coupled_prompt_fragments_have_clear_boundaries():
    prompt_dir = Path("src") / "agent" / "planning" / "prompts"
    phase_prompt = (prompt_dir / "coupled_phase_decision_system.txt").read_text(encoding="utf-8")
    action_prompt = (prompt_dir / "coupled_action_planner_system.txt").read_text(encoding="utf-8")

    assert not (prompt_dir / "coupled_action_phase_planner_system.txt").exists()
    assert "Do not output phase_decision" in phase_prompt
    assert "completion_assessment" in phase_prompt
    assert "control_actions" in phase_prompt
    assert "Runtime Control Boundary" in phase_prompt
    assert "Runtime control meanings and request timing are provided by runtime_control context" in phase_prompt
    assert "Runtime control requests belong in control_actions only and never in steps" in phase_prompt
    assert "Ask for semantic_plan_update" not in phase_prompt
    assert "Ask for ask_user" not in phase_prompt
    assert "phase_completion_evaluation is not_measurable" not in phase_prompt
    assert "phase_execution_summary.window_count is already high" not in phase_prompt
    assert "repeated same-kind actions have produced meaningful phase_progress_evidence" not in phase_prompt
    assert "endless small corrective actions" not in phase_prompt
    assert "Completion Contract" in phase_prompt
    assert "Every done_when list item is a required core condition" in phase_prompt
    assert "explicit either/or inside one done_when item" in phase_prompt
    assert "continue_when is non-exhaustive clarification" in phase_prompt
    assert "It is not a completion gate" in phase_prompt
    assert "Residual possibility, optional optimization" in phase_prompt
    assert 'Never use "no calibrated measurement", "no exact depth"' in phase_prompt
    assert "familiar target scale, apparent size, perspective, ground contact" in phase_prompt
    assert "mark contradicted when the available cues clearly indicate substantially nearer or farther" in phase_prompt
    assert "Reserve unknown for cases where the target or usable range cues are genuinely unavailable" in phase_prompt
    assert "Phase Decision Guide" not in phase_prompt
    assert "visual_alignment" not in phase_prompt
    assert "route_traversal" not in phase_prompt
    assert "spatial_relation" not in phase_prompt
    assert "circulation_or_orbit" not in phase_prompt
    assert "fragments.action.content.motion_history_summary" in phase_prompt
    assert "phase_start_relative_uav_motion summarizes the whole phase" in phase_prompt
    assert "Never claim that no translation occurred" in phase_prompt
    assert "recent_execution is only a short tail" in phase_prompt
    assert "phase_progress_evidence" not in phase_prompt
    assert "phase_execution_summary" not in phase_prompt
    assert "action_trend" not in phase_prompt
    assert "yaw_only_so_far" not in phase_prompt
    assert "Metric Authority" in phase_prompt
    assert "all_within_tolerance result" in phase_prompt
    assert "Tolerance is the completion band" in phase_prompt
    assert "absence of a deterministic metric is not missing evidence" in phase_prompt
    assert "vertical flight-path angle" not in phase_prompt
    assert "yaw, heading, vehicle pitch" not in phase_prompt
    assert "exactly one object for each current_phase.done_when item" in phase_prompt
    assert "status must be supported, unknown, or contradicted" in phase_prompt
    assert "evidence is explanatory only" in phase_prompt
    assert "cannot change supported back to unknown" in phase_prompt
    assert "do not invent an additional requirement" in phase_prompt
    assert "If continue_when is still true or evidence is uncertain" not in phase_prompt

    assert "Do not output phase_decision" in action_prompt
    assert "completion_assessment" in action_prompt
    assert "control_actions" in action_prompt
    assert "Runtime Control Actions" in action_prompt
    assert "Never put ask_user or semantic_plan_update in steps" in action_prompt
    assert "runtime_control.available" in action_prompt
    assert "semantic_plan_update may be requested alongside 1-3 safe UAV tool steps" in action_prompt
    assert "semantic_plan_update payload must include reason" in action_prompt
    assert "goal, done_when, continue_when, or metrics" in action_prompt
    assert "steps must be []" in action_prompt
    assert "use exactly one step" in action_prompt
    assert "kind=\"observe\"" in action_prompt
    assert "uav.hover" in action_prompt
    assert "uav.turn_relative" in action_prompt
    assert "uav.move_body" in action_prompt
    assert "direction" not in action_prompt
    assert "agent_messages" in action_prompt
    assert "Phase Decision Guide" not in action_prompt
    assert "route progress evidence" not in action_prompt
    assert "phase_transition_judge_system" not in phase_prompt + action_prompt
    assert '"blocking"' not in phase_prompt + action_prompt
    assert '"async"' not in phase_prompt + action_prompt
    assert '"use_when"' not in phase_prompt + action_prompt
    assert '"avoid_when"' not in phase_prompt + action_prompt


def test_orbit_skill_stays_compact_and_action_oriented():
    skill = (Path("plugins") / "knowledge" / "uav_flight" / "orbit_and_surround" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(skill.lower().split())

    assert "yaw alone is not orbit progress" in normalized
    assert "lateral or tangential translation" in normalized
    assert "brief radial repair and resume circulation" in normalized
    assert "stop orbit movement" in normalized
    assert "repeated radial corrections" in normalized
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
    assert "current_phase.goal, current_phase.done_when, current_phase.continue_when, and current_phase.metrics" in prompt
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
    assert "completion conditions, continuation conditions, evidence sources, or valid metrics" in prompt
    assert "phase name, goal, order, or task meaning" in prompt
    assert "requested_references" not in prompt
    assert "at most" not in prompt
    assert "Default to 0-1" not in prompt
