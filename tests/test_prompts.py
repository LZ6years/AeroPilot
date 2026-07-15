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
    assert '"done_when"' in criteria_shape
    assert '"observe"' in criteria_shape
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
    assert "roughly aligned" in criteria
    assert "exact center" in criteria
    assert 'Bare "centered in the camera frame" is too strict' in criteria
    assert "A small left/right offset alone is not a reason to continue" in criteria
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

    assert "视觉方向与动作一致性" in rules
    assert "画面左侧对应左向调整" in rules
    assert "画面右侧对应右向调整" in rules
    assert "负 yaw 表示向左转" in rules
    assert "正 yaw 表示向右转" in rules
    assert "向同侧做小幅转向" in rules
    assert "相反方向" in rules
    assert "为了把当前侧的可见参考居中" in rules
    assert "不能同时说向右转是为了让它居中" in rules
    assert "基础飞行常识始终适用" in rules
    assert "空间关系与路线关系" in rules
    assert "画面中的左右位置可以仅因转向而改变" in rules
    assert "不能单独证明无人机已经到达目标的某一物理侧" in rules
    assert "从目标右侧经过”表示无人机航线位于目标右侧" in rules
    assert "不表示目标应在画面右侧" in rules
    assert "目标通常位于画面左侧" in rules
    assert "语义计划不得把“目标在画面左侧或右侧”解释成“无人机在目标右侧或左侧”" in rules
    assert "搜索阶段只确认目标和可行路线" in rules
    assert "侧向经过阶段必须依赖平移或路线进展" in rules
    assert "只转向只能用于搜索或观察" in rules
    assert "不能作为到达目标某侧或完成经过的证据" in rules
    assert "向前、后、左、右、上、下" not in rules

    parser_prompt = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            Path("src/agent/semantic/prompts/task_parser_structure_system.txt"),
            Path("src/agent/semantic/prompts/task_parser_phase_criteria_system.txt"),
        ]
    )
    duplicated_route_rules = [
        "画面中的左右位置可以仅因转向而改变",
        "语义计划不得把“目标在画面左侧或右侧”解释成“无人机在目标右侧或左侧”",
        "搜索阶段只确认目标和可行路线",
        "侧向经过阶段必须依赖平移或路线进展",
        "只转向只能用于搜索或观察",
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
    assert "Do not rewrite task_goal, instruction, full phases" in prompt
    assert "Do not output UAV tool names" in prompt
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
    assert "phase_decision" in phase_prompt
    assert "control_actions" in phase_prompt
    assert "Runtime Control Boundary" in phase_prompt
    assert "Runtime control meanings and request timing are provided by the runtime_control context" in phase_prompt
    assert "Runtime control requests belong in control_actions only and never in steps" in phase_prompt
    assert "Ask for semantic_plan_update" not in phase_prompt
    assert "Ask for ask_user" not in phase_prompt
    assert "phase_completion_evaluation is not_measurable" not in phase_prompt
    assert "phase_execution_summary.window_count is already high" not in phase_prompt
    assert "repeated same-kind actions have produced meaningful phase_progress_evidence" not in phase_prompt
    assert "endless small corrective actions" not in phase_prompt
    assert "Phase Decision Guide" in phase_prompt
    assert "visual_alignment" in phase_prompt
    assert "route_traversal" in phase_prompt
    assert "spatial_relation" in phase_prompt
    assert "circulation_or_orbit" in phase_prompt
    assert "phase_progress_evidence" in phase_prompt
    assert "Metric Authority" in phase_prompt
    assert "all_within_tolerance is true" in phase_prompt
    assert "Tolerance is the completion band" in phase_prompt
    assert "non-metric evidence" in phase_prompt
    assert "vertical flight-path angle" not in phase_prompt
    assert "yaw, heading, vehicle pitch" not in phase_prompt
    assert "do not advance from yaw alone" in phase_prompt
    assert "meaningful lateral or tangential progress" in phase_prompt
    assert "repeating radius/standoff corrections" in phase_prompt
    assert "visible and roughly aligned; or target remains visible after repeated tiny yaw/alternating yaw corrections" in phase_prompt
    assert "slightly left/right of center" in phase_prompt
    assert "evidence to stop micro-adjusting and advance" in phase_prompt
    assert "one short translation or left/right image placement" in phase_prompt
    assert "repeated small yaw corrections should not block advance" not in phase_prompt

    assert "phase_decision" in action_prompt
    assert "control_actions" in action_prompt
    assert "Runtime Control Actions" in action_prompt
    assert "Never put ask_user or semantic_plan_update in steps" in action_prompt
    assert "runtime_control.available" in action_prompt
    assert "semantic_plan_update may be requested in control_actions alongside the normal 1-3 safe UAV tool steps" in action_prompt
    assert "semantic_plan_update payload must include reason" in action_prompt
    assert "goal, done_when, continue_when, or metrics" in action_prompt
    assert "continue_current_phase | advance_to_next_phase | complete_task" in action_prompt
    assert "steps must be []" in action_prompt
    assert "Never set it true with multiple steps" in action_prompt
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

    assert "yaw alone is not orbit progress" in skill
    assert "lateral or tangential shift" in skill
    assert "shift segment -> yaw re-align -> next shift segment" in skill
    assert "brief support work" in skill
    assert "continue orbit movement instead of" in skill
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
