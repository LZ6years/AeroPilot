"""Test prompt files used by model-backed planning."""

from __future__ import annotations

from pathlib import Path

from src.plugins.skills.selection import SKILL_SELECTION_SYSTEM_PROMPT


def test_task_parser_prompt_contains_lightweight_semantic_plan_contract():
    prompt = (Path("src") / "agent" / "semantic" / "prompts" / "task_parser_system.txt").read_text(encoding="utf-8")

    assert '"intent"' in prompt
    assert '"task_goal"' in prompt
    assert '"target_attributes"' in prompt
    assert '"direction":' not in prompt
    assert '"done_when"' in prompt
    assert '"observe"' in prompt
    assert '"continue_when"' in prompt
    assert '"metrics"' in prompt
    assert '"action_bias"' not in prompt
    assert '"forward_distance | right_distance | up_distance | planar_distance | yaw_delta | hover_stability"' in prompt
    assert "uav.turn_relative" in prompt
    assert "env.observe" in prompt
    assert "uav.<tool_name>" in prompt
    assert "distance_m" in prompt
    assert "Do not invent numeric metric targets" in prompt
    assert "Do not invent a distance metric." in prompt
    assert "rearward displacement" in prompt
    assert "backward movement" not in prompt
    assert "Numeric angle role rules" in prompt
    assert "A numeric angle is not automatically a yaw angle" in prompt
    assert "orientation_role" in prompt
    assert "horizontal_displacement_role" in prompt
    assert "vertical_path_role" in prompt
    assert "deviation from the vertical up/down direction" in prompt
    assert "atan(sqrt(vx^2 + vy^2) / abs(vz))" in prompt
    assert "Do not describe this as an angle relative to the horizontal plane" in prompt
    assert "forward_distance = D*cos(A)" in prompt
    assert "right_distance is signed lateral displacement" in prompt
    assert "avoid defaulting to yaw_delta" in prompt
    assert "Seeing or facing a target is useful evidence" in prompt
    assert "completion_criteria" not in prompt
    assert "visual_alignment" not in prompt
    assert "visual_smoke" not in prompt
    assert '"kind"' not in prompt
    assert '"axis"' not in prompt
    assert '"meaning"' not in prompt


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
    assert "转向只能改变观察方向" in rules
    assert "通过或穿越类任务的重点" in rules
    assert "向前、后、左、右、上、下" not in rules

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


def test_coupled_prompt_fragments_have_clear_boundaries():
    prompt_dir = Path("src") / "agent" / "planning" / "prompts"
    phase_prompt = (prompt_dir / "coupled_phase_decision_system.txt").read_text(encoding="utf-8")
    action_prompt = (prompt_dir / "coupled_action_planner_system.txt").read_text(encoding="utf-8")

    assert not (prompt_dir / "coupled_action_phase_planner_system.txt").exists()
    assert "phase_decision" in phase_prompt
    assert "Phase Decision Guide" in phase_prompt
    assert "visual_alignment" in phase_prompt
    assert "route_traversal" in phase_prompt
    assert "spatial_relation" in phase_prompt
    assert "phase_progress_evidence" in phase_prompt
    assert "Metric Authority" in phase_prompt
    assert "all_within_tolerance is true" in phase_prompt
    assert "Tolerance is the completion band" in phase_prompt
    assert "non-metric evidence" in phase_prompt
    assert "vertical flight-path angle" not in phase_prompt
    assert "yaw, heading, vehicle pitch" not in phase_prompt
    assert "do not advance from yaw alone" in phase_prompt
    assert "visible and roughly aligned; or target remains visible after repeated tiny yaw/alternating yaw corrections" in phase_prompt
    assert "slightly left/right of center" in phase_prompt
    assert "evidence to stop micro-adjusting and advance" in phase_prompt
    assert "one short translation or left/right image placement" in phase_prompt
    assert "repeated small yaw corrections should not block advance" not in phase_prompt

    assert "phase_decision" in action_prompt
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
