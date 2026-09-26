"""Prompt boundary tests for the sole native tool loop and reusable knowledge."""
from pathlib import Path
import json

def test_legacy_action_and_phase_judge_prompts_removed():
    prompt_dir = Path("src") / "agent" / "planning" / "prompts"

    assert not (prompt_dir / "action_planner_system.txt").exists()
    assert not (prompt_dir / "phase_transition_judge_system.txt").exists()



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
            Path("src/agent/planning/prompts/agent_tool_loop_system.txt"),
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



def test_tool_user_is_generic_and_task_prompt_preserves_evidence_roles():
    from src.tools.registry import default_registry
    prompt = Path("src/agent/planning/prompts/agent_tool_loop_system.txt").read_text(encoding="utf-8")
    tool_user = Path("src/agent/planning/prompts/tool_user_system.txt").read_text(encoding="utf-8")
    for name in default_registry().names():
        assert name not in tool_user
        if name not in {"runtime.update_plan", "knowledge.load"}:
            assert name not in prompt
    assert "runtime.update_plan" in prompt and "knowledge.load" in prompt
    assert "CURRENT" in prompt and "HISTORY" in prompt
    assert "required" in tool_user and "result_timing" in tool_user
    assert "one to three ordered physical" in tool_user
    for retired in ("commit_phase_turn", "phase_strategy", "completion_assessment", "done_when"):
        assert retired not in prompt


def test_runtime_boundaries_document_the_single_loop():
    guide = Path("AGENT.md").read_text(encoding="utf-8")
    for name in ("KnowledgeService", "HistoryProjector", "ToolRegistry", "protocol_context"):
        assert name in guide
    assert "report-reading boundary" in guide


def test_descriptions_keep_capability_boundaries_and_explicit_planning_preference():
    from src.tools.registry import default_registry
    registry = default_registry()
    plan = registry.get("runtime.update_plan").description
    assert "before the first physical action" in plan
    assert "simple single-step task can act directly" in plan
    assert "stable step IDs" in plan and "revision" in plan
    assert "methods or explanations" in registry.get("knowledge.load").description
    assert "empty detection does not prove absence" in registry.get("skill.visual_grounding.locate").description
    assert "Plan status alone is not completion evidence" in registry.get("runtime.report_completed").description
    assert "cannot continue safely or meaningfully" in registry.get("runtime.report_blocked").description
    ground = registry.get("skill.visual_grounding.locate").to_provider_tool()["function"]
    assert "inconsistent_depth_surfaces" in ground["description"]
    assert "not_requested" in ground["description"]
    assert "directly observed appearance" in ground["parameters"]["properties"]["query"]["description"]
    assert "x-flyagent-completion-evidence" not in str(ground)
