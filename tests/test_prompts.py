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

    assert "## Identity And View" in rules
    assert "## Spatial Consistency" in rules
    assert "Turn toward a visible cue" in rules
    assert "negative yaw means turn left" not in rules
    assert "positive yaw means turn right" not in rules
    assert "turning away" in rules
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
    duplicated_route_rules = ["Identity And View", "Operating Cycle", "Spatial Consistency"]
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
    assert "target's own facing direction" in user_preferences
    assert "observer's position around a target" in user_preferences
    assert "route used to approach or pass it" in user_preferences
    assert "Only an" in user_preferences
    assert "request to face, orient or turn" in user_preferences
    assert "configured output language" in user_preferences
    assert "normally use Chinese" not in user_preferences
    for implementation_term in ("Runtime", "Backend", "Schema", "PAST", "NOW", "overlay"):
        assert implementation_term not in user_preferences

    language_policy = Path(
        "src/agent/planning/prompts/output_language_system.txt"
    ).read_text(encoding="utf-8")
    assert "{language_name}" in language_policy
    assert "runtime.update_plan explanation and step" in language_policy
    assert "original\ninstruction" in language_policy



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
    assert "choose a different circulation viewpoint" in normalized
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


def test_visual_target_guidance_preserves_reference_frames_without_action_policy():
    root = Path("plugins/knowledge/uav_flight/visual_target_task_reasoning")
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    search = (root / "references/search_under_uncertainty.md").read_text(encoding="utf-8")

    assert "remains tied to that frame rather than the current heading" in skill
    assert "does not prescribe a later motion sequence" in skill
    assert "not a prescribed\n  motion sequence" in search
    assert "relevant area not yet covered" in search



def test_tool_user_is_generic_and_task_prompt_preserves_roles_and_active_viewpoint_change():
    from src.tools.registry import default_registry
    prompt = Path("src/agent/planning/prompts/agent_tool_loop_system.txt").read_text(encoding="utf-8")
    tool_user = Path("src/agent/planning/prompts/tool_user_system.txt").read_text(encoding="utf-8")
    for name in default_registry().names():
        assert name not in tool_user
        if name != "knowledge.load":
            assert name not in prompt
    assert "runtime.update_plan" not in prompt and "knowledge.load" in prompt
    assert "CURRENT" in prompt and "HISTORY" in prompt
    assert "Use CURRENT RGB to discover and identify candidates" in prompt
    assert "depth capability measures a selected visible candidate" in prompt
    assert "status `collision`" in prompt
    assert "recoverable environment outcome" in prompt
    assert "replan" in prompt
    assert "camera pitch" not in prompt.lower()
    assert "look down" not in prompt.lower()
    assert "missing capability" not in prompt.lower()
    assert "unknown tool" not in prompt.lower()
    assert "Past intent, plans and guidance" in prompt
    assert "Plans and knowledge organize decisions" in prompt
    assert "maximum permissions, not defaults" in prompt
    assert "first choose the next observation position and intended travel distance" in prompt
    assert "actual displacement and visual change" in prompt
    assert "Numeric limits are maximum permissions, not default" in tool_user
    for biased_phrase in (
        "substantial fraction",
        "larger yaw increment",
        "small yaw steps",
        "do not use tiny repeated steps",
    ):
        assert biased_phrase not in (prompt + tool_user).lower()
    assert "required inputs" in tool_user and "execution attributes" in tool_user
    assert "one to three ordered physical" in tool_user
    for retired in ("commit_phase_turn", "phase_strategy", "completion_assessment", "done_when"):
        assert retired not in prompt


def test_runtime_boundaries_document_the_single_loop():
    guide = Path("AGENT.md").read_text(encoding="utf-8")
    for name in ("KnowledgeService", "HistoryProjector", "ToolRegistry", "protocol_context"):
        assert name in guide
    assert "report-reading boundary" in guide


def test_descriptions_keep_capability_boundaries_without_forced_planning_or_early_blocking():
    from src.plugins import PluginRegistry
    from src.tools.registry import default_registry
    registry = default_registry(PluginRegistry.from_config())
    plan = registry.get("runtime.update_plan").description
    assert "optional execution plan" in plan
    assert "long or difficult route" in plan
    assert "multiple ordered stages" in plan
    assert "completed progress" in plan and "next intended objective" in plan
    assert "stable step IDs" in plan and "revision" in plan
    assert "does not execute actions" in plan
    assert "parallel-safe read-only" not in plan
    assert "methods or explanations" in registry.get("knowledge.load").description
    grounding_description = registry.get(
        "skill.visual_grounding.measure_visible_depth"
    ).description
    assert "candidate selected from the CURRENT RGB" in grounding_description
    assert "Never call" not in grounding_description
    assert "Plan status alone is not completion evidence" in registry.get("runtime.report_completed").description
    blocked = registry.get("runtime.report_blocked")
    assert "distinct useful actions or viewpoint changes" in blocked.description
    assert "Temporary non-visibility" in blocked.description
    assert "blocking_condition" not in blocked.input_schema["properties"]
    assert "missing capability" not in blocked.description.lower()
    ground = registry.get("skill.visual_grounding.measure_visible_depth").to_provider_tool()["function"]
    assert "inconsistent_depth_surfaces" in ground["description"]
    assert "no measurement was produced" in ground["description"]
    assert "metric_depth_unavailable" in ground["description"]
    assert "candidate selected from CURRENT RGB" in ground["parameters"]["properties"]["query"]["description"]
    assert "x-flyagent-completion-evidence" not in str(ground)


def test_visual_target_knowledge_keeps_discovery_in_current_visual_evidence():
    skill = Path(
        "plugins/knowledge/uav_flight/visual_target_task_reasoning/SKILL.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(skill.split())

    assert "Discover and identify candidates from CURRENT RGB" in normalized
    assert "surface depth measures a selected visible candidate" in normalized


def test_model_facing_guidance_removes_retired_conservative_pressure():
    paths = [
        Path("memory/FLYAGENT.md"),
        Path("memory/USER.md"),
        *Path("src/agent/planning/prompts").glob("*.txt"),
        *Path("plugins/knowledge/uav_flight").rglob("*.md"),
    ]
    retired = (
        "safe, bounded actions",
        "never assume omitted frames",
        "keep unobserved landmarks unconfirmed",
        "before the first physical action",
        "do not default to either tiny actions",
        "no safe useful option remains",
        "substantial fraction",
        "larger yaw increment",
        "small yaw steps",
        "do not use tiny repeated steps",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in retired:
            assert phrase not in text, f"{path} retains {phrase!r}"
