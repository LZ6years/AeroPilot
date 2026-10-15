"""Model-facing ownership boundaries, independent of editorial wording."""
from pathlib import Path


def test_model_facing_material_does_not_leak_benchmark_or_private_state():
    paths = [
        Path("memory/FLYAGENT.md"), Path("memory/USER.md"),
        *Path("src/agent/planning/prompts").glob("*.txt"),
        *Path("plugins/knowledge").rglob("*.md"),
    ]
    # These are privileged benchmark fields and implementation identifiers,
    # not phrases prescribing how the agent should navigate.
    forbidden = ("UAV-Flow", "AerialVLN", "nDTW", "SR judge", "case id",
                 "benchmark class", "reference_path", "target_pos",
                 "classified_instr", "leaderboard", "phase_progress_evidence",
                 "current_phase", "phase_decision")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{path} exposes {term!r}"


def test_tool_user_contract_does_not_duplicate_individual_tool_declarations():
    from src.tools.registry import default_registry
    tool_user = Path("src/agent/planning/prompts/tool_user_system.txt").read_text(encoding="utf-8")
    for name in default_registry().names():
        assert name not in tool_user


def test_output_language_template_keeps_its_substitution_parameter():
    template = Path("src/agent/planning/prompts/output_language_system.txt").read_text(encoding="utf-8")
    assert "{language_name}" in template
