"""Experiment-only plan exposure and frozen prompt overlays.

Production retains optional planning. Off-mode edits only the new experiment
snapshot and uses existing ToolRuntimePolicy.model_exposed filtering, so provider
declarations and Runtime's allowed calls agree without a navigation special case.
Every replacement is bounded and audited; unknown prompt text fails preparation.
"""
from pathlib import Path


PLAN_MODES = ("optional", "off")


def freeze_plan_mode(root, hashes, mode, digest):
    """Return provenance for exactly the off-mode snapshot modifications."""
    if mode not in PLAN_MODES:
        raise ValueError(f"Unknown plan mode: {mode}")
    if mode == "optional":
        return []
    replacements = {
        "src/tools/builtins/planning.py": [
            ('ToolRuntimePolicy("runtime_control", "immediate", "sidecar", True)',
             'ToolRuntimePolicy("runtime_control", "immediate", "sidecar", False)')],
        "src/agent/planning/prompts/agent_tool_loop_system.txt": [
            ("Past intent, plans and guidance", "Past intent and guidance"),
            ("## Planning And Knowledge", "## Knowledge"),
            ("Reuse guidance already loaded. Plans and knowledge organize decisions; observations",
             "Reuse guidance already loaded. Knowledge organizes decisions; observations"),
            ("CURRENT TASK STATE is the latest saved plan and knowledge index. Plan statuses\n"
             "record your progress judgment, not environmental proof or mission completion.\n"
             "When a plan is useful, keep its recorded stage aligned with observed progress.\n",
             "CURRENT TASK STATE contains the latest loaded knowledge index.\n")],
        "src/agent/planning/prompts/task_progress_reminder.txt": [
            ("If a plan exists, check whether its recorded stage matches observed progress.\n"
             "Update only for changed progress or approach; unchanged progress needs no update.\n", "")],
        "src/agent/planning/prompts/tool_user_system.txt": [
            ("control, exactly one physical call, independent `parallel_safe` reads,\n"
             "or one plan update together with such reads. Physical and terminal groups never mix\n",
             "control, exactly one physical call, or independent `parallel_safe` reads.\n"
             "Physical and terminal groups never mix\n"),
            ("not undo successful reads or a committed plan.", "not undo successful reads.")],
    }
    # Stage all text before writing, so an unrecognized source leaves no partly
    # transformed snapshot. This is never a historical-source restore routine.
    staged = []
    for relative, edits in replacements.items():
        path = Path(root) / "source" / relative
        text = path.read_text(encoding="utf-8")
        before = digest(path)
        for old, new in edits:
            if text.count(old) != 1:
                raise ValueError(f"Off-mode overlay expected one match: {relative}: {old!r}")
            text = text.replace(old, new)
        staged.append((relative, path, before, text))
    receipts = []
    for relative, path, before, text in staged:
        path.write_text(text, encoding="utf-8")
        hashes[relative] = digest(path)
        receipts.append({"source": relative, "before_sha256": before,
                         "after_sha256": hashes[relative], "reason": "experiment_plan_off"})
    return receipts


def admission_passed(record, mode="optional"):
    """Legacy optional admission stays 15 calls; off-mode requires three native calls."""
    if mode not in PLAN_MODES:
        raise ValueError(f"Unknown plan mode: {mode}")
    receipts = record.get("receipts", [])
    expected = 3 if mode == "off" else 15
    valid = (record.get("status") == "passed" and record.get("groups_completed") == 3
             and len(receipts) == expected
             and all(r.get("status") == "ok" and r.get("attempt") == 1 for r in receipts))
    if mode == "off":
        return valid and record.get("plan_probe_status") == "not_applicable" and record.get("native_probe_status") == "passed"
    return valid and record.get("plan_probe_status") == "passed"
