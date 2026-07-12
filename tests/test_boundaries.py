"""Test that FlyAgent does not import AirNavigation map/navigation internals."""

from __future__ import annotations

from pathlib import Path


def test_runtime_code_does_not_import_airnavigation_map_controller():
    # Scan source code for forbidden AirNavigation architecture dependencies.
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src").rglob("*.py"))

    assert "MapController" not in source
    assert "UAV_navigater" not in source


def test_runtime_control_contract_keeps_model_context_compact():
    paths = [
        *Path("src/agent/planning/prompts").glob("*.txt"),
        *Path("src/runtime").rglob("*.py"),
        *Path("docs/runtime_control").rglob("*.md"),
    ]
    forbidden_terms = [
        "runtime_control_capabilities",
        '"blocking"',
        '"async"',
        '"use_when"',
        '"avoid_when"',
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in text, f"{path} should not expose verbose runtime control field {term!r}"


def test_runtime_control_does_not_enter_tool_dispatcher_or_skill_selection():
    dispatcher_text = (Path("src") / "tools" / "dispatcher.py").read_text(encoding="utf-8")
    skill_selection_text = (Path("src") / "plugins" / "skills" / "selection.py").read_text(encoding="utf-8")

    assert "control_actions" not in dispatcher_text
    assert "RuntimeControl" not in dispatcher_text
    assert "control_actions" not in skill_selection_text
    assert "RuntimeControl" not in skill_selection_text
