"""Test that FlyAgent does not import AirNavigation map/navigation internals."""

from __future__ import annotations

import ast
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


def test_runtime_control_does_not_enter_tool_dispatcher_or_knowledge_reader():
    dispatcher_text = (Path("src") / "tools" / "dispatcher.py").read_text(encoding="utf-8")
    knowledge_text = Path("src/plugins/skills/knowledge_service.py").read_text(encoding="utf-8")

    assert "control_actions" not in dispatcher_text
    assert "RuntimeControl" not in dispatcher_text
    assert "control_actions" not in knowledge_text
    assert "RuntimeControl" not in knowledge_text
    assert not Path("src/plugins/skills/selection.py").exists()


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


def test_lower_layers_do_not_import_runtime():
    roots = (
        Path("src/agent"),
        Path("src/memory"),
        Path("src/plugins"),
        Path("src/tools"),
        Path("src/uav"),
        Path("src/sim_server"),
    )
    violations = {
        str(path): sorted(
            name
            for name in _imports(path)
            if name == "src.runtime" or name.startswith("src.runtime.")
        )
        for root in roots
        for path in root.rglob("*.py")
        if any(
            name == "src.runtime" or name.startswith("src.runtime.")
            for name in _imports(path)
        )
    }
    assert violations == {}


def test_context_registry_is_plugin_agnostic_and_old_config_paths_are_removed():
    imports = _imports(Path("src/memory/context/registry.py"))
    assert not any(
        name == "src.plugins" or name.startswith("src.plugins.")
        for name in imports
    )
    assert not Path("src/runtime/configs/model_config.py").exists()
    assert not Path("src/runtime/configs/action_config.py").exists()
    assert not Path("src/runtime/configs/safety_config.py").exists()


def test_tool_registry_does_not_import_or_discover_plugins():
    for path in Path("src/tools").rglob("*.py"):
        imports = _imports(path)
        source = path.read_text(encoding="utf-8-sig")
        assert not any(
            name == "src.plugins" or name.startswith("src.plugins.")
            for name in imports
        ), path
        assert "PluginRegistry.discover" not in source
        assert "PluginRegistry.from_config" not in source


def test_memory_does_not_import_agent_implementations():
    violations = {
        str(path): sorted(
            name
            for name in _imports(path)
            if name == "src.agent" or name.startswith("src.agent.")
        )
        for path in Path("src/memory").rglob("*.py")
        if any(
            name == "src.agent" or name.startswith("src.agent.")
            for name in _imports(path)
        )
    }
    assert violations == {}


def test_task_runtime_does_not_import_plugins():
    imports = _imports(Path("src/runtime/runners/task_runtime.py"))
    assert not any(
        name == "src.plugins" or name.startswith("src.plugins.")
        for name in imports
    )


def test_plugin_discovery_has_one_production_composition_root():
    occurrences = []
    for path in Path("src").rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        if "PluginRegistry.discover" in source or "PluginRegistry.from_config" in source:
            occurrences.append(str(path).replace("\\", "/"))
    assert occurrences == ["src/runtime/factory.py"]


def test_default_tests_do_not_import_manual_scripts():
    violations = {
        str(path): sorted(
            name
            for name in _imports(path)
            if name == "tests.manual" or name.startswith("tests.manual.")
        )
        for path in Path("tests").glob("test_*.py")
        if any(
            name == "tests.manual" or name.startswith("tests.manual.")
            for name in _imports(path)
        )
    }
    assert violations == {}
