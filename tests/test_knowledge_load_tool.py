"""Knowledge reads are full, versioned, bounded to declared files and model-free."""

import pytest


def test_concurrent_delivery_is_atomic_and_new_instruction_is_independent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    service = KnowledgeService(catalog(tmp_path))
    barrier = threading.Barrier(4)
    def load(_):
        barrier.wait(timeout=3)
        return service.load("navigation", ["reference.md"])
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(load, range(4)))
    assert sum(result["body_included"] for result in results) == 1
    assert sum(len(result["references"]) for result in results) == 1
    assert service.loaded_index()[0]["reference_paths"] == ["reference.md"]
    assert KnowledgeService(catalog(tmp_path)).load("navigation")["body_included"]


def test_flight_catalog_has_consistent_selection_fields_without_schema_expansion():
    from src.plugins import PluginRegistry
    service = KnowledgeService(PluginRegistry.from_config().skill_catalog)
    entries = service.catalog()
    assert len(entries) == 6
    for item in entries:
        assert set(item) == {"name", "description"}
        assert item["description"].strip()
        body = service.load(item["name"])["body"]
        assert "## Completion And Stop" in body
    names = {item["name"] for item in entries}
    assert {"basic_flight_control", "abnormal_situation_recovery"} <= names
    assert "vertical_landing_and_altitude_control" not in names

from src.plugins.contracts.skill import Skill
from src.plugins.skills.catalog import SkillCatalog
from src.plugins.skills.knowledge_service import KnowledgeService
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import ToolRegistry
from src.tools.builtins.knowledge import knowledge_tool


def catalog(tmp_path, reference=None):
    root = tmp_path / "entry"
    root.mkdir(exist_ok=True)
    ref = reference or root / "reference.md"
    if reference is None:
        ref.write_text("full reference " * 1000, encoding="utf-8")
    return SkillCatalog([Skill(
        plugin_id="knowledge.test", name="navigation", description="Navigation guidance",
        summary="short", skill_dir=root, skill_path=root / "SKILL.md",
        body="knowledge正文" * 1000, reference_paths=(ref,),
    )])


def test_complete_content_incremental_references_and_snapshot(tmp_path):
    entries = catalog(tmp_path)
    service = KnowledgeService(entries)
    entries.skills[0].reference_paths[0].write_text("changed later", encoding="utf-8")
    first = service.load("navigation")
    assert len(first["body"]) > 2400 and first["references"] == []
    second = service.load("navigation", ["reference.md"])
    assert "body" not in second and not second["body_included"]
    assert second["version"] == first["version"]
    assert len(second["references"][0]["text"]) > 1600
    assert "changed later" not in second["references"][0]["text"]
    third = service.load("navigation", ["reference.md"])
    assert third["references"] == [] and "body" not in third
    third["reference_manifest"].clear()
    assert service.load("navigation")["reference_manifest"]
    assert service.loaded_index()[0]["reference_paths"] == ["reference.md"]
    assert set(service.catalog()[0]) == {"name", "description"}


@pytest.mark.parametrize("paths", [["../secret"], ["reference.md", "reference.md"], "reference.md", [1]])
def test_invalid_request_is_atomic(tmp_path, paths):
    service = KnowledgeService(catalog(tmp_path))
    with pytest.raises(ValueError):
        service.load("navigation", paths)
    assert service.loaded_index() == []
    assert service.load("navigation")["body_included"]


def test_undeclared_entry_and_outside_reference(tmp_path):
    service = KnowledgeService(catalog(tmp_path))
    with pytest.raises(ValueError):
        service.load("disabled")
    outside = tmp_path / "outside.md"
    outside.write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        KnowledgeService(catalog(tmp_path, outside))


def test_duplicate_names_fail_and_new_instruction_reloads(tmp_path):
    entries = catalog(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        KnowledgeService(SkillCatalog(entries.skills * 2))
    first = KnowledgeService(entries)
    first.load("navigation")
    assert KnowledgeService(entries).load("navigation")["body_included"]


def test_knowledge_dispatch_never_calls_environment_or_capability(tmp_path):
    class NoExternalCalls:
        def __getattr__(self, name):
            pytest.fail(f"knowledge must not access external service: {name}")

    registry = ToolRegistry()
    registry.register(knowledge_tool())
    # Inject dependencies without loading default plugins or external backends.
    dispatcher = ToolDispatcher.__new__(ToolDispatcher)
    dispatcher.registry = registry
    dispatcher.env_client = NoExternalCalls()
    dispatcher.skill_capabilities = NoExternalCalls()
    knowledge = KnowledgeService(catalog(tmp_path))
    result = dispatcher.call(
        "scene", "uav", "knowledge.load", {"name": "navigation"},
        knowledge_service=knowledge,
    )
    assert result.ok and result.data["body_included"]
    assert "not environmental evidence" in result.message
    again = dispatcher.call(
        "scene", "uav", "knowledge.load", {"name": "navigation"},
        knowledge_service=knowledge,
    )
    assert again.ok and "body" not in again.data
    with pytest.raises(ValueError):
        dispatcher.call(
            "scene", "uav", "knowledge.load",
            {"name": "navigation", "extra": 1},
            knowledge_service=knowledge,
        )
    with pytest.raises(ValueError, match="not bound"):
        dispatcher.call("scene", "uav", "knowledge.load", {"name": "navigation"})
