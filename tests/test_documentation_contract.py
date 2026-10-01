"""Keep the maintained documentation aligned with the production API."""

from __future__ import annotations

import re
from pathlib import Path


CURRENT_DOCUMENTS = (
    Path("README.md"),
    Path("docs/README.md"),
    Path("docs/architecture.md"),
    Path("docs/context_v3.md"),
    Path("docs/plugin_system.md"),
    Path("docs/testing.md"),
    Path("docs/directory_structure.md"),
    Path("docs/run_outputs.md"),
    Path("docs/task_example.md"),
    Path("docs/runtime_control/README.md"),
    Path("docs/custom_modules/README.md"),
    Path("docs/simulator_control/README.md"),
    Path("docs/evaluation/aerial_vln/README.md"),
    Path("docs/evaluation/travel_uav/README.md"),
    Path("docs/evaluation/uav_flow/README.md"),
)


def test_current_document_links_resolve():
    missing: list[str] = []
    for document in CURRENT_DOCUMENTS:
        text = document.read_text(encoding="utf-8-sig")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            path_text = target.split("#", 1)[0]
            resolved = (document.parent / path_text).resolve()
            if not resolved.exists():
                missing.append(f"{document}: {target}")
    assert missing == []


def test_current_documents_do_not_publish_retired_context_api():
    forbidden = (
        "ContextComposer",
        "ObservationContextBuilder",
        "ModelInputAssembler",
        "runtime.commit_phase_turn",
        "semantic_phases",
        "Context V2",
        "context_v2.md",
        "model_content",
        "model_inputs",
        "plan_updated",
        "knowledge_loaded",
    )
    violations = {
        str(document): [term for term in forbidden if term in document.read_text(encoding="utf-8-sig")]
        for document in CURRENT_DOCUMENTS
    }
    assert {path: terms for path, terms in violations.items() if terms} == {}


def test_archived_markdown_is_labeled_historical():
    unlabeled = []
    for path in Path("docs/archive").rglob("*.md"):
        text = path.read_text(encoding="utf-8-sig")
        if "Status: Historical" not in text[:300]:
            unlabeled.append(str(path))
    assert unlabeled == []
