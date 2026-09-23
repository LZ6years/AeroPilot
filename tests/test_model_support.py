"""Generic parser helpers survive removal of the old semantic planning chain."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.agent.model_support import load_required_text, loads_json_object


@pytest.mark.parametrize("module", ["src.memory.context.registry", "src.plugins", "src.agent.planning.task_plan"])
def test_context_plugin_plan_imports_do_not_depend_on_collection_order(module):
    """Cold imports detect cycles masked by the full suite's module cache."""
    result = subprocess.run([sys.executable, "-c", f"import {module}"],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("text", ['{"value": 1}', '```json\n{"value": 1}\n```',
                                  '```\n{"value": 1}\n```', 'Public text {"value": 1} end'])
def test_json_object_reader(text):
    assert loads_json_object(text) == {"value": 1}


@pytest.mark.parametrize("text", ['[]', 'null', '"text"', '{broken}', 'not json'])
def test_json_reader_does_not_repair_non_object_or_invalid_response(text):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        loads_json_object(text)


def test_required_text_preserves_complete_unicode_content(tmp_path):
    path = tmp_path / "rules.md"
    content = "完整规则" * 1000
    path.write_text("\ufeff  " + content + "  ", encoding="utf-8")
    assert load_required_text(path, label="rules") == content
    path.write_text(" \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        load_required_text(path, label="rules", error_type=RuntimeError)
    with pytest.raises(ValueError, match="not found"):
        load_required_text(tmp_path / "missing", label="rules")
