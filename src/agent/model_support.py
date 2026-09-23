"""Small model-output and required-text helpers; no planning policy or IO calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_AGENT_RULES_PATH = Path(__file__).resolve().parents[2] / "memory" / "FLYAGENT.md"
DEFAULT_USER_PREFERENCES_PATH = Path(__file__).resolve().parents[2] / "memory" / "USER.md"


def load_required_text(path: Path, *, label: str, error_type: type[Exception] = ValueError) -> str:
    if not path.exists():
        raise error_type(f"{label} file not found: {path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise error_type(f"{label} file is empty: {path}")
    return text


def loads_json_object(content: str) -> dict[str, Any]:
    """Preserve the existing fenced/prose JSON reader without semantic repair."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value
