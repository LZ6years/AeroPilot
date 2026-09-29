"""Load instruction-stable identity and preference documents once."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_IDENTITY_PATH = Path("memory") / "FLYAGENT.md"
DEFAULT_USER_PREFERENCES_PATH = Path("memory") / "USER.md"


def read_context_document(path: str | Path) -> dict[str, Any]:
    document = Path(path)
    if not document.exists():
        return {}
    text = document.read_text(encoding="utf-8-sig").strip()
    return {"text": text, "source": str(document)} if text else {}


class ContextDocuments:
    """Immutable-at-construction stable model documents for one Runtime."""

    def __init__(
        self,
        *,
        identity: dict[str, Any] | None = None,
        user_preferences: dict[str, Any] | None = None,
        identity_path: str | Path = DEFAULT_IDENTITY_PATH,
        user_preferences_path: str | Path = DEFAULT_USER_PREFERENCES_PATH,
    ) -> None:
        self.identity = {
            **read_context_document(identity_path),
            **dict(identity or {}),
        }
        self.user_preferences = {
            **read_context_document(user_preferences_path),
            **dict(user_preferences or {}),
        }
