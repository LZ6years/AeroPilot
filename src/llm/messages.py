"""Optional ordered message-block builder, with no prompt or history policy."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class MessageBuilder:
    """Keys are caller-local handles, never provider fields or predefined recipes."""

    def __init__(self) -> None:
        self._blocks: list[tuple[str | None, list[dict[str, Any]]]] = []

    def _index(self, key: str) -> int:
        for index, (block_key, _) in enumerate(self._blocks):
            if block_key == key:
                return index
        raise KeyError(key)

    def extend(self, messages: list[dict[str, Any]], *, key: str | None = None) -> MessageBuilder:
        if key is not None:
            if not isinstance(key, str) or not key:
                raise ValueError("message key must be a non-empty string")
            if any(existing == key for existing, _ in self._blocks):
                raise ValueError(f"duplicate message key: {key}")
        self._blocks.append((key, deepcopy(messages)))
        return self

    def add_system(self, text: str, *, key: str | None = None) -> MessageBuilder:
        return self.extend([{"role": "system", "content": text}], key=key)

    def add_user(self, content: Any, *, key: str | None = None) -> MessageBuilder:
        return self.extend([{"role": "user", "content": content}], key=key)

    def get(self, key: str) -> list[dict[str, Any]]:
        return deepcopy(self._blocks[self._index(key)][1])

    def replace(self, key: str, messages: list[dict[str, Any]]) -> MessageBuilder:
        self._blocks[self._index(key)] = (key, deepcopy(messages))
        return self

    def remove(self, key: str) -> MessageBuilder:
        del self._blocks[self._index(key)]
        return self

    def build(self) -> list[dict[str, Any]]:
        return deepcopy([message for _, messages in self._blocks for message in messages])
