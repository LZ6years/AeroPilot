"""Explicit, caller-owned limits for independent read-only tool execution.

No file is loaded implicitly and no setting changes model requests. Existing
callers stay synchronous; benchmarks may inject an experimental configuration.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ReadOnlyExecutionConfig:
    max_workers: int = 1
    max_capability_calls: int = 2
    capability_limits: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.capability_limits, Mapping):
            raise ValueError("capability_limits must be a mapping")
        for name, value in (("max_workers", self.max_workers),
                            ("max_capability_calls", self.max_capability_calls),
                            *self.capability_limits.items()):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("capability identifiers must be non-empty strings")
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "capability_limits", MappingProxyType(dict(self.capability_limits)))

    def capability_limit(self, key):
        return self.capability_limits.get(key, self.max_capability_calls)

    def to_dict(self):
        return {"max_workers": self.max_workers, "max_capability_calls": self.max_capability_calls,
                "capability_limits": dict(self.capability_limits)}


def load_read_only_execution_config(path: str | Path) -> ReadOnlyExecutionConfig:
    """Require an explicit readable file; never silently substitute defaults."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or set(data) - {"max_workers", "max_capability_calls", "capability_limits"}:
        raise ValueError("Invalid read-only execution configuration")
    if not isinstance(data.get("capability_limits", {}), dict):
        raise ValueError("capability_limits must be an object")
    return ReadOnlyExecutionConfig(**data)
