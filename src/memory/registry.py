"""Context module registry and config loading."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from src.memory.modules import DEFAULT_MODULES
from src.memory.modules.base import ContextModule


class ContextModuleConfigError(ValueError):
    """Raised when context module configuration is invalid."""


class ContextModuleRegistry:
    MODULE_KEYS = ("instruction", "image", "pose", "action", "map")

    def __init__(self, modules: dict[str, ContextModule], disabled: list[str] | None = None) -> None:
        self.modules = dict(modules)
        self.disabled = list(disabled or [])

    @classmethod
    def from_config(cls, path: str | Path = "configs/context_modules.json") -> "ContextModuleRegistry":
        config_path = Path(path)
        if not config_path.exists():
            return cls.default()
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        return cls.from_dict(data)

    @classmethod
    def default(cls) -> "ContextModuleRegistry":
        return cls({key: module_cls() for key, module_cls in DEFAULT_MODULES.items()})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextModuleRegistry":
        module_config = dict(data.get("modules", {}))
        custom_modules = dict(data.get("custom_modules", {}))
        modules: dict[str, ContextModule] = {}
        disabled: list[str] = []
        for key in cls.MODULE_KEYS:
            value = module_config.get(key, "default")
            if value is False:
                disabled.append(key)
                continue
            if value == "default":
                modules[key] = DEFAULT_MODULES[key]()
                continue
            if not isinstance(value, str):
                raise ContextModuleConfigError(f"Invalid module config for {key}: {value!r}")
            modules[key] = _load_custom_module(value, custom_modules)
            if modules[key].name != key:
                modules[key].name = key
        return cls(modules, disabled)

    def build_fragments(self, sources, task_context: dict[str, Any]):
        fragments = []
        for key in self.MODULE_KEYS:
            module = self.modules.get(key)
            if module is None:
                continue
            fragment = module.build(sources, task_context)
            if fragment.name != key:
                raise ContextModuleConfigError(f"Module {key} returned fragment {fragment.name}")
            fragments.append(fragment)
        return fragments


def _load_custom_module(name: str, custom_modules: dict[str, Any]) -> ContextModule:
    spec = custom_modules.get(name)
    if not isinstance(spec, dict):
        raise ContextModuleConfigError(f"Unknown custom context module: {name}")
    module_path = str(spec.get("module_path", "")).strip()
    class_name = str(spec.get("class_name", "")).strip()
    if not module_path or not class_name:
        raise ContextModuleConfigError(f"Custom module {name} requires module_path and class_name")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ContextModuleConfigError(f"Custom module class not found: {module_path}.{class_name}")
    instance = cls()
    if not isinstance(instance, ContextModule):
        raise ContextModuleConfigError(f"Custom module must extend ContextModule: {module_path}.{class_name}")
    return instance
