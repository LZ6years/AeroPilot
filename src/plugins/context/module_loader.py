"""Load ContextModule implementations contributed by plugins."""

from __future__ import annotations

import importlib

from src.memory.context.modules.base import ContextModule
from src.plugins.contracts.context import ContextModuleContribution


class PluginContextModuleLoadError(ValueError):
    """Raised when a plugin context module cannot be loaded."""


def load_plugin_context_module(contribution: ContextModuleContribution) -> ContextModule:
    module = importlib.import_module(contribution.module_path)
    cls = getattr(module, contribution.class_name, None)
    if cls is None:
        raise PluginContextModuleLoadError(
            f"Plugin context module class not found: {contribution.module_path}.{contribution.class_name}"
        )
    instance = cls()
    if not isinstance(instance, ContextModule):
        raise PluginContextModuleLoadError(
            f"Plugin context module must extend ContextModule: {contribution.module_path}.{contribution.class_name}"
        )
    instance.name = contribution.slot
    return instance
