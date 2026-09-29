"""Pure registry for already-constructed Context Module instances.

Plugin discovery belongs to ``runtime.factory``. Keeping this registry free of
Plugins and Runtime prevents the former Memory -> Plugins -> Memory cycle.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from src.memory.context.contracts import (
    ContextModuleResult,
    FragmentAudit,
    MemorySourceBundle,
    TaskContextInput,
)
from src.memory.context.modules import DEFAULT_MODULES
from src.memory.context.modules.base import ContextModule


class ContextModuleConfigError(ValueError):
    """Raised when context module configuration is invalid."""


class ContextModuleRegistry:
    DEFAULT_MODULE_ORDER = ("skills", "image", "pose", "motion", "map")
    MAX_SEMANTIC_EVIDENCE_ITEMS = 8
    MAX_SEMANTIC_EVIDENCE_CHARS = 160

    def __init__(
        self,
        modules: dict[str, ContextModule],
        *,
        disabled: list[str] | None = None,
        module_order: list[str] | None = None,
        history_policies: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.modules = dict(modules)
        self.disabled = list(disabled or [])
        self.module_order = list(module_order or modules.keys())
        self.history_policies = copy.deepcopy(history_policies or {})
        for slot, policy in self.history_policies.items():
            if slot not in self.modules or slot in self.disabled:
                raise ContextModuleConfigError(
                    f"History policy requires enabled module: {slot}"
                )
            if not isinstance(policy, dict):
                raise ContextModuleConfigError(
                    f"History policy must be an object: {slot}"
                )
            try:
                self.modules[slot].validate_history_policy(copy.deepcopy(policy))
            except ValueError as exc:
                raise ContextModuleConfigError(
                    f"Invalid history policy for {slot}: {exc}"
                ) from exc

    @classmethod
    def default(
        cls, *, plugin_warnings: list[str] | None = None
    ) -> "ContextModuleRegistry":
        modules = {
            key: (
                module_cls(warnings=plugin_warnings)
                if key == "skills"
                else module_cls()
            )
            for key, module_cls in DEFAULT_MODULES.items()
        }
        return cls(modules, module_order=list(cls.DEFAULT_MODULE_ORDER))

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        plugin_modules: dict[str, ContextModule] | None = None,
        plugin_warnings: list[str] | None = None,
    ) -> "ContextModuleRegistry":
        """Bind configured slots to instances supplied by the composition root."""

        configured = data.get("context_modules", {})
        module_config = dict(configured) if isinstance(configured, dict) else {}
        if not module_config:
            module_config = {key: "default" for key in cls.DEFAULT_MODULE_ORDER}
        modules: dict[str, ContextModule] = {}
        disabled: list[str] = []
        order: list[str] = []
        supplied = dict(plugin_modules or {})
        for raw_slot, provider in module_config.items():
            slot = str(raw_slot)
            if provider is False:
                disabled.append(slot)
                continue
            order.append(slot)
            if provider == "default":
                module_cls = DEFAULT_MODULES.get(slot)
                if module_cls is None:
                    raise ContextModuleConfigError(
                        f"No default context module for slot: {slot}"
                    )
                modules[slot] = (
                    module_cls(warnings=plugin_warnings)
                    if slot == "skills"
                    else module_cls()
                )
            elif slot in supplied:
                modules[slot] = supplied[slot]
                modules[slot].name = slot
            else:
                raise ContextModuleConfigError(
                    f"Composition root did not supply plugin module {slot}: {provider}"
                )

        options = data.get("context_options", {})
        if not isinstance(options, dict):
            raise ContextModuleConfigError("context_options must be a JSON object")
        for slot, values in options.items():
            if slot not in modules or not isinstance(values, dict):
                raise ContextModuleConfigError(
                    f"Options require an enabled module and object: {slot}"
                )
            try:
                modules[slot].configure(copy.deepcopy(values))
            except ValueError as exc:
                raise ContextModuleConfigError(
                    f"Invalid options for {slot}: {exc}"
                ) from exc

        history = data.get("context_history", {})
        if not isinstance(history, dict):
            raise ContextModuleConfigError("context_history must be a JSON object")
        return cls(
            modules,
            disabled=disabled,
            module_order=order,
            history_policies=history,
        )

    def history_bindings(
        self,
    ) -> dict[str, tuple[ContextModule, dict[str, Any]]]:
        return {
            slot: (self.modules[slot], copy.deepcopy(policy))
            for slot, policy in self.history_policies.items()
        }

    def build_fragments(
        self, sources: MemorySourceBundle, stage_context: TaskContextInput
    ) -> tuple[ContextModuleResult, ...]:
        results: list[ContextModuleResult] = []
        for slot in self.module_order:
            if slot in self.disabled:
                continue
            module = self.modules.get(slot)
            if module is None:
                continue
            started = time.perf_counter()
            result = module.build(sources, stage_context)
            if not isinstance(result, ContextModuleResult):
                raise ContextModuleConfigError(
                    f"Module {slot} must return ContextModuleResult"
                )
            if result.fragment.name != slot or result.audit.name != slot:
                raise ContextModuleConfigError(
                    f"Module {slot} returned mismatched fragment/audit names"
                )
            audit = FragmentAudit(
                name=result.audit.name,
                source=result.audit.source,
                confidence=result.audit.confidence,
                warnings=result.audit.warnings,
                artifact_refs=result.audit.artifact_refs,
                metadata=result.audit.metadata,
                build_duration_ms=(time.perf_counter() - started) * 1000.0,
            )
            results.append(ContextModuleResult(result.fragment, audit))
        return tuple(results)

    def describe_modules(self) -> dict[str, Any]:
        return {
            slot: self.modules[slot].describe_output()
            for slot in self.module_order
            if slot in self.modules and slot not in self.disabled
        }

    def semantic_evidence_capabilities(self) -> list[str]:
        capabilities: list[str] = []
        seen: set[str] = set()
        for slot in self.module_order:
            if slot in self.disabled:
                continue
            module = self.modules.get(slot)
            if module is None:
                continue
            try:
                declared = module.describe_output().get("semantic_evidence")
            except Exception:
                continue
            if not isinstance(declared, list):
                continue
            for item in declared:
                if not isinstance(item, str):
                    continue
                normalized = " ".join(item.split())
                if not normalized or normalized in seen:
                    continue
                if len(normalized) > self.MAX_SEMANTIC_EVIDENCE_CHARS:
                    normalized = (
                        normalized[: self.MAX_SEMANTIC_EVIDENCE_CHARS - 3].rstrip()
                        + "..."
                    )
                seen.add(normalized)
                capabilities.append(normalized)
                if len(capabilities) >= self.MAX_SEMANTIC_EVIDENCE_ITEMS:
                    return capabilities
        return capabilities
