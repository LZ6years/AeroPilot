"""Single production composition root for Runtime, Plugins, Tools and Context."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.context.compression import (
    HistoryCompressionConfig,
    ModelHistoryCompactor,
)
from src.agent.context.media import MediaHydrator
from src.agent.context.projection import HistoryProjector
from src.agent.context.request_assembler import RequestAssembler
from src.agent.context.window_manager import ContextWindowManager
from src.agent.safety import SafetySupervisor
from src.env_client import EnvClient
from src.llm import DefaultModelClient
from src.memory.context.builder import ModelContextBuilder
from src.memory.context.registry import ContextModuleRegistry
from src.plugins import PluginRegistry
from src.plugins.context import load_plugin_context_module
from src.plugins.skills.dispatcher import SkillCapabilityDispatcher
from src.plugins.skills.knowledge_service import KnowledgeService
from src.agent.config import (
    ModelConfig,
    PlannerConfig,
    ProviderConfig,
    ToolLoopConfig,
)
from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.context.media_store import ModelMediaStore
from src.tools.registry import default_registry
from src.tools.dispatcher import ToolDispatcher
from src.uav.config import UavSafetyConfig


class RuntimeFactory:
    """Discover plugins once and inject the same registry into every subsystem."""

    def __init__(
        self,
        plugins_config_path: str | Path = "configs/plugins.json",
        *,
        context_config_path: str | Path = "configs/context.json",
        plugin_registry: PluginRegistry | None = None,
    ) -> None:
        self.plugins_config_path = Path(plugins_config_path)
        self.config = self._read_config(self.plugins_config_path)
        self.context_config_path = Path(context_config_path)
        context_data = self._read_config(self.context_config_path)
        self.history_compression_config = HistoryCompressionConfig.from_mapping(
            context_data.get("history_projection", {})
        )
        self.plugin_registry = plugin_registry or PluginRegistry.discover(self.config)

    @staticmethod
    def _read_config(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError(f"config {path} must be a JSON object")
        return data

    def context_registry(self, run_dir: str | Path) -> ContextModuleRegistry:
        supplied = {}
        configured = self.config.get("context_modules", {})
        if isinstance(configured, dict):
            for slot, provider in configured.items():
                if provider in {False, "default"}:
                    continue
                contribution = self.plugin_registry.context_module_for(
                    str(slot), str(provider)
                )
                if contribution is None:
                    raise ValueError(
                        f"Unknown plugin context module for {slot}: {provider}"
                    )
                supplied[str(slot)] = load_plugin_context_module(contribution)
        registry = ContextModuleRegistry.from_dict(
            self.config,
            plugin_modules=supplied,
            plugin_warnings=self.plugin_registry.warnings,
        )
        writer = ModelMediaStore(run_dir)
        for module in registry.modules.values():
            bind = getattr(module, "bind_media_writer", None)
            if callable(bind):
                bind(writer)
        return registry

    def context_builder(
        self,
        run_dir: str | Path,
        safety_config: UavSafetyConfig,
    ) -> ModelContextBuilder:
        catalog = [
            {"name": skill.name, "description": skill.description}
            for skill in self.plugin_registry.skill_catalog.skills
            if skill.capability is None
        ]
        return ModelContextBuilder(
            safety_config=safety_config,
            registry=self.context_registry(run_dir),
            knowledge_catalog=catalog,
        )

    def create_task_runtime(
        self,
        env_client: EnvClient,
        recorder: Any,
        *,
        model_config: ModelConfig | None = None,
        provider_config: ProviderConfig | None = None,
        planner_config: PlannerConfig | None = None,
        tool_loop_config: ToolLoopConfig | None = None,
        planner: Any | None = None,
        safety: SafetySupervisor | None = None,
        shutdown_policy: str = "hover",
        stabilization_config: UavStabilizationConfig | None = None,
        event_sink=None,
        runtime_loop_guard_windows: int = 300,
        context_builder: ModelContextBuilder | None = None,
        read_only_execution_config: ReadOnlyExecutionConfig | None = None,
    ):
        # Local import avoids making the TaskRuntime module a composition root.
        from src.runtime.runners.task_runtime import TaskRuntime

        safety = safety or SafetySupervisor()
        builder = context_builder or self.context_builder(
            recorder.run_dir, safety.safety_config
        )
        if model_config is not None and callable(
            getattr(model_config, "provider_config", None)
        ):
            provider_config = provider_config or model_config.provider_config()
            planner_config = planner_config or model_config.planner_config()
            tool_loop_config = tool_loop_config or model_config.tool_loop_config()
        planner = planner or ModelActionPlanner(
            provider_config=provider_config or ProviderConfig(),
            planner_config=planner_config or PlannerConfig(),
            tool_loop_config=tool_loop_config or ToolLoopConfig(),
        )
        tool_loop_config = (
            tool_loop_config
            or getattr(planner, "tool_loop_config", None)
            or ToolLoopConfig()
        )
        tool_registry = default_registry(
            self.plugin_registry, safety.safety_config
        )
        tool_dispatcher = ToolDispatcher(
            env_client,
            registry=tool_registry,
            capability_dispatcher=SkillCapabilityDispatcher(
                self.plugin_registry
            ),
        )
        if isinstance(planner, ModelActionPlanner):
            hydrator = MediaHydrator(base_dir=recorder.run_dir)
            media_policy = {
                "max_request_images": self.history_compression_config.max_request_images,
                "max_tool_evidence_images": (
                    self.history_compression_config.max_tool_evidence_images
                ),
                "keep_recent_observations": (
                    self.history_compression_config.keep_recent_observations
                ),
                "pin_instruction_start": (
                    self.history_compression_config.pin_instruction_start
                ),
                "pin_current": self.history_compression_config.pin_current,
            }
            projector = HistoryProjector(
                builder.registry.history_bindings(),
                assembler=RequestAssembler(hydrator),
                media_policy=media_policy,
            )
            planner_settings = planner.provider_config.to_provider_settings()
            compactor_settings = replace(
                planner_settings,
                timeout_sec=self.history_compression_config.compactor_timeout_sec,
                provider_timeout_retries=(
                    self.history_compression_config.compactor_timeout_retries
                ),
                provider_connection_retries=(
                    self.history_compression_config.compactor_connection_retries
                ),
            )
            compactor = ModelHistoryCompactor(
                DefaultModelClient(compactor_settings),
                summary_max_tokens=(
                    self.history_compression_config.summary_max_tokens
                ),
            )
            planner.history_projector = projector
            planner.context_window_manager = ContextWindowManager(
                projector,
                self.history_compression_config,
                compactor=compactor,
            )
        return TaskRuntime(
            env_client,
            recorder,
            planner=planner,
            safety=safety,
            shutdown_policy=shutdown_policy,
            stabilization_config=stabilization_config,
            event_sink=event_sink,
            runtime_loop_guard_windows=runtime_loop_guard_windows,
            context_builder=builder,
            tool_loop_config=tool_loop_config,
            tool_dispatcher=tool_dispatcher,
            knowledge_service_factory=lambda: KnowledgeService(
                self.plugin_registry.skill_catalog
            ),
            read_only_execution_config=read_only_execution_config,
        )

    def create_smoke_runtime(
        self,
        env_client: EnvClient,
        recorder: Any,
        *,
        observation_modalities: tuple[str, ...] = ("rgb", "depth", "pose"),
    ):
        """Create the built-in smoke Runtime through the same registry graph."""

        from src.runtime.runners.smoke_runtime import SmokeRuntime

        registry = default_registry(self.plugin_registry)
        dispatcher = ToolDispatcher(
            env_client,
            registry=registry,
            capability_dispatcher=SkillCapabilityDispatcher(
                self.plugin_registry
            ),
        )
        return SmokeRuntime(
            env_client,
            recorder,
            dispatcher,
            observation_modalities=observation_modalities,
        )
