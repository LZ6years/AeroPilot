"""Dispatch registered tool calls through EnvClient."""

from __future__ import annotations

from src.env_client.base import EnvClient
from src.env_client.schemas import ActionResult, Observation
from src.plugins import PluginRegistry
from src.plugins.skills.dispatcher import SkillCapabilityDispatcher
from src.tools.registry import ToolRegistry, default_registry


class ToolDispatcher:
    def __init__(
        self,
        env_client: EnvClient,
        registry: ToolRegistry | None = None,
        plugin_registry: PluginRegistry | None = None,
    ) -> None:
        # Keep the environment client and available tool registry.
        self.env_client = env_client
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()
        self.registry = registry or default_registry(self.plugin_registry)
        self.skill_capabilities = SkillCapabilityDispatcher(self.plugin_registry)

    def call(
        self,
        scene_instance_id: str,
        agent_id: str,
        tool_name: str,
        args: dict,
        runtime_context: dict | None = None,
    ) -> ActionResult:
        # Validate and route a primitive tool action to EnvClient.
        spec = self.registry.get(tool_name)
        self._validate_required_args(spec.required_args, args)
        if spec.kind == "skill_capability":
            return self.skill_capabilities.call(scene_instance_id, agent_id, tool_name, args, runtime_context)
        if spec.kind != "uav_primitive":
            raise ValueError(f"Tool {tool_name} is not an action tool")
        action = tool_name.removeprefix("uav.")
        return self.env_client.execute_action(scene_instance_id, agent_id, action, args)

    def observe(
        self, scene_instance_id: str, agent_id: str, modalities: list[str]
    ) -> Observation:
        # Route raw observation requests to EnvClient.
        self.registry.get("env.observe")
        return self.env_client.get_observation(scene_instance_id, agent_id, modalities)

    @staticmethod
    def _validate_required_args(required_args: tuple[str, ...], args: dict) -> None:
        # Ensure required tool arguments are present before dispatch.
        missing = [name for name in required_args if name not in args]
        if missing:
            raise ValueError(f"Missing required tool args: {', '.join(missing)}")
