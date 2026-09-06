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
        if spec.kind == "skill_capability":
            self._validate_input_schema(spec.input_schema, args)
            return self.skill_capabilities.call(scene_instance_id, agent_id, tool_name, args, runtime_context)
        if spec.kind != "uav_primitive":
            raise ValueError(f"Tool {tool_name} is not an action tool")
        action = spec.metadata.get("backend_action")
        if not isinstance(action, str) or not action:
            raise ValueError(f"Tool {tool_name} has no backend_action binding")
        return self.env_client.execute_action(scene_instance_id, agent_id, action, args)

    def observe(
        self, scene_instance_id: str, agent_id: str, modalities: list[str]
    ) -> Observation:
        # Route raw observation requests to EnvClient.
        self.registry.get("env.observe")
        return self.env_client.get_observation(scene_instance_id, agent_id, modalities)

    @staticmethod
    def _validate_input_schema(schema: dict, args: dict) -> None:
        """Enforce the common required/additional-properties contract.

        Full numeric and enum validation is performed before a model-requested
        batch is accepted. Dispatcher keeps this final defensive boundary for
        direct callers and lifecycle tools.
        """

        required_args = schema.get("required", []) if isinstance(schema, dict) else []
        missing = [name for name in required_args if name not in args]
        if missing:
            raise ValueError(f"Missing required tool args: {', '.join(missing)}")
        if isinstance(schema, dict) and schema.get("additionalProperties") is False:
            properties = schema.get("properties", {})
            allowed = set(properties) if isinstance(properties, dict) else set()
            extra = sorted(set(args) - allowed)
            if extra:
                raise ValueError(f"Unsupported tool args: {', '.join(extra)}")
