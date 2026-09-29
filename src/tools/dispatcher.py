"""Dispatch registered tool calls through EnvClient."""

from __future__ import annotations

from typing import Any

from src.env_client.base import EnvClient
from src.env_client.schemas import ActionResult, Observation
from src.tools.registry import ToolRegistry
from src.tools.schemas import ReadExecutionHints
from src.tools.validation import validate_tool_args


class ToolDispatcher:
    def read_execution_hints(self, tool_name: str, args: dict) -> ReadExecutionHints:
        """Resolve internal resource identity from the registered tool kind."""
        spec = self.registry.get(tool_name)
        if spec.runtime.effect != "read_only":
            raise ValueError("only read-only tools have read execution hints")
        if spec.kind == "knowledge":
            return ReadExecutionHints(serial_key="knowledge:" + str(args.get("name", "")))
        if spec.kind == "skill_capability":
            return ReadExecutionHints("capability", capability_key=spec.name)
        # Internal environment reads share a client and are never submitted as
        # parallel model calls under their current single-call contracts.
        return ReadExecutionHints(serial_key="environment")

    def __init__(
        self,
        env_client: EnvClient,
        registry: ToolRegistry,
        capability_dispatcher: Any | None = None,
    ) -> None:
        # Keep the environment client and available tool registry.
        self.env_client = env_client
        self.registry = registry
        self.skill_capabilities = capability_dispatcher

    def call(
        self,
        scene_instance_id: str,
        agent_id: str,
        tool_name: str,
        args: dict,
        runtime_context: dict | None = None,
        *,
        knowledge_service: Any | None = None,
    ) -> ActionResult:
        # Validate and route a primitive tool action to EnvClient.
        spec = self.registry.get(tool_name)
        if spec.kind == "knowledge":
            self._validate_input_schema(spec.input_schema, args)
            if knowledge_service is None:
                raise ValueError("knowledge service is not bound to an instruction")
            data = knowledge_service.load(**args)
            return ActionResult(
                ok=True, action=tool_name,
                message="Knowledge guidance loaded; this is not environmental evidence.",
                data=data,
            )
        if spec.kind == "skill_capability":
            self._validate_input_schema(spec.input_schema, args)
            if self.skill_capabilities is None:
                raise ValueError("skill capability dispatcher is not configured")
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
        """Apply the complete native ToolSpec contract at the final boundary."""

        validate_tool_args(args, schema)
