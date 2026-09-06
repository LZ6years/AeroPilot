"""OpenAI-compatible native tool-call planner for FlyAgent.

This provider boundary has no JSON-plan fallback. It consumes the canonical
AgentTranscript and returns one AssistantTurn whose tool calls are resolved to
FlyAgent's canonical ToolSpec names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.provider_http import request_openai_compatible_json
from src.agent.transcript import (
    AgentToolCall,
    AgentTranscript,
    AssistantTurn,
    PublicIntentSource,
)
from src.runtime.configs.model_config import ModelConfig, load_model_config

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


class ModelActionPlannerConfigError(RuntimeError):
    """Raised when the native tool-call planner has unusable configuration."""


class AgentProtocolError(ValueError):
    """Raised when a provider response cannot be represented canonically.

    ``rejected_turn`` contains only the safe response fields needed to audit
    what the provider attempted. It intentionally excludes private reasoning
    fields and arbitrary provider metadata.
    """

    def __init__(
        self,
        message: str,
        *,
        rejected_turn: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.rejected_turn = dict(rejected_turn or {})


class ModelActionPlanner:
    planner_mode = "model"

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        prompt_path: str | None = None,
        *,
        max_repair_attempts: int | None = None,
        repair_planner: Any | None = None,
    ) -> None:
        if max_repair_attempts is not None or repair_planner is not None:
            raise ModelActionPlannerConfigError(
                "The native tool loop repairs errors in transcript; separate repair planners are unsupported."
            )
        self.model_config = model_config or load_model_config()
        configured_prompt = prompt_path or self.model_config.action_planner_prompt_path or None
        self.system_prompt = self._load_system_prompt(configured_prompt)
        self.last_request: dict[str, Any] = {}
        self.last_response: dict[str, Any] = {}

    def plan_turn(
        self,
        transcript: AgentTranscript,
        registry: "ToolRegistry",
        *,
        allowed_tool_names: set[str] | None = None,
    ) -> AssistantTurn:
        """Request exactly one native assistant/tool-call turn."""

        if not self.model_config.api_key:
            raise ModelActionPlannerConfigError(
                "Model action planner requires FLYAGENT_MODEL_API_KEY"
            )
        if self.model_config.provider != "openai_compatible":
            raise ModelActionPlannerConfigError(
                f"Unsupported model provider: {self.model_config.provider}"
            )
        specs = registry.model_specs(allowed_tool_names)
        if not any(spec.name == "runtime.commit_phase_turn" for spec in specs):
            raise ModelActionPlannerConfigError(
                "runtime.commit_phase_turn must be exposed in every mission turn"
            )
        context = transcript.latest_context()
        messages = [
            *_memory_system_messages(context),
            {"role": "system", "content": self.system_prompt},
            *transcript.provider_messages(registry),
        ]
        payload = {
            "model": self.model_config.model_name,
            "messages": messages,
            "tools": [spec.to_provider_tool() for spec in specs],
            "tool_choice": "auto",
            "temperature": 0,
        }
        self.last_request = _redact_request(payload)
        data = self._request(payload)
        self.last_response = dict(data)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            error = "provider response missing choices[0].message"
            raise AgentProtocolError(
                error,
                rejected_turn=_provider_rejection_snapshot(data, error),
            ) from exc
        if not isinstance(message, dict):
            error = "provider choices[0].message must be an object"
            raise AgentProtocolError(
                error,
                rejected_turn=_provider_rejection_snapshot(data, error),
            )
        try:
            return _parse_assistant_turn(message, registry)
        except AgentProtocolError as exc:
            raise AgentProtocolError(
                str(exc),
                rejected_turn=_provider_rejection_snapshot(data, str(exc)),
            ) from exc

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return request_openai_compatible_json(
            self.model_config,
            payload,
            error_prefix="Native tool-call request failed",
        )

    @staticmethod
    def _load_system_prompt(prompt_path: str | None) -> str:
        path = (
            Path(prompt_path)
            if prompt_path
            else Path(__file__).resolve().parent / "prompts" / "agent_tool_loop_system.txt"
        )
        return path.read_text(encoding="utf-8").strip()


def _parse_assistant_turn(message: dict[str, Any], registry: "ToolRegistry") -> AssistantTurn:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise AgentProtocolError("assistant tool_calls must be a list")
    calls: list[AgentToolCall] = []
    seen: set[str] = set()
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise AgentProtocolError("assistant tool call must be an object")
        if raw.get("type") != "function":
            raise AgentProtocolError("assistant tool call type must be function")
        call_id = str(raw.get("id", "")).strip()
        function = raw.get("function")
        if not call_id or call_id in seen or not isinstance(function, dict):
            raise AgentProtocolError("assistant tool calls require unique ids and function objects")
        seen.add(call_id)
        wire_name = str(function.get("name", "")).strip()
        try:
            spec = registry.from_wire_name(wire_name)
        except KeyError as exc:
            raise AgentProtocolError(f"unknown provider tool: {wire_name}") from exc
        try:
            args = json.loads(function.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError) as exc:
            raise AgentProtocolError(
                f"tool {wire_name} arguments are not valid JSON"
            ) from exc
        if not isinstance(args, dict):
            raise AgentProtocolError(f"tool {wire_name} arguments must be an object")
        calls.append(AgentToolCall(call_id, spec.name, args))
    content = message.get("content")
    public_intent = content.strip() if isinstance(content, str) else ""
    if not calls:
        raise AgentProtocolError(
            "mission assistant turn must call a tool; use runtime.commit_phase_turn to finish a phase"
        )
    public_intent_source: PublicIntentSource = "provider_content"
    if not public_intent:
        public_intent = _fallback_public_intent(calls)
        public_intent_source = "tool_arguments_fallback"
    turn_id = str(message.get("id") or f"assistant_{calls[0].tool_call_id}")
    return AssistantTurn(
        turn_id=turn_id,
        public_intent=public_intent,
        tool_calls=calls,
        public_intent_source=public_intent_source,
    )


def _fallback_public_intent(calls: list[AgentToolCall]) -> str:
    """Describe model-authored Tool arguments without inventing reasoning."""

    commit = next(
        (call for call in calls if call.tool_name == "runtime.commit_phase_turn"),
        None,
    )
    if commit is not None:
        action_window = commit.args.get("action_window")
        if isinstance(action_window, dict):
            intent = action_window.get("intent")
            if isinstance(intent, str) and intent.strip():
                return intent.strip()
        phase = commit.args.get("phase")
        phase_label = (
            phase.strip()
            if isinstance(phase, str) and phase.strip()
            else "current"
        )
        return f"Commit phase decision for {phase_label}."

    if len(calls) == 1:
        observation_id = calls[0].args.get("observation_id")
        if isinstance(observation_id, str) and observation_id.strip():
            return f"Request {calls[0].tool_name} for observation {observation_id.strip()}."
    names = ", ".join(call.tool_name for call in calls)
    return f"Request tool call(s): {names}."


def _provider_rejection_snapshot(
    response: dict[str, Any], error: str
) -> dict[str, Any]:
    """Extract auditable wire evidence without persisting hidden reasoning."""

    choices = response.get("choices")
    choices = choices if isinstance(choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    raw_calls = message.get("tool_calls")
    raw_calls = raw_calls if isinstance(raw_calls, list) else []
    tool_calls = []
    for raw in raw_calls:
        raw = raw if isinstance(raw, dict) else {}
        function = raw.get("function")
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments")
        parsed_arguments: Any = arguments
        arguments_status = "missing"
        if arguments is not None:
            try:
                parsed_arguments = (
                    json.loads(arguments)
                    if isinstance(arguments, str)
                    else arguments
                )
                arguments_status = (
                    "parsed" if isinstance(parsed_arguments, dict) else "invalid"
                )
            except (json.JSONDecodeError, TypeError):
                arguments_status = "invalid"
        tool_calls.append(
            {
                "tool_call_id": str(raw.get("id") or ""),
                "type": raw.get("type"),
                "wire_name": str(function.get("name") or ""),
                "arguments": _redact_provider_value(parsed_arguments),
                "arguments_status": arguments_status,
            }
        )
    return {
        "error": error,
        "finish_reason": choice.get("finish_reason"),
        "content_present": bool(content.strip()) if isinstance(content, str) else False,
        "content": _redact_provider_value(content) if isinstance(content, str) else None,
        "tool_calls": tool_calls,
    }


def _redact_provider_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image_data_url]"
    if isinstance(value, list):
        return [_redact_provider_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_provider_value(item) for key, item in value.items()}
    return value


def _memory_system_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    messages = []
    for key, label in (("rules", "FlyAgent identity and operating rules"), ("user_preferences", "User preferences")):
        value = context.get(key)
        if isinstance(value, dict) and value:
            messages.append(
                {
                    "role": "system",
                    "content": f"{label}:\n{json.dumps(value, ensure_ascii=False)}",
                }
            )
    return messages


def _redact_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the exact protocol shape while avoiding duplicate image bytes in debug state."""

    def clean(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("data:image/"):
            return "[image_data_url]"
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    return clean(payload)
