"""OpenAI-compatible native tool-call planner for FlyAgent.

This provider boundary has no JSON-plan fallback. It consumes the canonical
AgentTranscript and returns one AssistantTurn whose tool calls are resolved to
FlyAgent's canonical ToolSpec names.
"""

from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn
from src.runtime.configs.model_config import ModelConfig, load_model_config

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


class ModelActionPlannerConfigError(RuntimeError):
    """Raised when the native tool-call planner has unusable configuration."""


class AgentProtocolError(ValueError):
    """Raised when a provider response cannot be represented canonically."""


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
            raise AgentProtocolError("provider response missing choices[0].message") from exc
        if not isinstance(message, dict):
            raise AgentProtocolError("provider choices[0].message must be an object")
        return _parse_assistant_turn(message, registry)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.model_config.chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.model_config.timeout_sec
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Native tool-call request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Native tool-call request failed: {exc}") from exc

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
    if calls and not public_intent:
        raise AgentProtocolError(
            "assistant must include a concise public intent with tool calls"
        )
    if not calls:
        raise AgentProtocolError(
            "mission assistant turn must call a tool; use runtime.commit_phase_turn to finish a phase"
        )
    turn_id = str(message.get("id") or f"assistant_{calls[0].tool_call_id}")
    return AssistantTurn(turn_id=turn_id, public_intent=public_intent, tool_calls=calls)


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
