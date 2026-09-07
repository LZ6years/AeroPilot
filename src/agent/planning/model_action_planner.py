"""OpenAI-compatible native tool-call planner for FlyAgent.

This provider boundary has no JSON-plan fallback. It consumes the canonical
AgentTranscript and returns one AssistantTurn whose tool calls are resolved to
FlyAgent's canonical ToolSpec names.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
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
        # In-memory only until Runtime writes a decision artifact. Image bytes
        # are separated from the JSON request so the durable record can point
        # at content-addressed assets without embedding base64.
        self.last_exchange: dict[str, Any] = {}

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
        request_snapshot, media = _snapshot_request(payload)
        self.last_request = request_snapshot
        self.last_response = {}
        self.last_exchange = {
            "request": copy.deepcopy(request_snapshot),
            "media": media,
            "response": None,
        }
        data = self._request(payload)
        self.last_response = dict(data)
        self.last_exchange["response"] = _public_response_snapshot(data)
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
    """Compatibility wrapper returning the exact request with media references."""

    request, _ = _snapshot_request(payload)
    return request


def _snapshot_request(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate provider image bytes from an otherwise exact request snapshot.

    The returned media entries remain in memory only. Runtime persists each
    digest once, replaces ``data`` with an asset reference, and only then writes
    the decision JSON. Message order and image positions are retained so review
    can reproduce what the provider received without exposing base64.
    """

    media: list[dict[str, Any]] = []

    def clean(value: Any, path: tuple[Any, ...] = ()) -> Any:
        if isinstance(value, str) and value.startswith("data:image/"):
            decoded = _decode_image_data_url(value)
            if decoded is None:
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                media.append(
                    {
                        "sha256": digest,
                        "mime_type": None,
                        "suffix": ".bin",
                        "data": None,
                        "request_path": list(path),
                        "unavailable_reason": "invalid_image_data_url",
                    }
                )
            else:
                content, mime_type, suffix = decoded
                digest = hashlib.sha256(content).hexdigest()
                media.append(
                    {
                        "sha256": digest,
                        "mime_type": mime_type,
                        "suffix": suffix,
                        "data": content,
                        "request_path": list(path),
                    }
                )
            return f"[provider_media sha256={digest}]"
        if isinstance(value, list):
            return [clean(item, (*path, index)) for index, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: clean(item, (*path, key)) for key, item in value.items()}
        return value

    snapshot = clean(payload)
    for item in media:
        item.update(_request_media_label(payload, item.get("request_path")))
    return snapshot, media


def _request_media_label(
    payload: dict[str, Any], request_path: Any
) -> dict[str, Any]:
    """Copy the adjacent model-visible image label into the audit snapshot."""

    if not isinstance(request_path, list) or len(request_path) < 6:
        return {}
    try:
        messages_index = request_path.index("messages")
        message_index = int(request_path[messages_index + 1])
        content_index = int(request_path[messages_index + 3])
        content = payload["messages"][message_index]["content"]
        label = content[content_index - 1]
    except (ValueError, TypeError, KeyError, IndexError):
        return {}
    if not isinstance(label, dict) or label.get("type") != "text":
        return {}
    text = label.get("text")
    if not isinstance(text, str):
        return {}
    lines = text.splitlines()
    if not lines or not lines[0].startswith(("HISTORY ", "CURRENT ")):
        return {}
    first = lines[0].split()
    result: dict[str, Any] = {"timeline_role": first[0], "label": text}
    for field in first[1:]:
        key, separator, value = field.partition("=")
        if separator and key in {"observation_id", "frame_id"}:
            result[key] = value
    for line in lines[1:]:
        for key in ("scene_size", "display_size", "scene_bounds_in_display_pixels"):
            prefix = f"{key}="
            if line.startswith(prefix):
                result[key] = line.removeprefix(prefix)
    return result


def _decode_image_data_url(value: str) -> tuple[bytes, str, str] | None:
    try:
        header, encoded = value.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header.lower():
        return None
    mime_type = header[5:].split(";", 1)[0].lower()
    suffix = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }.get(mime_type)
    if suffix is None:
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    return (data, mime_type, suffix) if data else None


def _public_response_snapshot(response: dict[str, Any]) -> dict[str, Any]:
    """Persist only provider fields that are public protocol evidence."""

    choices = response.get("choices")
    choices = choices if isinstance(choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    raw_message = choice.get("message")
    raw_message = raw_message if isinstance(raw_message, dict) else {}
    message: dict[str, Any] = {
        "role": raw_message.get("role") or "assistant",
        "content": _redact_provider_value(raw_message.get("content")),
    }
    if "tool_calls" in raw_message:
        message["tool_calls"] = _redact_provider_value(raw_message.get("tool_calls"))
    snapshot = {
        "finish_reason": choice.get("finish_reason"),
        "message": message,
    }
    usage = response.get("usage")
    if isinstance(usage, dict):
        snapshot["usage"] = _redact_provider_value(usage)
    return snapshot
