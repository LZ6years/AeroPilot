"""OpenAI-compatible Chat Completions wire encoding, not agent response semantics."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from src.llm.contracts import ModelRequest, ProviderSettings
from src.llm.audit import redact_provider_value


def encode_request(settings: ProviderSettings, request: ModelRequest) -> dict[str, Any]:
    if settings.provider != "openai_compatible":
        raise ValueError(f"Unsupported model provider: {settings.provider}")
    payload = {"model": settings.model_name, "messages": deepcopy(request.messages)}
    if request.tools is not None:
        payload["tools"] = deepcopy(request.tools)
    payload.update(deepcopy(request.parameters))
    if settings.stream:
        # Streaming is an explicit provider transport choice. Do not add
        # ``stream: false`` to otherwise frozen non-stream requests.
        payload["stream"] = True
    return payload


def decode_response(body: bytes, *, stream: bool = False) -> dict[str, Any]:
    """Decode one Chat Completions response into the canonical JSON envelope.

    Callers select streaming explicitly because some compatible gateways only
    produce a useful answer when ``stream: true`` is present. SSE transport is
    normalized here so Agent and Runtime layers never depend on delta events.
    """

    if stream:
        return _decode_sse_chat_response(body)
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("model provider response must be a JSON object")
    return value


def is_complete_stream_response(body: bytes) -> bool:
    """Return whether buffered stream bytes contain a terminal usable response.

    This predicate is used only when the transport ends before an SSE ``[DONE]``
    marker. A terminal finish reason is necessary, and any native tool call must
    have a complete identity, function name and JSON-object argument payload.
    """

    try:
        value = decode_response(body, stream=True)
    except (ValueError, UnicodeDecodeError, TypeError, json.JSONDecodeError):
        return False
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, dict) or choice.get("finish_reason") is None:
            return False
        message = choice.get("message")
        if not isinstance(message, dict):
            return False
        calls = message.get("tool_calls")
        if choice.get("finish_reason") == "tool_calls" and not calls:
            return False
        if calls is None:
            continue
        if not isinstance(calls, list):
            return False
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                return False
            function = call.get("function")
            if not isinstance(function, dict):
                return False
            if not isinstance(function.get("name"), str) or not function["name"]:
                return False
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                return False
            try:
                decoded_arguments = json.loads(arguments)
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
            if not isinstance(decoded_arguments, dict):
                return False
    return True


def _decode_sse_chat_response(body: bytes) -> dict[str, Any]:
    """Aggregate OpenAI-style SSE deltas without retaining private reasoning."""

    text = body.decode("utf-8")
    # A provider may ignore ``stream: true`` and return an ordinary JSON
    # envelope. Accepting that response keeps the explicit request compatible
    # without guessing a transport from model names or endpoint hosts.
    if not text.lstrip().startswith("data:"):
        value = json.loads(text)
        if not isinstance(value, dict):
            raise TypeError("model provider response must be a JSON object")
        return value

    chunks: list[dict[str, Any]] = []
    for payload in _sse_data_payloads(text):
        if payload == "[DONE]":
            break
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("SSE data must contain a JSON object")
        if isinstance(value.get("error"), dict):
            message = value["error"].get("message") or "unknown provider error"
            from src.llm.recovery import service_error
            raise service_error(message, status_code=value["error"].get("status_code"))
        chunks.append(value)
    if not chunks:
        raise ValueError("model provider SSE response contained no JSON chunks")
    return _merge_chat_completion_chunks(chunks)


def _sse_data_payloads(text: str) -> list[str]:
    """Return joined ``data`` fields for complete SSE events."""

    payloads: list[str] = []
    data_lines: list[str] = []
    for line in [*text.splitlines(), ""]:
        if not line:
            if data_lines:
                payloads.append("\n".join(data_lines))
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data" or not separator:
            continue
        data_lines.append(value[1:] if value.startswith(" ") else value)
    return payloads


def _merge_chat_completion_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one non-stream Chat Completions envelope from ordered deltas."""

    result: dict[str, Any] = {"object": "chat.completion"}
    choices: dict[int, dict[str, Any]] = {}
    returned_model = ""
    for chunk in chunks:
        for key in ("id", "created", "system_fingerprint", "service_tier"):
            if key in chunk and key not in result:
                result[key] = deepcopy(chunk[key])
        model = chunk.get("model")
        if isinstance(model, str) and model:
            if returned_model and returned_model != model:
                raise ValueError("model provider changed model within one SSE response")
            returned_model = model
            result["model"] = model
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            result["usage"] = deepcopy(usage)

        raw_choices = chunk.get("choices")
        if raw_choices is None:
            continue
        if not isinstance(raw_choices, list):
            raise TypeError("model provider SSE choices must be a list")
        for position, raw_choice in enumerate(raw_choices):
            if not isinstance(raw_choice, dict):
                raise TypeError("model provider SSE choice must be an object")
            index = raw_choice.get("index", position)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("model provider SSE choice index must be non-negative")
            state = choices.setdefault(
                index,
                {
                    "role": "assistant",
                    "content_parts": [],
                    "tool_calls": {},
                    "finish_reason": None,
                },
            )
            delta = raw_choice.get("delta")
            if delta is None:
                delta = raw_choice.get("message")
            if delta is not None:
                if not isinstance(delta, dict):
                    raise TypeError("model provider SSE delta must be an object")
                _merge_message_delta(state, delta)
            finish_reason = raw_choice.get("finish_reason")
            if finish_reason is not None:
                state["finish_reason"] = deepcopy(finish_reason)

    if not choices:
        raise ValueError("model provider SSE response contained no choices")
    result["choices"] = [
        _finalize_stream_choice(index, choices[index]) for index in sorted(choices)
    ]
    return result


def _merge_message_delta(state: dict[str, Any], delta: dict[str, Any]) -> None:
    """Merge only public assistant content and native tool-call fields."""

    role = delta.get("role")
    if isinstance(role, str) and role:
        state["role"] = role
    content = delta.get("content")
    if content is not None:
        if not isinstance(content, str):
            raise TypeError("model provider SSE content delta must be a string")
        state["content_parts"].append(content)
    raw_calls = delta.get("tool_calls")
    if raw_calls is None:
        return
    if not isinstance(raw_calls, list):
        raise TypeError("model provider SSE tool_calls delta must be a list")
    for position, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            raise TypeError("model provider SSE tool call must be an object")
        index = raw_call.get("index", position)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("model provider SSE tool call index must be non-negative")
        call = state["tool_calls"].setdefault(
            index,
            {"id": "", "type": "function", "name_parts": [], "argument_parts": []},
        )
        call_id = raw_call.get("id")
        if isinstance(call_id, str) and call_id:
            call["id"] = call_id
        call_type = raw_call.get("type")
        if isinstance(call_type, str) and call_type:
            call["type"] = call_type
        function = raw_call.get("function")
        if function is None:
            continue
        if not isinstance(function, dict):
            raise TypeError("model provider SSE tool function must be an object")
        name = function.get("name")
        arguments = function.get("arguments")
        if name is not None:
            if not isinstance(name, str):
                raise TypeError("model provider SSE tool name must be a string")
            call["name_parts"].append(name)
        if arguments is not None:
            if not isinstance(arguments, str):
                raise TypeError("model provider SSE tool arguments must be a string")
            call["argument_parts"].append(arguments)


def _finalize_stream_choice(index: int, state: dict[str, Any]) -> dict[str, Any]:
    content_parts = state["content_parts"]
    message: dict[str, Any] = {
        "role": state["role"],
        "content": "".join(content_parts) if content_parts else None,
    }
    if state["tool_calls"]:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": call["type"],
                "function": {
                    "name": "".join(call["name_parts"]),
                    "arguments": "".join(call["argument_parts"]),
                },
            }
            for _, call in sorted(state["tool_calls"].items())
        ]
    return {
        "index": index,
        "message": message,
        "finish_reason": state["finish_reason"],
    }


def public_response_snapshot(response: dict[str, Any]) -> dict[str, Any]:
    """Persist only provider fields that are public protocol evidence."""

    choices = response.get("choices")
    choices = choices if isinstance(choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    raw_message = choice.get("message")
    raw_message = raw_message if isinstance(raw_message, dict) else {}
    message: dict[str, Any] = {
        "role": raw_message.get("role") or "assistant",
        "content": redact_provider_value(raw_message.get("content")),
    }
    if "tool_calls" in raw_message:
        message["tool_calls"] = redact_provider_value(raw_message.get("tool_calls"))
    snapshot = {
        "finish_reason": choice.get("finish_reason"),
        "message": message,
    }
    usage = response.get("usage")
    if isinstance(usage, dict):
        snapshot["usage"] = redact_provider_value(usage)
    return snapshot
