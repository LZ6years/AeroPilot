"""OpenAI Responses wire adapter normalized to FlyAgent's model contracts.

FlyAgent keeps one Chat-style, provider-independent conversation shape inside
the Agent.  This module translates only at the HTTP boundary: native tool
calls become Responses ``function_call`` items on the wire and are normalized
back to ``choices[0].message.tool_calls`` for callers.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from src.llm.contracts import ModelRequest, ProviderSettings


def encode_responses_request(
    settings: ProviderSettings,
    request: ModelRequest,
) -> dict[str, Any]:
    if settings.provider != "openai_compatible":
        raise ValueError(f"Unsupported model provider: {settings.provider}")
    if settings.stream:
        raise ValueError("streaming Responses API is not supported")
    payload: dict[str, Any] = {
        "model": settings.model_name,
        "input": responses_input_items(request.messages),
        # The client is stateless and always resubmits its authoritative input.
        "store": False,
    }
    if request.tools is not None:
        payload["tools"] = [responses_function_tool(item) for item in request.tools]
    for key, value in deepcopy(request.parameters).items():
        if key == "max_tokens":
            payload["max_output_tokens"] = value
        elif key == "reasoning_effort":
            payload["reasoning"] = {"effort": value}
        elif key == "response_format":
            payload["text"] = {"format": _response_text_format(value)}
        else:
            payload[key] = value
    return payload


def decode_responses_response(body: bytes) -> dict[str, Any]:
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("model provider response must be a JSON object")
    return normalize_responses_response(value)


def normalize_responses_response(value: dict[str, Any]) -> dict[str, Any]:
    """Return the public Chat-style envelope consumed by existing callers."""

    raw_output = value.get("output")
    if not isinstance(raw_output, list):
        raise TypeError("Responses output must be a list")
    content_parts: list[str] = []
    calls: list[dict[str, Any]] = []
    for item in raw_output:
        if not isinstance(item, dict):
            raise TypeError("Responses output item must be an object")
        item_type = item.get("type")
        if item_type == "message":
            raw_content = item.get("content")
            if not isinstance(raw_content, list):
                raise TypeError("Responses message content must be a list")
            for part in raw_content:
                if not isinstance(part, dict):
                    raise TypeError("Responses message content item must be an object")
                if part.get("type") == "output_text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise TypeError("Responses output_text must contain text")
                    content_parts.append(text)
        elif item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not all(isinstance(field, str) and field for field in (call_id, name)):
                raise ValueError("Responses function_call requires call_id and name")
            if not isinstance(arguments, str):
                raise TypeError("Responses function_call arguments must be a string")
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )

    status = value.get("status")
    incomplete = value.get("incomplete_details")
    incomplete_reason = (
        incomplete.get("reason") if isinstance(incomplete, dict) else None
    )
    if incomplete_reason == "max_output_tokens":
        finish_reason = "length"
    elif calls:
        finish_reason = "tool_calls"
    elif status == "incomplete":
        finish_reason = "incomplete"
    else:
        finish_reason = "stop"
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }
    if calls:
        message["tool_calls"] = calls
    result: dict[str, Any] = {
        "id": value.get("id"),
        "object": "chat.completion",
        "model": value.get("model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    usage = value.get("usage")
    if isinstance(usage, dict):
        result["usage"] = deepcopy(usage)
    if "instructions" in value:
        # This is a documented public Responses field. Keeping it available in
        # the normalized envelope lets provider diagnostics inspect upstream
        # instruction injection without changing the caller-owned message list.
        result["instructions"] = deepcopy(value.get("instructions"))
    return result


def responses_input_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the canonical chat history to reusable Responses input items."""

    result: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("tool message requires tool_call_id")
            output = message.get("content")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
            result.append(
                {"type": "function_call_output", "call_id": call_id, "output": output}
            )
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"unsupported Responses message role: {role}")
        content = _response_content(message.get("content"), role=str(role))
        if content:
            result.append({"role": role, "content": content})
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            continue
        if not isinstance(raw_calls, list):
            raise TypeError("assistant tool_calls must be a list")
        for raw_call in raw_calls:
            function = raw_call.get("function") if isinstance(raw_call, dict) else None
            if not isinstance(function, dict):
                raise TypeError("assistant tool call function must be an object")
            call_id = raw_call.get("id")
            name = function.get("name")
            arguments = function.get("arguments")
            if not all(isinstance(field, str) and field for field in (call_id, name)):
                raise ValueError("assistant tool call requires id and function name")
            if not isinstance(arguments, str):
                raise TypeError("assistant tool call arguments must be a string")
            result.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )
    return result


def _response_content(content: Any, *, role: str) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if not isinstance(content, list):
        raise TypeError("message content must be a string, list, or null")
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise TypeError("message content item must be an object")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise TypeError("text content item must contain text")
            parts.append({"type": text_type, "text": text})
        elif part_type == "image_url":
            if role != "user":
                raise ValueError("Responses input images must belong to user messages")
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str) or not url:
                raise ValueError("image_url content item requires a URL")
            converted: dict[str, Any] = {"type": "input_image", "image_url": url}
            detail = image.get("detail") if isinstance(image, dict) else None
            if isinstance(detail, str) and detail:
                converted["detail"] = detail
            parts.append(converted)
        else:
            raise ValueError(f"unsupported message content type: {part_type}")
    return parts


def responses_function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one canonical function tool to a Responses function tool."""

    if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
        raise ValueError("Responses adapter supports function tools only")
    function = deepcopy(tool["function"])
    result = {"type": "function", **function}
    return result


def _response_text_format(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("response_format must be an object")
    format_type = value.get("type")
    if format_type == "json_object":
        return {"type": "json_object"}
    if format_type != "json_schema":
        raise ValueError(f"unsupported response_format type: {format_type}")
    schema = value.get("json_schema")
    if not isinstance(schema, dict):
        raise TypeError("json_schema response format must contain json_schema")
    return {"type": "json_schema", **deepcopy(schema)}
