"""OpenAI-compatible Chat Completions wire encoding, not agent response semantics."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from src.llm.contracts import ModelRequest, ProviderSettings
from src.llm.audit import redact_provider_value


def chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url:
        return "https://api.openai.com/v1/chat/completions"
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def encode_request(settings: ProviderSettings, request: ModelRequest) -> dict[str, Any]:
    if settings.provider != "openai_compatible":
        raise ValueError(f"Unsupported model provider: {settings.provider}")
    payload = {"model": settings.model_name, "messages": deepcopy(request.messages)}
    if request.tools is not None:
        payload["tools"] = deepcopy(request.tools)
    payload.update(deepcopy(request.parameters))
    return payload


def encode_headers(settings: ProviderSettings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"}


def decode_response(body: bytes) -> dict[str, Any]:
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("model provider response must be a JSON object")
    return value


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
