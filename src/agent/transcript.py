"""Canonical, append-only Agent transcript and provider-message projection.

The transcript is the single owner of model intent, tool calls, tool results,
and observation checkpoints. Context modules still build current-state data;
they do not retain conversational history.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent.planning.model_inputs import build_model_user_content

if TYPE_CHECKING:
    from src.memory.stores.event_store import MemoryEventWriter
    from src.tools.registry import ToolRegistry


@dataclass(frozen=True)
class AgentToolCall:
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "args": dict(self.args),
        }


@dataclass(frozen=True)
class AssistantTurn:
    turn_id: str
    public_intent: str
    tool_calls: list[AgentToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "public_intent": self.public_intent,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    status: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[str] = field(default_factory=list)
    result_ref: str | None = None
    requested_args: dict[str, Any] = field(default_factory=dict)
    executed_args: dict[str, Any] = field(default_factory=dict)
    safety_interventions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "message": self.message,
            "data": dict(self.data),
            "artifact_refs": list(self.artifact_refs),
            "requested_args": dict(self.requested_args),
            "executed_args": dict(self.executed_args),
            "safety_interventions": [dict(item) for item in self.safety_interventions],
        }
        if self.result_ref:
            data["result_ref"] = self.result_ref
        return data


@dataclass(frozen=True)
class _TranscriptItem:
    kind: str
    payload: Any


class AgentTranscript:
    """Keep the complete active-task transcript and persist every event."""

    def __init__(self, writer: "MemoryEventWriter") -> None:
        self.writer = writer
        self.items: list[_TranscriptItem] = []
        self.instruction_id: str | None = None

    def start_instruction(self, instruction: str, instruction_id: str | None) -> None:
        self.items = []
        self.instruction_id = instruction_id
        self._append("user_instruction", {"instruction": instruction})

    def append_state(self, context: dict[str, Any], observation_id: str | None) -> None:
        self._append(
            "state_checkpoint",
            {
                "observation_id": observation_id,
                "context": copy.deepcopy(context),
            },
        )

    def append_assistant(self, turn: AssistantTurn) -> None:
        self._append("assistant_turn", turn)

    def append_tool_result(self, result: ToolResult) -> None:
        self._append("tool_result", result)

    def append_protocol_feedback(self, message: str) -> None:
        self._append("protocol_feedback", {"message": message})

    def latest_context(self) -> dict[str, Any]:
        for item in reversed(self.items):
            if item.kind == "state_checkpoint":
                return copy.deepcopy(item.payload["context"])
        return {}

    def latest_observation_id(self) -> str | None:
        """Return the ID attached to the newest chronological state checkpoint."""

        for item in reversed(self.items):
            if item.kind == "state_checkpoint":
                value = item.payload.get("observation_id")
                return str(value) if value else None
        return None

    def provider_messages(self, registry: "ToolRegistry") -> list[dict[str, Any]]:
        """Render the complete active transcript without compaction."""

        messages: list[dict[str, Any]] = []
        seen_media: set[str] = set()
        for item in self.items:
            if item.kind == "user_instruction":
                messages.append({"role": "user", "content": item.payload["instruction"]})
                continue
            if item.kind == "state_checkpoint":
                observation_id = item.payload.get("observation_id") or "unknown"
                provider_context = _provider_state_context(
                    item.payload["context"], seen_media=seen_media
                )
                content = build_model_user_content(
                    provider_context, include_memory_documents=False
                )
                prefix = {
                    "type": "text",
                    "text": (
                        f"STATE CHECKPOINT observation_id={observation_id}. "
                        "Treat this as a new chronological state."
                    ),
                }
                if isinstance(content, list):
                    content = [prefix, *content]
                else:
                    content = [prefix, {"type": "text", "text": content}]
                messages.append({"role": "user", "content": content})
                continue
            if item.kind == "assistant_turn":
                turn: AssistantTurn = item.payload
                tool_calls = []
                for call in turn.tool_calls:
                    spec = registry.get(call.tool_name)
                    tool_calls.append(
                        {
                            "id": call.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": spec.wire_name,
                                "arguments": json.dumps(call.args, ensure_ascii=False),
                            },
                        }
                    )
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn.public_intent or None,
                }
                if tool_calls:
                    message["tool_calls"] = tool_calls
                messages.append(message)
                continue
            if item.kind == "tool_result":
                result: ToolResult = item.payload
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": json.dumps(result.to_dict(), ensure_ascii=False),
                    }
                )
                continue
            if item.kind == "protocol_feedback":
                messages.append({"role": "user", "content": item.payload["message"]})
        return messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "items": [
                {
                    "kind": item.kind,
                    "payload": _payload_dict(item.payload),
                }
                for item in self.items
            ],
        }

    def _append(self, kind: str, payload: Any) -> None:
        self.items.append(_TranscriptItem(kind, payload))
        self.writer.write_transcript_event(
            instruction_id=self.instruction_id,
            kind=kind,
            payload=_strip_binary(_payload_dict(payload)),
        )


def _payload_dict(payload: Any) -> dict[str, Any]:
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, dict) else {"value": value}
    return dict(payload) if isinstance(payload, dict) else {"value": payload}


def _strip_binary(value: Any) -> Any:
    """Keep durable transcript records readable while in-memory history stays exact."""

    if isinstance(value, dict):
        return {
            key: "[provider_media_rehydrated_from_artifact]"
            if key == "data" and isinstance(item, str) and item.startswith("data:image/")
            else _strip_binary(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_strip_binary(item) for item in value]
    return value


def _provider_state_context(
    context: dict[str, Any], *, seen_media: set[str]
) -> dict[str, Any]:
    """Remove capability declarations already represented by provider tools.

    The complete Runtime/debug checkpoint remains available in the canonical
    transcript record. Only its provider projection drops duplicate schemas and
    availability lists. Dynamic runtime-control status remains state evidence.
    """

    projected = copy.deepcopy(context)
    projected.pop("allowed_action_constraints", None)
    runtime_control = projected.get("runtime_control")
    if isinstance(runtime_control, dict):
        runtime_control.pop("available", None)
        runtime_control.pop("contracts", None)
        if not runtime_control:
            projected.pop("runtime_control", None)
    _drop_duplicate_media(projected, seen_media)
    return projected


def _drop_duplicate_media(context: dict[str, Any], seen_media: set[str]) -> None:
    """Send each uniquely identified image once at its first transcript position."""

    fragments = context.get("fragments")
    if not isinstance(fragments, list):
        return
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        inputs = fragment.get("model_inputs")
        if not isinstance(inputs, list):
            continue
        unique_inputs = []
        for item in inputs:
            if not isinstance(item, dict) or item.get("type") != "image":
                unique_inputs.append(item)
                continue
            identities = _media_identities(item)
            if identities & seen_media:
                continue
            seen_media.update(identities)
            unique_inputs.append(item)
        fragment["model_inputs"] = unique_inputs


def _media_identities(item: dict[str, Any]) -> set[str]:
    data = item.get("data")
    if isinstance(data, str) and data.startswith("data:image/"):
        digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
        return {f"provider_media_sha256:{digest}"}
    identities: set[str] = set()
    for key in ("artifact_ref", "frame_id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            identities.add(f"{key}:{value}")
    observation_id = item.get("observation_id")
    role = item.get("role")
    if isinstance(observation_id, str) and observation_id and isinstance(role, str):
        identities.add(f"observation_role:{observation_id}:{role}")
    return identities
