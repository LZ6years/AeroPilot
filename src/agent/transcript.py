"""Canonical, append-only Agent transcript and provider-message projection.

The transcript is the single owner of model intent, tool calls, tool results,
and observation checkpoints. Context modules still build current-state data;
they do not retain conversational history.
"""

from __future__ import annotations

import copy
import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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


PublicIntentSource = Literal["provider_content", "tool_arguments_fallback"]


@dataclass(frozen=True)
class AssistantTurn:
    """One canonical provider turn with an auditable intent summary.

    ``public_intent`` is the text Runtime may use for user-facing action
    reasons. ``public_intent_source`` distinguishes provider-authored content
    from the deterministic compatibility summary used when a provider emits
    valid ToolCalls with ``content=null``.
    """

    turn_id: str
    public_intent: str
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    public_intent_source: PublicIntentSource = "provider_content"

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "public_intent": self.public_intent,
            "public_intent_source": self.public_intent_source,
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
    model_inputs: list[dict[str, Any]] = field(default_factory=list)

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
        if self.model_inputs:
            data["model_inputs"] = [dict(item) for item in self.model_inputs]
        return data

    def to_provider_dict(self) -> dict[str, Any]:
        """Return the compact result replayed to the model.

        Capability handlers may retain calculation diagnostics under
        ``data.audit_details``. Those fields remain durable for review, but are
        deliberately excluded from the conversational result.
        """

        payload = self.to_dict()
        result_data = payload.get("data")
        if isinstance(result_data, dict):
            result_data.pop("audit_details", None)
        payload.pop("model_inputs", None)
        return payload


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

    def append_provider_rejection(self, rejected_turn: dict[str, Any]) -> None:
        """Persist an unexecutable provider turn for audit, never replay."""

        self._append("provider_turn_rejected", copy.deepcopy(rejected_turn))

    def append_task_terminal(self, payload: dict[str, Any]) -> None:
        """Persist a terminal task outcome without projecting it to the provider."""

        self._append("task_terminal", copy.deepcopy(payload))

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

    def prior_evidence_reference_ids(self) -> set[str]:
        """Return observation and call references preceding the latest assistant turn.

        A terminal control call must cite evidence that already existed when
        the provider made its decision. Excluding the newest assistant turn
        prevents a report from using its own ToolCall ID as circular evidence.
        """

        limit = len(self.items)
        if limit and self.items[-1].kind == "assistant_turn":
            limit -= 1
        references: set[str] = set()
        for item in self.items[:limit]:
            if item.kind == "state_checkpoint":
                observation_id = item.payload.get("observation_id")
                if observation_id:
                    references.add(str(observation_id))
            elif item.kind == "assistant_turn":
                turn: AssistantTurn = item.payload
                references.update(call.tool_call_id for call in turn.tool_calls)
            elif item.kind == "tool_result":
                result: ToolResult = item.payload
                references.add(result.tool_call_id)
                if result.result_ref:
                    references.add(result.result_ref)
        return references

    def phase_tool_results(
        self,
        phase_name: str,
        tool_name: str,
    ) -> list[ToolResult]:
        """Return tool results from the current contiguous phase transcript."""

        start_index = 0
        found_phase = False
        for index in range(len(self.items) - 1, -1, -1):
            item = self.items[index]
            if item.kind != "state_checkpoint":
                continue
            context = item.payload.get("context")
            current_phase = (
                context.get("current_phase") if isinstance(context, dict) else None
            )
            checkpoint_phase = (
                str(current_phase.get("name") or "")
                if isinstance(current_phase, dict)
                else ""
            )
            if checkpoint_phase == phase_name:
                found_phase = True
                start_index = index
                continue
            if found_phase:
                break
        if not found_phase:
            return []
        return [
            item.payload
            for item in self.items[start_index:]
            if item.kind == "tool_result"
            and isinstance(item.payload, ToolResult)
            and item.payload.tool_name == tool_name
        ]

    def provider_messages(self, registry: "ToolRegistry") -> list[dict[str, Any]]:
        """Render the complete active transcript without compaction."""

        messages: list[dict[str, Any]] = []
        seen_media: set[str] = set()
        last_state_index_by_observation: dict[str, int] = {}
        for index, candidate in enumerate(self.items):
            if candidate.kind != "state_checkpoint":
                continue
            observation_id = candidate.payload.get("observation_id")
            if observation_id:
                last_state_index_by_observation[str(observation_id)] = index
        latest_image_state_index = max(
            last_state_index_by_observation.values(), default=-1
        )
        for item_index, item in enumerate(self.items):
            if item.kind == "user_instruction":
                messages.append({"role": "user", "content": item.payload["instruction"]})
                continue
            if item.kind == "state_checkpoint":
                observation_id = item.payload.get("observation_id") or "unknown"
                provider_context = _provider_state_context(
                    item.payload["context"],
                    seen_media=seen_media,
                    include_media=(
                        last_state_index_by_observation.get(str(observation_id))
                        == item_index
                    ),
                    timeline_role=(
                        "CURRENT"
                        if item_index == latest_image_state_index
                        else "HISTORY"
                    ),
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
                    # A fallback intent helps Runtime and reviewers, but it was
                    # not emitted by the provider and must not be fabricated
                    # in the provider conversation history.
                    "content": (
                        turn.public_intent or None
                        if turn.public_intent_source == "provider_content"
                        else None
                    ),
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
                        "content": json.dumps(
                            result.to_provider_dict(), ensure_ascii=False
                        ),
                    }
                )
                evidence_message = _tool_evidence_message(result)
                if evidence_message is not None:
                    messages.append(evidence_message)
                continue
            if item.kind == "protocol_feedback":
                messages.append({"role": "user", "content": item.payload["message"]})
            # provider_turn_rejected is audit-only. Replaying malformed or
            # unknown ToolCalls without ToolResults can itself violate the
            # provider protocol; the following feedback is the repair input.
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
    context: dict[str, Any],
    *,
    seen_media: set[str],
    include_media: bool,
    timeline_role: str,
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
    _drop_duplicate_media(
        projected,
        seen_media,
        include_media=include_media,
        timeline_role=timeline_role,
    )
    return projected


def _drop_duplicate_media(
    context: dict[str, Any],
    seen_media: set[str],
    *,
    include_media: bool,
    timeline_role: str,
) -> None:
    """Attach each observation image once and label its chronological role."""
    checkpoint_media: set[str] = set()
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
            if not include_media:
                continue
            identities = _media_identities(item)
            if identities & checkpoint_media:
                continue
            if identities & seen_media:
                continue
            seen_media.update(identities)
            checkpoint_media.update(identities)
            item["timeline_role"] = timeline_role
            unique_inputs.append(item)
        fragment["model_inputs"] = unique_inputs


def _media_identities(item: dict[str, Any]) -> set[str]:
    identities: set[str] = set()
    data = item.get("data")
    if isinstance(data, str) and data.startswith("data:image/"):
        digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
        identities.add(f"provider_media_sha256:{digest}")
    for key in ("artifact_ref", "frame_id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            identities.add(f"{key}:{value}")
    observation_id = item.get("observation_id")
    if isinstance(observation_id, str) and observation_id:
        identities.add(f"observation:{observation_id}")
    return identities


def _tool_evidence_message(result: ToolResult) -> dict[str, Any] | None:
    """Project safe path-based ToolResult evidence as a following user turn.

    Provider tool messages stay compact JSON strings for broad compatibility.
    The assistant ToolCall is closed by that tool message before this ordinary
    multimodal user message asks the model to verify its own selected region.
    """

    content: list[dict[str, Any]] = []
    for item in result.model_inputs:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        path = Path(str(item.get("path") or item.get("artifact_ref") or ""))
        if not path.is_file():
            continue
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        mime_type = str(item.get("mime_type") or "image/png")
        observation_id = str(item.get("observation_id") or "unknown")
        usage = str(item.get("usage") or "Verify this tool evidence.")
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"TOOL EVIDENCE tool_call_id={result.tool_call_id} "
                        f"observation_id={observation_id}. {usage}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}"
                    },
                },
            ]
        )
    if not content:
        return None
    return {"role": "user", "content": content}
