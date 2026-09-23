"""Canonical, append-only Agent transcript and isolated history snapshots.

The transcript is the single owner of model intent, tool calls, tool results,
and observation checkpoints. Context modules still build current-state data;
they do not retain conversational history.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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
class TranscriptEvent:
    """One canonical event; snapshot() isolates nested payloads for readers."""
    kind: str
    payload: Any


class AgentTranscript:
    """Keep the complete active-task transcript and persist every event."""

    def __init__(self, writer: "MemoryEventWriter") -> None:
        self.writer = writer
        self.items: list[TranscriptEvent] = []
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


    def snapshot(self) -> tuple[TranscriptEvent, ...]:
        """Return isolated events so request projections cannot mutate history."""
        return tuple(copy.deepcopy(self.items))

    def provider_messages(self, registry: "ToolRegistry") -> list[dict[str, Any]]:
        """Compatibility entry: full history, using the shared projector."""
        from src.agent.history_projection import HistoryProjector

        return HistoryProjector().project(self, registry).messages

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
        self.items.append(TranscriptEvent(kind, payload))
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
