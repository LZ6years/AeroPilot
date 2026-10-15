"""Canonical, append-only Agent transcript and isolated history snapshots.

The transcript is the single owner of model intent, tool calls, tool results,
and observation checkpoints. Context modules still build current-state data;
they do not retain conversational history.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from src.tools.diagnostics import Diagnostic

from src.memory.context.contracts import (
    InstructionEnvelope,
    MediaRef,
    StateCheckpoint,
    TRANSCRIPT_SCHEMA,
    assert_binary_free,
)

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
    diagnostic: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.diagnostic is not None:
            validated = Diagnostic(**self.diagnostic)
            if validated.execution is None:
                raise ValueError("tool-call diagnostic requires execution")
            object.__setattr__(self, "diagnostic", validated.to_dict())
        assert_binary_free(self.model_inputs)
        for item in self.model_inputs:
            if item.get("type") != "image":
                continue
            raw_ref = item.get("media_ref")
            if not isinstance(raw_ref, dict):
                raise ValueError("V3 tool image inputs require media_ref")
            MediaRef(**raw_ref)

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
        if self.diagnostic is not None:
            data["diagnostic"] = dict(self.diagnostic)
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

    def start_instruction(
        self,
        instruction: str,
        instruction_id: str | None,
        envelope: InstructionEnvelope,
    ) -> None:
        self.items = []
        self.instruction_id = instruction_id
        if envelope.instruction_id != instruction_id:
            raise ValueError("instruction envelope identity mismatch")
        self._append("instruction_envelope", envelope)
        self._append("user_instruction", {"instruction": instruction})

    def append_state(self, checkpoint: StateCheckpoint) -> None:
        """Append the exact immutable checkpoint published by Runtime."""

        self._append("state_checkpoint", checkpoint)

    def append_assistant(self, turn: AssistantTurn) -> None:
        self._append("assistant_turn", turn)

    def append_tool_result(self, result: ToolResult) -> None:
        assert_binary_free(result.to_dict())
        self._append("tool_result", result)

    def append_protocol_feedback(self, message: str, diagnostic: dict[str, Any] | None = None) -> None:
        payload = {"message": message}
        if diagnostic is not None:
            payload["diagnostic"] = Diagnostic(**diagnostic).to_dict()
        self._append("protocol_feedback", payload)

    def append_provider_rejection(self, rejected_turn: dict[str, Any]) -> None:
        """Persist an unexecutable provider turn for audit, never replay."""

        self._append("provider_turn_rejected", copy.deepcopy(rejected_turn))

    def append_task_terminal(self, payload: dict[str, Any]) -> None:
        """Persist a terminal task outcome without projecting it to the provider."""

        self._append("task_terminal", copy.deepcopy(payload))

    def instruction_envelope(self) -> dict[str, Any]:
        for item in self.items:
            if item.kind == "instruction_envelope":
                return _payload_dict(item.payload)
        return {}

    def latest_checkpoint(self) -> StateCheckpoint | None:
        for item in reversed(self.items):
            if item.kind == "state_checkpoint":
                return copy.deepcopy(item.payload)
        return None

    def latest_observation_id(self) -> str | None:
        """Return the ID attached to the newest chronological state checkpoint."""

        for item in reversed(self.items):
            if item.kind == "state_checkpoint":
                value = getattr(item.payload, "observation_id", None)
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
                observation_id = getattr(item.payload, "observation_id", None)
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

    @property
    def event_count(self) -> int:
        """Return the current append-only boundary without exposing the list."""

        return len(self.items)

    def snapshot_range(
        self, start: int, end: int | None = None
    ) -> tuple[TranscriptEvent, ...]:
        """Copy one validated half-open range for bounded request projection."""

        stop = len(self.items) if end is None else end
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start <= stop <= len(self.items)
        ):
            raise ValueError("snapshot range must be within the transcript")
        return tuple(copy.deepcopy(self.items[start:stop]))

    def snapshot_before(self, event_count: int) -> tuple[TranscriptEvent, ...]:
        """Copy the prefix ending at an exclusive, previously recorded boundary.

        Runtime can append and validate a turn before paying for the control
        tool's pre-turn evidence view. Never include the submitted turn itself.
        """
        if type(event_count) is not int or not 0 <= event_count <= len(self.items):
            raise ValueError("event_count must be an integer within the transcript")
        return tuple(copy.deepcopy(self.items[:event_count]))

    def provider_messages(self, registry: "ToolRegistry") -> list[dict[str, Any]]:
        """Project the complete V3 history through the shared projector."""
        from src.agent.context.projection import HistoryProjector

        return HistoryProjector().project(self, registry).messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRANSCRIPT_SCHEMA,
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
        assert_binary_free(_payload_dict(payload))
        self.items.append(TranscriptEvent(kind, payload))
        self.writer.write_transcript_event(
            instruction_id=self.instruction_id,
            kind=kind,
            payload=_payload_dict(payload),
            schema_version=TRANSCRIPT_SCHEMA,
        )


def _payload_dict(payload: Any) -> dict[str, Any]:
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, dict) else {"value": value}
    return dict(payload) if isinstance(payload, dict) else {"value": payload}
