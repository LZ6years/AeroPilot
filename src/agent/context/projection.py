"""Project an immutable V2 transcript snapshot into provider messages."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.agent.context.request_assembler import RequestAssembler
from src.agent.transcript import AgentTranscript, AssistantTurn, ToolResult, TranscriptEvent
from src.memory.context.contracts import HISTORY_PROJECTION_SCHEMA, HistoryImageCandidate
from src.memory.context.modules.base import ContextModule

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ProjectionResult:
    messages: list[dict[str, Any]]
    audit: dict[str, Any]


class HistoryProjector:
    """Select history identities first, then hydrate only selected media."""

    def __init__(
        self,
        bindings: dict[str, tuple[ContextModule, dict[str, Any]]] | None = None,
        *,
        assembler: RequestAssembler | None = None,
    ) -> None:
        self.bindings = {
            slot: (module, copy.deepcopy(config))
            for slot, (module, config) in (bindings or {}).items()
        }
        for module, config in self.bindings.values():
            module.validate_history_policy(copy.deepcopy(config))
        self.assembler = assembler or RequestAssembler()

    def project(
        self, transcript: AgentTranscript, registry: "ToolRegistry"
    ) -> ProjectionResult:
        return self._project_owned_snapshot(transcript.snapshot(), registry)

    def project_snapshot(
        self, events: tuple[TranscriptEvent, ...], registry: "ToolRegistry"
    ) -> ProjectionResult:
        return self._project_owned_snapshot(copy.deepcopy(events), registry)

    def _selection(
        self, items: tuple[TranscriptEvent, ...]
    ) -> tuple[dict[str, set[str]], dict[str, Any]]:
        observations: dict[str, dict[str, Any]] = {}
        for item in items:
            if item.kind != "state_checkpoint":
                continue
            checkpoint = _payload_dict(item.payload)
            observation_id = checkpoint.get("observation_id")
            if observation_id and str(observation_id) not in observations:
                observations[str(observation_id)] = checkpoint
        selected: dict[str, set[str]] = {}
        policies: dict[str, Any] = {}
        for slot, (module, config) in self.bindings.items():
            candidates = tuple(
                HistoryImageCandidate(
                    observation_id,
                    sequence,
                    tuple(_fragment_frame_ids(checkpoint, slot)),
                )
                for sequence, (observation_id, checkpoint) in enumerate(
                    observations.items()
                )
            )
            output = tuple(
                module.select_history_images(candidates, copy.deepcopy(config))
            )
            if (
                any(not isinstance(value, str) for value in output)
                or len(output) != len(set(output))
                or not set(output) <= observations.keys()
            ):
                raise ValueError(f"Invalid history selection from module {slot}")
            chosen = set(output)
            selected[slot] = chosen
            policies[slot] = {
                "config": copy.deepcopy(config),
                "selected_observation_ids": [
                    key for key in observations if key in chosen
                ],
                "omitted_count": len(observations) - len(chosen),
            }
        return selected, {
            "schema_version": HISTORY_PROJECTION_SCHEMA,
            "observation_count": len(observations),
            "policies": policies,
            "observation_image_count": 0,
            "tool_image_count": 0,
            "missing_images": [],
        }

    def _project_owned_snapshot(
        self, items: tuple[TranscriptEvent, ...], registry: "ToolRegistry"
    ) -> ProjectionResult:
        selected, audit = self._selection(items)
        messages: list[dict[str, Any]] = []
        envelope: dict[str, Any] = {}
        first_state_index_by_observation: dict[str, int] = {}
        for index, event in enumerate(items):
            if event.kind != "state_checkpoint":
                continue
            checkpoint = _payload_dict(event.payload)
            observation_id = checkpoint.get("observation_id")
            if observation_id:
                first_state_index_by_observation.setdefault(str(observation_id), index)
        newest_observation = next(reversed(first_state_index_by_observation), None)
        seen_media: set[str] = set()

        for index, event in enumerate(items):
            if event.kind == "instruction_envelope":
                envelope = _payload_dict(event.payload)
                continue
            if event.kind == "user_instruction":
                messages.append(
                    {"role": "user", "content": event.payload["instruction"]}
                )
                context_message = self.assembler.instruction_context_message(envelope)
                if context_message is not None:
                    messages.append(context_message)
                continue
            if event.kind == "state_checkpoint":
                checkpoint = _payload_dict(event.payload)
                observation_id = str(checkpoint.get("observation_id") or "unknown")
                include_media = (
                    first_state_index_by_observation.get(observation_id) == index
                )
                inputs = _selected_checkpoint_inputs(
                    checkpoint,
                    observation_id=observation_id,
                    include_media=include_media,
                    selected=selected,
                    seen_media=seen_media,
                )
                timeline_role = (
                    "CURRENT" if observation_id == newest_observation else "HISTORY"
                )
                try:
                    content = self.assembler.checkpoint_content(
                        checkpoint,
                        model_inputs=inputs,
                        timeline_role=timeline_role,
                    )
                except ValueError as exc:
                    content = self.assembler.checkpoint_content(
                        checkpoint, model_inputs=[], timeline_role=timeline_role
                    )
                    audit["missing_images"].append(
                        {"observation_id": observation_id, "reason": str(exc)}
                    )
                prefix = {
                    "type": "text",
                    "text": (
                        f"STATE CHECKPOINT checkpoint_id={checkpoint.get('checkpoint_id')} "
                        f"observation_id={observation_id}. Treat this as new chronological state."
                    ),
                }
                if isinstance(content, list):
                    body = [prefix, *content]
                else:
                    body = [prefix, {"type": "text", "text": content}]
                image_count = sum(part.get("type") == "image_url" for part in body)
                audit["observation_image_count"] += image_count
                if include_media and not image_count:
                    prefix["text"] += (
                        " Observation image unavailable or omitted by history sampling; "
                        "no older image substitutes CURRENT."
                    )
                messages.append({"role": "user", "content": body})
                continue
            if event.kind == "assistant_turn":
                turn: AssistantTurn = event.payload
                tool_calls = [
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": registry.get(call.tool_name).wire_name,
                            "arguments": json.dumps(call.args, ensure_ascii=False),
                        },
                    }
                    for call in turn.tool_calls
                ]
                message: dict[str, Any] = {
                    "role": "assistant",
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
            if event.kind == "tool_result":
                result: ToolResult = event.payload
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": json.dumps(
                            result.to_provider_dict(), ensure_ascii=False
                        ),
                    }
                )
                evidence = self._tool_evidence_message(result)
                if evidence is not None:
                    messages.append(evidence)
                    audit["tool_image_count"] += sum(
                        part.get("type") == "image_url"
                        for part in evidence["content"]
                    )
                continue
            if event.kind == "protocol_feedback":
                messages.append({"role": "user", "content": event.payload["message"]})
        return ProjectionResult(messages, audit)

    def _tool_evidence_message(
        self, result: ToolResult
    ) -> dict[str, Any] | None:
        content: list[dict[str, Any]] = []
        for item in result.model_inputs:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            raw_ref = item.get("media_ref")
            if not isinstance(raw_ref, dict):
                continue
            ref = raw_ref
            try:
                prepared = self.assembler.media_hydrator.hydrate(ref)
            except ValueError:
                continue
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"TOOL EVIDENCE tool_call_id={result.tool_call_id} "
                            f"observation_id={ref.get('observation_id', 'unknown')}. "
                            f"{ref.get('usage') or 'Verify this tool evidence.'}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": prepared.data_url},
                    },
                ]
            )
        return {"role": "user", "content": content} if content else None


def _payload_dict(payload: Any) -> dict[str, Any]:
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, dict) else {}
    return copy.deepcopy(payload) if isinstance(payload, dict) else {}


def _fragment_frame_ids(checkpoint: dict[str, Any], slot: str) -> list[str]:
    return [
        str(ref.get("frame_id") or "")
        for fragment in checkpoint.get("fragments", [])
        if isinstance(fragment, dict) and fragment.get("name") == slot
        for item in fragment.get("model_inputs", [])
        if isinstance(item, dict)
        for ref in [item.get("media_ref")]
        if isinstance(ref, dict)
    ]


def _selected_checkpoint_inputs(
    checkpoint: dict[str, Any],
    *,
    observation_id: str,
    include_media: bool,
    selected: dict[str, set[str]],
    seen_media: set[str],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for fragment in checkpoint.get("fragments", []):
        if not isinstance(fragment, dict):
            continue
        slot = str(fragment.get("name") or "")
        chosen = slot not in selected or observation_id in selected[slot]
        for item in fragment.get("model_inputs", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                inputs.append(copy.deepcopy(item))
                continue
            ref = item.get("media_ref")
            if not include_media or not chosen or not isinstance(ref, dict):
                continue
            # The same bytes can be a valid observation at two different
            # times. Deduplicate repeated references within one observation,
            # while the hydrator's SHA cache avoids reading/encoding twice.
            identity = (
                f"{observation_id}:"
                f"{ref.get('sha256') or ref.get('ref') or ''}"
            )
            if not identity or identity in seen_media:
                continue
            if str(ref.get("observation_id") or observation_id) != observation_id:
                continue
            seen_media.add(identity)
            inputs.append(copy.deepcopy(item))
    return inputs
