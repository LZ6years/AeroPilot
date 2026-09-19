"""Read-only projection of canonical history into provider messages.

This layer owns request visibility, never Runtime execution or durable history.
"""

from __future__ import annotations

import base64
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.transcript import AgentTranscript, AssistantTurn, ToolResult, TranscriptEvent
from src.agent.planning.model_inputs import build_model_user_content
from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import HistoryImageCandidate

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ProjectionResult:
    """An independent request view and human-only selection diagnostics."""

    messages: list[dict[str, Any]]
    audit: dict[str, Any]


class HistoryProjector:
    """Project a snapshot; the source transcript remains untouched."""

    def __init__(
        self, bindings: dict[str, tuple[ContextModule, dict[str, Any]]] | None = None
    ) -> None:
        # Keep actual module instances (plugins may replace them), but isolate
        # caller-owned settings. This layer does not discover or load modules.
        self.bindings = {
            slot: (module, copy.deepcopy(config))
            for slot, (module, config) in (bindings or {}).items()
        }
        for module, config in self.bindings.values():
            module.validate_history_policy(copy.deepcopy(config))

    def _selection(self, items: tuple[TranscriptEvent, ...]) -> tuple[dict[str, set[str]], dict[str, Any]]:
        observations: dict[str, dict[str, Any]] = {}
        for item in items:
            if item.kind == "state_checkpoint" and item.payload.get("observation_id"):
                # Assignment updates the source checkpoint but preserves the
                # observation's FIRST insertion position, including phase changes.
                observations[str(item.payload["observation_id"])] = item.payload["context"]
        selected, policies = {}, {}
        for slot, (module, config) in self.bindings.items():
            candidates = []
            for sequence, (observation_id, context) in enumerate(observations.items()):
                frame_ids = tuple(
                    str(media.get("frame_id") or "")
                    for fragment in context.get("fragments", [])
                    if fragment.get("name") == slot
                    for media in fragment.get("model_inputs", [])
                    if media.get("type") == "image"
                )
                candidates.append(HistoryImageCandidate(observation_id, sequence, frame_ids))
            output = tuple(module.select_history_images(tuple(candidates), copy.deepcopy(config)))
            if (any(not isinstance(value, str) for value in output)
                    or len(output) != len(set(output)) or not set(output) <= observations.keys()):
                raise ValueError(f"Invalid history selection from module {slot}")
            chosen = set(output)
            selected[slot] = chosen
            policies[slot] = {
                "config": copy.deepcopy(config),
                "selected_observation_ids": [key for key in observations if key in chosen],
                "omitted_count": len(observations) - len(chosen),
            }
        return selected, {
            "schema_version": "history_projection_v1",
            "policy_version": "module_history_selection_v1",
            "observation_count": len(observations),
            "policies": policies,
            "observation_image_count": 0,
            "tool_image_count": 0,
            "missing_images": [],
        }

    def project(self, transcript: AgentTranscript, registry: "ToolRegistry") -> ProjectionResult:
        """Read an isolated snapshot and project it using configured policies."""
        return self.project_snapshot(transcript.snapshot(), registry)

    def project_snapshot(
        self, events: tuple[TranscriptEvent, ...], registry: "ToolRegistry"
    ) -> ProjectionResult:
        """Project explicit events, also usable by offline readers without writes."""
        items = copy.deepcopy(events)
        selected, audit = self._selection(items)
        messages: list[dict[str, Any]] = []
        seen_media: set[str] = set()
        last_state_index_by_observation: dict[str, int] = {}
        for index, candidate in enumerate(items):
            if candidate.kind != "state_checkpoint":
                continue
            observation_id = candidate.payload.get("observation_id")
            if observation_id:
                last_state_index_by_observation[str(observation_id)] = index
        newest_observation = next(reversed(last_state_index_by_observation), None)
        latest_image_state_index = last_state_index_by_observation.get(newest_observation, -1)
        for item_index, item in enumerate(items):
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
                    observation_id=str(observation_id),
                    selected=selected,
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
                image_count = sum(part.get("type") == "image_url" for part in content)
                audit["observation_image_count"] += image_count
                is_last = last_state_index_by_observation.get(str(observation_id)) == item_index
                if is_last:
                    omitted_slots = [slot for slot, ids in selected.items() if observation_id not in ids]
                    if omitted_slots:
                        prefix["text"] += " Observation image omitted by history sampling in this request."
                    elif not image_count:
                        prefix["text"] += " Observation image unavailable in this request; no older image substitutes CURRENT."
                        audit["missing_images"].append({"observation_id": observation_id, "reason": "no_provider_ready_image"})
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
                expected_images = sum(media.get("type") == "image" for media in result.model_inputs)
                actual_images = 0
                if evidence_message is not None:
                    messages.append(evidence_message)
                    actual_images = sum(
                        part.get("type") == "image_url" for part in evidence_message["content"]
                    )
                audit["tool_image_count"] += actual_images
                if actual_images != expected_images:
                    audit["missing_images"].append({
                        "tool_call_id": result.tool_call_id,
                        "reason": "tool_evidence_image_unavailable",
                        "missing_count": expected_images - actual_images,
                    })
                continue
            if item.kind == "protocol_feedback":
                messages.append({"role": "user", "content": item.payload["message"]})
            # provider_turn_rejected is audit-only. Replaying malformed or
            # unknown ToolCalls without ToolResults can itself violate the
            # provider protocol; the following feedback is the repair input.
        return ProjectionResult(messages, audit)


def _provider_state_context(
    context: dict[str, Any],
    *,
    seen_media: set[str],
    include_media: bool,
    timeline_role: str,
    observation_id: str,
    selected: dict[str, set[str]],
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
        observation_id=observation_id,
        selected=selected,
    )
    return projected


def _drop_duplicate_media(
    context: dict[str, Any],
    seen_media: set[str],
    *,
    include_media: bool,
    timeline_role: str,
    observation_id: str,
    selected: dict[str, set[str]],
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
        slot = str(fragment.get("name") or "")
        chosen = slot not in selected or observation_id in selected[slot]
        for item in inputs:
            if not isinstance(item, dict) or item.get("type") != "image":
                unique_inputs.append(item)
                continue
            if not include_media or not chosen:
                continue
            if slot in selected and str(item.get("observation_id") or observation_id) != observation_id:
                # A legacy reference cannot smuggle an unselected observation
                # into a selected checkpoint or acquire the CURRENT label.
                continue
            # Scope identity to the actual observation, never pixel bytes.
            # Legacy repeated reference inputs retain their own observation ID.
            identities = _media_identities(item, observation_id)
            if identities & checkpoint_media:
                continue
            if identities & seen_media:
                continue
            seen_media.update(identities)
            checkpoint_media.update(identities)
            item["timeline_role"] = timeline_role
            unique_inputs.append(item)
        fragment["model_inputs"] = unique_inputs


def _media_identities(item: dict[str, Any], checkpoint_observation_id: str) -> set[str]:
    identities: set[str] = set()
    observation_id = str(item.get("observation_id") or checkpoint_observation_id)
    for key in ("artifact_ref", "frame_id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            identities.add(f"{observation_id}:{key}:{value}")
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
