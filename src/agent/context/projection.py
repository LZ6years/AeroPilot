"""Project immutable transcript events into bounded provider request history.

The projector owns chronological boundaries, deterministic facts, evidence
selection, and final message shape. Semantic compression is injected through
the context-window manager and never changes the durable transcript.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.agent.context.compression.contracts import (
    CompressedHistorySection,
    CompressionWindow,
    EvidenceCatalogEntry,
    HistoryMediaEntry,
)
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


@dataclass(frozen=True)
class ProjectionPlan:
    """A projector-owned immutable plan for one possible compaction boundary."""

    compactable_end_event_index: int | None
    windows: tuple[CompressionWindow, ...]
    evidence_catalog: tuple[EvidenceCatalogEntry, ...]
    deterministic_facts: dict[str, Any]
    exact_loaded_knowledge: tuple[dict[str, Any], ...]
    prefix_media: tuple[HistoryMediaEntry, ...]
    current_state_hint: dict[str, Any]
    instruction: str
    source_digest: str | None


@dataclass(frozen=True)
class _DecisionWindow:
    start: int
    end: int
    assistant_index: int
    closed: bool


@dataclass(frozen=True)
class _VisualSelection:
    candidate: HistoryMediaEntry
    tier: str


class HistoryProjector:
    """Select history identities first, then hydrate only selected media."""

    def __init__(
        self,
        bindings: dict[str, tuple[ContextModule, dict[str, Any]]] | None = None,
        *,
        assembler: RequestAssembler | None = None,
        media_policy: dict[str, Any] | None = None,
    ) -> None:
        self.bindings = {
            slot: (module, copy.deepcopy(config))
            for slot, (module, config) in (bindings or {}).items()
        }
        for module, config in self.bindings.values():
            module.validate_history_policy(copy.deepcopy(config))
        self.assembler = assembler or RequestAssembler()
        self.media_policy = copy.deepcopy(media_policy) if media_policy else None

    def project(
        self, transcript: AgentTranscript, registry: "ToolRegistry"
    ) -> ProjectionResult:
        return self._project_owned_snapshot(
            transcript.snapshot(), registry, event_indices=None, section=None
        )

    def project_snapshot(
        self, events: tuple[TranscriptEvent, ...], registry: "ToolRegistry"
    ) -> ProjectionResult:
        return self._project_owned_snapshot(
            copy.deepcopy(events), registry, event_indices=None, section=None
        )

    def project_compacted(
        self,
        prelude: tuple[TranscriptEvent, ...],
        tail: tuple[TranscriptEvent, ...],
        section: CompressedHistorySection,
        registry: "ToolRegistry",
    ) -> ProjectionResult:
        """Render one capsule followed by an untouched chronological tail."""

        items = tuple(copy.deepcopy((*prelude, *tail)))
        indices = tuple(range(len(prelude))) + tuple(
            range(section.through_event_index, section.through_event_index + len(tail))
        )
        return self._project_owned_snapshot(
            items, registry, event_indices=indices, section=section
        )

    def analyze(
        self,
        events: tuple[TranscriptEvent, ...],
        *,
        base_event_index: int,
        keep_recent_decision_windows: int,
        all_events_for_evidence: tuple[TranscriptEvent, ...] | None = None,
        max_compactable_windows: int | None = None,
        registry: "ToolRegistry | None" = None,
        current_task_state: dict[str, Any] | None = None,
    ) -> ProjectionPlan:
        """Choose the oldest complete prefix without invoking a model."""

        owned = copy.deepcopy(events)
        windows = _decision_windows(owned)
        closed = [window for window in windows if window.closed]
        if len(closed) <= keep_recent_decision_windows:
            return _empty_plan(owned, current_task_state)
        eligible = (
            closed[:-keep_recent_decision_windows]
            if keep_recent_decision_windows
            else closed
        )
        if max_compactable_windows is not None:
            if max_compactable_windows < 1:
                return _empty_plan(owned, current_task_state)
            eligible = eligible[:max_compactable_windows]
        cut_local = eligible[-1].end
        if cut_local <= 0:
            return _empty_plan(owned, current_task_state)
        semantic_windows = _compression_windows(
            owned, eligible, cut_local, base_event_index, registry
        )
        full_events = copy.deepcopy(all_events_for_evidence or owned)
        cut_global = base_event_index + cut_local
        full_prefix = full_events[:cut_global] if all_events_for_evidence else owned[:cut_local]
        prefix_base = 0 if all_events_for_evidence else base_event_index
        selected_by_module, _ = self._selection(full_prefix)
        prefix_media = _deduplicate_visual_candidates(
            list(
                _visual_candidates(
                    full_prefix,
                    tuple(range(prefix_base, prefix_base + len(full_prefix))),
                    selected_by_module,
                )
            )
        )
        serialized = [_event_dict(item) for item in full_prefix]
        digest = hashlib.sha256(
            json.dumps(serialized, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ProjectionPlan(
            compactable_end_event_index=cut_global,
            windows=semantic_windows,
            evidence_catalog=_evidence_catalog(full_prefix),
            deterministic_facts=_deterministic_facts(full_prefix, semantic_windows),
            exact_loaded_knowledge=_exact_loaded_knowledge(full_prefix),
            prefix_media=tuple(prefix_media),
            current_state_hint=copy.deepcopy(current_task_state or {}),
            instruction=_instruction(full_events),
            source_digest=digest,
        )

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
                for sequence, (observation_id, checkpoint) in enumerate(observations.items())
            )
            output = tuple(module.select_history_images(candidates, copy.deepcopy(config)))
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
                "selected_observation_ids": [key for key in observations if key in chosen],
                "omitted_count": len(observations) - len(chosen),
            }
        return selected, {
            "schema_version": HISTORY_PROJECTION_SCHEMA,
            "observation_count": len(observations),
            "policies": policies,
            "observation_image_count": 0,
            "tool_image_count": 0,
            "total_image_count": 0,
            "missing_images": [],
        }

    def _selected_visual_candidates(
        self,
        items: tuple[TranscriptEvent, ...],
        event_indices: tuple[int, ...],
        section: CompressedHistorySection | None,
    ) -> tuple[_VisualSelection, ...]:
        module_selected, _ = self._selection(items)
        candidates = [*(section.prefix_media if section else ())]
        candidates.extend(_visual_candidates(items, event_indices, module_selected))
        candidates = _deduplicate_visual_candidates(candidates)
        if not self.media_policy:
            return tuple(_VisualSelection(item, "unbounded") for item in candidates)
        current_observation_id = _latest_observation_id(items)
        cited_refs = _summary_evidence_refs(section) if section else frozenset()
        return _select_visual_candidates(
            candidates,
            self.media_policy,
            compacted_through=(section.through_event_index if section else None),
            current_observation_id=current_observation_id,
            cited_refs=cited_refs,
        )

    def _project_owned_snapshot(
        self,
        items: tuple[TranscriptEvent, ...],
        registry: "ToolRegistry",
        *,
        event_indices: tuple[int, ...] | None,
        section: CompressedHistorySection | None,
    ) -> ProjectionResult:
        selected_by_module, audit = self._selection(items)
        indices = event_indices or tuple(range(len(items)))
        selected_visual = self._selected_visual_candidates(items, indices, section)
        selected_keys = {item.candidate.key for item in selected_visual}
        audit["selected_visual_evidence_refs"] = [
            item.candidate.evidence_ref for item in selected_visual
        ]
        audit["selected_visual_evidence"] = [
            {
                "event_index": item.candidate.event_index,
                "evidence_ref": item.candidate.evidence_ref,
                "observation_id": item.candidate.observation_id,
                "frame_id": item.candidate.frame_id,
                "source_kind": item.candidate.source_kind,
                "selection_tier": item.tier,
                "ref": item.candidate.media_ref.get("ref"),
                "sha256": item.candidate.media_ref.get("sha256"),
            }
            for item in selected_visual
        ]
        if self.media_policy:
            audit["global_media_policy"] = copy.deepcopy(self.media_policy)
        if section is not None:
            audit["compaction"] = {
                "generation": section.generation,
                "through_event_index": section.through_event_index,
                "source_digest": section.source_digest,
            }
        messages: list[dict[str, Any]] = []
        envelope: dict[str, Any] = {}
        first_state_index_by_observation: dict[str, int] = {}
        for local_index, event in enumerate(items):
            if event.kind != "state_checkpoint":
                continue
            checkpoint = _payload_dict(event.payload)
            observation_id = checkpoint.get("observation_id")
            if observation_id:
                first_state_index_by_observation.setdefault(
                    str(observation_id), indices[local_index]
                )
        newest_observation = next(reversed(first_state_index_by_observation), None)
        seen_media: set[str] = set()
        prefix_visual = (
            [
                item.candidate
                for item in selected_visual
                if item.candidate.event_index < section.through_event_index
            ]
            if section is not None
            else []
        )
        for local_index, event in enumerate(items):
            global_index = indices[local_index]
            if event.kind == "instruction_envelope":
                envelope = _payload_dict(event.payload)
                continue
            if event.kind == "user_instruction":
                messages.append({"role": "user", "content": event.payload["instruction"]})
                context_message = self.assembler.instruction_context_message(envelope)
                if context_message is not None:
                    messages.append(context_message)
                if section is not None:
                    messages.append(self.assembler.compressed_history_message(section))
                    for entry in prefix_visual:
                        if entry.key in seen_media:
                            continue
                        try:
                            messages.append(
                                self.assembler.history_evidence_message(entry)
                            )
                        except ValueError as exc:
                            audit["missing_images"].append(
                                {
                                    "observation_id": entry.observation_id,
                                    "evidence_ref": entry.evidence_ref,
                                    "reason": str(exc),
                                }
                            )
                            continue
                        seen_media.add(entry.key)
                        counter = (
                            "tool_image_count"
                            if entry.source_kind == "tool_evidence"
                            else "observation_image_count"
                        )
                        audit[counter] += 1
                continue
            if event.kind == "state_checkpoint":
                checkpoint = _payload_dict(event.payload)
                observation_id = str(checkpoint.get("observation_id") or "unknown")
                include_media = first_state_index_by_observation.get(observation_id) == global_index
                inputs = _selected_checkpoint_inputs(
                    checkpoint,
                    observation_id=observation_id,
                    include_media=include_media,
                    selected=selected_by_module,
                    selected_visual=selected_keys,
                    seen_media=seen_media,
                )
                timeline_role = "CURRENT" if observation_id == newest_observation else "HISTORY"
                try:
                    content = self.assembler.checkpoint_content(
                        checkpoint, model_inputs=inputs, timeline_role=timeline_role
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
                body = [prefix, *content] if isinstance(content, list) else [
                    prefix,
                    {"type": "text", "text": content},
                ]
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
                        "content": json.dumps(result.to_provider_dict(), ensure_ascii=False),
                    }
                )
                evidence = self._tool_evidence_message(
                    result, selected_keys=selected_keys, seen_media=seen_media
                )
                if evidence is not None:
                    messages.append(evidence)
                    audit["tool_image_count"] += sum(
                        part.get("type") == "image_url" for part in evidence["content"]
                    )
                continue
            if event.kind == "protocol_feedback":
                messages.append({"role": "user", "content": event.payload["message"]})
        audit["total_image_count"] = (
            audit["observation_image_count"]
            + audit["tool_image_count"]
        )
        return ProjectionResult(messages, audit)

    def _tool_evidence_message(
        self,
        result: ToolResult,
        *,
        selected_keys: set[str],
        seen_media: set[str],
    ) -> dict[str, Any] | None:
        content: list[dict[str, Any]] = []
        for item in result.model_inputs:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            raw_ref = item.get("media_ref")
            if not isinstance(raw_ref, dict):
                continue
            observation_id = str(raw_ref.get("observation_id") or "unknown")
            identity = _media_identity(observation_id, raw_ref)
            if identity not in selected_keys or identity in seen_media:
                continue
            try:
                prepared = self.assembler.media_hydrator.hydrate(raw_ref)
            except ValueError:
                continue
            seen_media.add(identity)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"TOOL EVIDENCE tool_call_id={result.tool_call_id} "
                            f"observation_id={observation_id}. "
                            f"{raw_ref.get('usage') or 'Verify this tool evidence.'}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": prepared.data_url}},
                ]
            )
        return {"role": "user", "content": content} if content else None


def _empty_plan(
    events: tuple[TranscriptEvent, ...],
    current_task_state: dict[str, Any] | None,
) -> ProjectionPlan:
    return ProjectionPlan(
        compactable_end_event_index=None,
        windows=(),
        evidence_catalog=(),
        deterministic_facts={},
        exact_loaded_knowledge=(),
        prefix_media=(),
        current_state_hint=copy.deepcopy(current_task_state or {}),
        instruction=_instruction(events),
        source_digest=None,
    )


def _decision_windows(events: tuple[TranscriptEvent, ...]) -> tuple[_DecisionWindow, ...]:
    assistants = [index for index, event in enumerate(events) if event.kind == "assistant_turn"]
    windows: list[_DecisionWindow] = []
    for position, assistant_index in enumerate(assistants):
        start = 0 if position == 0 else assistant_index
        end = assistants[position + 1] if position + 1 < len(assistants) else len(events)
        turn = events[assistant_index].payload
        call_ids = {call.tool_call_id for call in getattr(turn, "tool_calls", [])}
        result_ids = {
            event.payload.tool_call_id
            for event in events[assistant_index + 1 : end]
            if event.kind == "tool_result"
        }
        windows.append(
            _DecisionWindow(start, end, assistant_index, bool(call_ids) and call_ids <= result_ids)
        )
    return tuple(windows)


def _compression_windows(
    events: tuple[TranscriptEvent, ...],
    windows: list[_DecisionWindow],
    cut_local: int,
    base_event_index: int,
    registry: "ToolRegistry | None",
) -> tuple[CompressionWindow, ...]:
    result: list[CompressionWindow] = []
    for number, window in enumerate(windows, start=1):
        if not window.closed or window.end > cut_local:
            continue
        payloads = tuple(
            payload
            for event in events[window.start : window.end]
            for payload in [_event_for_compression(event, registry)]
            if payload is not None
        )
        result.append(
            CompressionWindow(
                window_id=f"decision_window_{number:04d}",
                start_event_index=base_event_index + window.start,
                end_event_index=base_event_index + window.end,
                events=payloads,
            )
        )
    return tuple(result)


def _event_for_compression(
    event: TranscriptEvent, registry: "ToolRegistry | None"
) -> dict[str, Any] | None:
    """Build the narrow semantic view used only by the summary model."""

    if event.kind == "assistant_turn":
        turn: AssistantTurn = event.payload
        calls = []
        for call in turn.tool_calls:
            calls.append(
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                    "effect": _tool_effect(registry, call.tool_name),
                    "args": _compression_call_args(call.tool_name, call.args),
                }
            )
        return {
            "kind": event.kind,
            "payload": {
                "public_intent": turn.public_intent,
                "tool_calls": calls,
            },
        }
    if event.kind == "tool_result":
        result: ToolResult = event.payload
        payload: dict[str, Any] = {
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "effect": _tool_effect(registry, result.tool_name),
            "status": result.status,
            "message": result.message,
            "result_ref": result.result_ref,
            "requested_args": copy.deepcopy(result.requested_args),
            "executed_args": copy.deepcopy(result.executed_args),
            "safety_interventions": copy.deepcopy(result.safety_interventions),
        }
        return {"kind": event.kind, "payload": payload}
    if event.kind == "state_checkpoint":
        payload = _payload_dict(event.payload)
        return {
            "kind": event.kind,
            "payload": {
                "observation_id": payload.get("observation_id"),
                "reason": payload.get("reason"),
                "motion": _checkpoint_motion(payload),
            },
        }
    if event.kind == "protocol_feedback":
        return {"kind": event.kind, "payload": {"message": event.payload.get("message")}}
    if event.kind == "provider_turn_rejected":
        payload = _payload_dict(event.payload)
        return {
            "kind": event.kind,
            "payload": {
                key: copy.deepcopy(payload.get(key))
                for key in ("error", "content_present", "tool_calls")
                if key in payload
            },
        }
    return None


def _tool_effect(registry: "ToolRegistry | None", tool_name: str) -> str | None:
    if registry is None:
        return None
    try:
        return registry.get(tool_name).runtime.effect
    except KeyError:
        return None


def _compression_call_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Remove repeated plan bodies while preserving ordinary tool arguments."""

    if tool_name != "runtime.update_plan":
        return copy.deepcopy(args)
    # The transient CURRENT TASK STATE retains the authoritative current plan.
    # Replaying every historical full step list wastes the semantic budget
    # without adding execution evidence.
    return {
        key: copy.deepcopy(value)
        for key, value in args.items()
        if key not in {"steps", "plan", "task_plan"}
    }


def _checkpoint_motion(checkpoint: dict[str, Any]) -> dict[str, Any]:
    for fragment in checkpoint.get("fragments", []):
        if not isinstance(fragment, dict) or fragment.get("name") != "motion":
            continue
        content = fragment.get("content")
        if not isinstance(content, dict):
            return {}
        return {
            key: copy.deepcopy(content.get(key))
            for key in (
                "instruction_start_relative_uav_motion",
                "last_window_uav_motion",
            )
            if key in content
        }
    return {}


def _event_dict(event: TranscriptEvent) -> dict[str, Any]:
    return {"kind": event.kind, "payload": _payload_dict(event.payload)}


def _instruction(events: tuple[TranscriptEvent, ...]) -> str:
    for event in events:
        if event.kind == "user_instruction":
            value = event.payload.get("instruction")
            return value if isinstance(value, str) else ""
    return ""


def _evidence_catalog(
    events: tuple[TranscriptEvent, ...]
) -> tuple[EvidenceCatalogEntry, ...]:
    """Assign deterministic aliases to every citable prefix identity.

    Rejected provider turns are not replayed as assistant messages, but their
    ToolCall IDs are legitimate evidence for explaining protocol failures.
    Non-evidence implementation IDs such as checkpoints, windows, and artifact
    paths deliberately never enter this catalog.
    """

    refs: dict[str, str] = {}

    def add(value: Any, kind: str) -> None:
        if value is None:
            return
        canonical = str(value)
        if canonical:
            refs.setdefault(canonical, kind)

    for event in events:
        if event.kind == "state_checkpoint":
            add(_payload_dict(event.payload).get("observation_id"), "observation")
        elif event.kind == "assistant_turn":
            for call in event.payload.tool_calls:
                add(call.tool_call_id, "tool_call")
        elif event.kind == "tool_result":
            add(event.payload.tool_call_id, "tool_call")
            add(event.payload.result_ref, "tool_result")
        elif event.kind == "provider_turn_rejected":
            payload = _payload_dict(event.payload)
            tool_calls = payload.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                add(call.get("tool_call_id") or call.get("id"), "rejected_tool_call")
    return tuple(
        EvidenceCatalogEntry(
            alias=f"E{index:04d}",
            canonical_ref=canonical_ref,
            kind=kind,
        )
        for index, (canonical_ref, kind) in enumerate(refs.items(), start=1)
    )


def _deterministic_facts(
    events: tuple[TranscriptEvent, ...], windows: tuple[CompressionWindow, ...]
) -> dict[str, Any]:
    tool_status_counts: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    physical_windows = 0
    boundary_checkpoint: dict[str, Any] = {}
    for event in events:
        if event.kind == "state_checkpoint":
            checkpoint = _payload_dict(event.payload)
            boundary_checkpoint = {
                "observation_id": checkpoint.get("observation_id"),
                "reason": checkpoint.get("reason"),
                "motion": _checkpoint_motion(checkpoint),
            }
            if checkpoint.get("reason") == "physical_window_completed":
                physical_windows += 1
        elif event.kind == "tool_result":
            result: ToolResult = event.payload
            by_status = tool_status_counts.setdefault(result.tool_name, {})
            by_status[result.status] = by_status.get(result.status, 0) + 1
            if result.status != "ok":
                failures.append(
                    {
                        "tool_call_id": result.tool_call_id,
                        "tool_name": result.tool_name,
                        "status": result.status,
                        "message": result.message,
                        "result_ref": result.result_ref,
                    }
                )
            for intervention in result.safety_interventions:
                interventions.append(
                    {"tool_call_id": result.tool_call_id, **copy.deepcopy(intervention)}
                )
    return {
        "compacted_decision_window_count": sum(
            event.kind == "assistant_turn" for event in events
        ),
        "physical_window_count": physical_windows,
        "tool_status_counts": tool_status_counts,
        "latest_failures": failures[-5:],
        "latest_safety_interventions": interventions[-5:],
        "boundary_checkpoint": boundary_checkpoint,
    }


def _exact_loaded_knowledge(
    events: tuple[TranscriptEvent, ...]
) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.kind != "tool_result":
            continue
        result: ToolResult = event.payload
        if result.tool_name != "knowledge.load" or result.status != "ok":
            continue
        payload = result.to_provider_dict()
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if digest not in seen:
            seen.add(digest)
            values.append(payload)
    return tuple(values)


def _visual_candidates(
    events: tuple[TranscriptEvent, ...],
    event_indices: tuple[int, ...],
    selected_by_module: dict[str, set[str]],
) -> tuple[HistoryMediaEntry, ...]:
    values: list[HistoryMediaEntry] = []
    for local_index, event in enumerate(events):
        event_index = event_indices[local_index]
        if event.kind == "state_checkpoint":
            checkpoint = _payload_dict(event.payload)
            observation_id = str(checkpoint.get("observation_id") or "unknown")
            for fragment in checkpoint.get("fragments", []):
                if not isinstance(fragment, dict):
                    continue
                slot = str(fragment.get("name") or "")
                if slot in selected_by_module and observation_id not in selected_by_module[slot]:
                    continue
                for item in fragment.get("media_refs", []):
                    ref = item.get("media_ref") if isinstance(item, dict) else None
                    if not isinstance(item, dict) or item.get("type") != "image" or not isinstance(ref, dict):
                        continue
                    values.append(
                        _candidate(
                            ref,
                            event_index,
                            evidence_ref=observation_id,
                            source_kind="observation",
                            checkpoint_reason=str(checkpoint.get("reason") or ""),
                        )
                    )
        elif event.kind == "tool_result":
            result: ToolResult = event.payload
            for item in result.model_inputs:
                ref = item.get("media_ref") if isinstance(item, dict) else None
                if not isinstance(item, dict) or item.get("type") != "image" or not isinstance(ref, dict):
                    continue
                values.append(
                    _candidate(
                        ref,
                        event_index,
                        evidence_ref=result.tool_call_id,
                        source_kind="tool_evidence",
                        tool_call_id=result.tool_call_id,
                    )
                )
    return tuple(values)


def _candidate(
    ref: dict[str, Any],
    event_index: int,
    *,
    evidence_ref: str,
    source_kind: str,
    checkpoint_reason: str | None = None,
    tool_call_id: str | None = None,
) -> HistoryMediaEntry:
    observation_id = str(ref.get("observation_id") or "unknown")
    return HistoryMediaEntry(
        event_index=event_index,
        evidence_ref=evidence_ref,
        observation_id=observation_id,
        frame_id=str(ref.get("frame_id") or "unknown"),
        media_ref=copy.deepcopy(ref),
        source_kind=source_kind,
        checkpoint_reason=checkpoint_reason,
        tool_call_id=tool_call_id,
    )


def _deduplicate_visual_candidates(
    candidates: list[HistoryMediaEntry],
) -> list[HistoryMediaEntry]:
    by_key: dict[str, HistoryMediaEntry] = {}
    for candidate in sorted(candidates, key=lambda item: item.event_index):
        by_key.setdefault(candidate.key, candidate)
    return list(by_key.values())


def _select_visual_candidates(
    candidates: list[HistoryMediaEntry],
    policy: dict[str, Any],
    *,
    compacted_through: int | None,
    current_observation_id: str | None,
    cited_refs: frozenset[str],
) -> tuple[_VisualSelection, ...]:
    """Choose visual anchors without letting tool evidence crowd out RGB history."""

    if not candidates:
        return ()
    budget = int(policy["max_request_images"])
    chosen: dict[str, str] = {}

    def choose(candidate: HistoryMediaEntry, tier: str) -> None:
        if len(chosen) < budget:
            chosen.setdefault(candidate.key, tier)

    observations = [
        item for item in candidates if item.source_kind == "observation"
    ]
    # Context module order places the primary image before auxiliary visual
    # modules. Keep one representative per observation so a single checkpoint
    # cannot consume the entire cross-source request budget.
    representatives: dict[str, HistoryMediaEntry] = {}
    for item in observations:
        representatives.setdefault(item.observation_id, item)
    observation_values = list(representatives.values())

    if policy.get("pin_instruction_start"):
        instruction_start = next(
            (
                item
                for item in observation_values
                if item.checkpoint_reason == "instruction_start"
            ),
            None,
        )
        if instruction_start is not None:
            choose(instruction_start, "instruction_start")

    current = next(
        (
            item
            for item in reversed(observation_values)
            if item.observation_id == current_observation_id
        ),
        None,
    )
    if policy.get("pin_current") and current is not None:
        choose(current, "current")

    recent_count = int(policy["keep_recent_observations"])
    recent_pool = [
        item
        for item in observation_values
        if current is None or item.key != current.key
    ]
    for item in recent_pool[-recent_count:]:
        choose(item, "recent_observation")

    eligible_tools = [
        item
        for item in candidates
        if item.source_kind == "tool_evidence"
        and (
            compacted_through is None
            or item.event_index >= compacted_through
            or item.evidence_ref in cited_refs
        )
    ]
    tool_limit = min(
        int(policy["max_tool_evidence_images"]),
        max(0, budget - len(chosen)),
    )
    for item in sorted(
        eligible_tools, key=lambda candidate: candidate.event_index, reverse=True
    )[:tool_limit]:
        tier = (
            "capsule_tool_evidence"
            if compacted_through is not None
            and item.event_index < compacted_through
            else "tail_tool_evidence"
        )
        choose(item, tier)

    remaining_slots = budget - len(chosen)
    older_observations = [
        item for item in observation_values if item.key not in chosen
    ]
    for item in _log_spaced_candidates(older_observations, remaining_slots):
        choose(item, "log_history")

    return tuple(
        _VisualSelection(item, chosen[item.key])
        for item in candidates
        if item.key in chosen
    )


def _log_spaced_candidates(
    candidates: list[HistoryMediaEntry], slots: int
) -> tuple[HistoryMediaEntry, ...]:
    """Return recent-to-old log-spaced identities, including both extremes."""

    if slots <= 0 or not candidates:
        return ()
    if slots >= len(candidates):
        return tuple(candidates)
    if slots == 1:
        return (candidates[-1],)
    total = len(candidates)
    ages: list[int] = []
    for index in range(slots):
        age = math.floor(
            math.exp(index / (slots - 1) * math.log(total)) - 1 + 0.5
        )
        age = min(total - 1, max(age, ages[-1] + 1 if ages else 0))
        ages.append(age)
    return tuple(candidates[total - age - 1] for age in ages)


def _media_identity(observation_id: str, ref: dict[str, Any]) -> str:
    return f"{observation_id}:{ref.get('sha256') or ref.get('ref') or ''}"


def _latest_observation_id(events: tuple[TranscriptEvent, ...]) -> str | None:
    for event in reversed(events):
        if event.kind != "state_checkpoint":
            continue
        observation_id = _payload_dict(event.payload).get("observation_id")
        if observation_id:
            return str(observation_id)
    return None


def _summary_evidence_refs(
    section: CompressedHistorySection,
) -> frozenset[str]:
    refs: set[str] = set()
    for items in section.summary.to_dict().values():
        for item in items:
            values = item.get("evidence_refs")
            if isinstance(values, list):
                refs.update(str(value) for value in values if value)
    return frozenset(refs)


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
        for item in fragment.get("media_refs", [])
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
    selected_visual: set[str],
    seen_media: set[str],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for fragment in checkpoint.get("fragments", []):
        if not isinstance(fragment, dict):
            continue
        slot = str(fragment.get("name") or "")
        chosen = slot not in selected or observation_id in selected[slot]
        for item in fragment.get("media_refs", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image":
                inputs.append(copy.deepcopy(item))
                continue
            ref = item.get("media_ref")
            if not include_media or not chosen or not isinstance(ref, dict):
                continue
            identity = _media_identity(observation_id, ref)
            if identity not in selected_visual or identity in seen_media:
                continue
            if str(ref.get("observation_id") or observation_id) != observation_id:
                continue
            seen_media.add(identity)
            inputs.append(copy.deepcopy(item))
    return inputs
