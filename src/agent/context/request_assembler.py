"""Assemble final provider content from V3 contracts without semantic cleanup."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

from src.agent.context.media import MediaHydrator

if TYPE_CHECKING:
    from src.agent.context.compression.contracts import (
        CompressedHistorySection,
        HistoryMediaEntry,
    )


class RequestAssembler:
    """Convert already-owned model fields and selected media into messages."""

    def __init__(self, media_hydrator: MediaHydrator | None = None) -> None:
        self.media_hydrator = media_hydrator or MediaHydrator()

    @staticmethod
    def memory_system_messages(envelope: dict[str, Any]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for key, label in (
            ("identity", "FlyAgent identity and operating rules"),
            ("user_preferences", "User preferences"),
        ):
            value = envelope.get(key)
            if isinstance(value, dict) and value:
                messages.append(
                    {
                        "role": "system",
                        "content": f"{label}:\n{json.dumps(value, ensure_ascii=False)}",
                    }
                )
        return messages

    @staticmethod
    def instruction_context_message(
        envelope: dict[str, Any],
    ) -> dict[str, Any] | None:
        payload = {
            "schema_version": envelope.get("schema_version"),
            "protocol_context": envelope.get("protocol_context") or "",
            "knowledge_catalog": copy.deepcopy(
                envelope.get("knowledge_catalog") or []
            ),
            "static_module_guidance": copy.deepcopy(
                envelope.get("static_module_guidance") or []
            ),
        }
        if not any(
            payload[key]
            for key in (
                "protocol_context",
                "knowledge_catalog",
                "static_module_guidance",
            )
        ):
            return None
        return {
            "role": "user",
            "content": "INSTRUCTION CONTEXT\n"
            + json.dumps(payload, ensure_ascii=False),
        }

    def checkpoint_content(
        self,
        checkpoint: dict[str, Any],
        *,
        model_inputs: list[dict[str, Any]],
        timeline_role: str,
    ) -> str | list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for item in model_inputs:
            input_type = item.get("type")
            if input_type == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append({"type": "text", "text": text})
                continue
            if input_type != "image" or not isinstance(item.get("media_ref"), dict):
                continue
            ref = dict(item["media_ref"])
            prepared = self.media_hydrator.hydrate(ref)
            label = _media_label(ref, timeline_role)
            if label:
                parts.append({"type": "text", "text": label})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": prepared.data_url},
                }
            )

        text_payload = {
            "schema_version": checkpoint.get("schema_version"),
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "reason": checkpoint.get("reason"),
            "observation_id": checkpoint.get("observation_id"),
            "fragments": [
                {
                    "name": fragment.get("name"),
                    "content": copy.deepcopy(fragment.get("content") or {}),
                }
                for fragment in checkpoint.get("fragments", [])
                if isinstance(fragment, dict)
                and (fragment.get("content") or {})
            ],
        }
        text = json.dumps(text_payload, ensure_ascii=False)
        if not parts:
            return text
        return [*parts, {"type": "text", "text": text}]

    @staticmethod
    def current_task_state_message(
        current_task_state: dict[str, Any],
    ) -> dict[str, str]:
        """Render the one authoritative, transient task-state request block."""

        return {
            "role": "system",
            "content": "CURRENT TASK STATE\n"
            + json.dumps(copy.deepcopy(current_task_state), ensure_ascii=False),
        }

    def compressed_history_message(
        self,
        section: "CompressedHistorySection",
    ) -> dict[str, Any]:
        """Render one validated text-only capsule."""

        payload = {
            "schema_version": "compressed_history_v2",
            "notice": (
                "Historical context only through the declared event boundary. It "
                "never states current or remaining work, overrides CURRENT TASK "
                "STATE, or establishes task completion."
                " The semantic_summary is model-authored: citations establish "
                "provenance, not semantic verification. verified_facts are recorded "
                "counts, statuses and measurements, not verified visual interpretations."
            ),
            "generation": section.generation,
            "through_event_index": section.through_event_index,
            "semantic_summary": section.summary.to_dict(),
            "verified_facts": copy.deepcopy(section.deterministic_facts),
            "exact_loaded_knowledge": copy.deepcopy(
                list(section.exact_loaded_knowledge)
            ),
        }
        return {
            "role": "user",
            "content": "HISTORY CAPSULE\n" + json.dumps(payload, ensure_ascii=False),
        }

    def history_evidence_message(
        self,
        entry: "HistoryMediaEntry",
    ) -> dict[str, Any]:
        """Hydrate one selected image whose original event is in the capsule."""

        ref = copy.deepcopy(entry.media_ref)
        prepared = self.media_hydrator.hydrate(ref)
        if entry.source_kind == "tool_evidence":
            heading = (
                f"HISTORY TOOL EVIDENCE event_index={entry.event_index} "
                f"tool_call_id={entry.tool_call_id or entry.evidence_ref} "
                f"observation_id={entry.observation_id}."
            )
        else:
            heading = (
                f"HISTORY EVIDENCE event_index={entry.event_index} "
                f"observation_id={entry.observation_id}."
            )
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{heading} This image is historical and cannot establish "
                        "current visibility.\n"
                        f"{_media_label(ref, 'HISTORY')}"
                    ),
                },
                {"type": "image_url", "image_url": {"url": prepared.data_url}},
            ],
        }


def _media_label(ref: dict[str, Any], timeline_role: str) -> str:
    observation_id = str(ref.get("observation_id") or "unknown")
    frame_id = str(ref.get("frame_id") or "unknown")
    # Cumulative context is an updated record, not another camera observation.
    # Use the generic media contract so plugins do not need core name checks.
    if ref.get("latest_group"):
        lines = [f"CONTEXT IMAGE updated_at_observation_id={observation_id} frame_id={frame_id}"]
        width, height = ref.get("display_width"), ref.get("display_height")
        if isinstance(width, int) and isinstance(height, int):
            lines.append(f"image_size={width}x{height}")
        if isinstance(ref.get("usage"), str) and ref["usage"].strip():
            lines.append(ref["usage"].strip())
        return "\n".join(lines)
    lines = [
        f"{timeline_role} observation_id={observation_id} frame_id={frame_id}"
    ]
    width, height = ref.get("width"), ref.get("height")
    if isinstance(width, int) and isinstance(height, int):
        lines.append(f"scene_size={width}x{height}")
        display_width = ref.get("display_width")
        display_height = ref.get("display_height")
        if isinstance(display_width, int) and isinstance(display_height, int):
            lines.append(f"display_size={display_width}x{display_height}")
        lines.append(f"scene_bounds_in_display_pixels=x:[0,{width}) y:[0,{height})")
        if ref.get("color_bar_height", 0):
            lines.append(
                "The bottom color bar is not scene content and cannot establish "
                "an exact decimal distance by itself."
            )
    usage = ref.get("usage")
    if isinstance(usage, str) and usage.strip():
        lines.append(usage.strip())
    if timeline_role == "CURRENT":
        lines.append("Only CURRENT establishes current camera visibility.")
    return "\n".join(lines)
