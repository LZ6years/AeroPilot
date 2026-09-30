"""Stateless, image-free model-backed semantic history compressor."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from src.agent.context.compression.contracts import (
    CompactionCallResult,
    CompressionDraft,
    CompressionRequest,
)
from src.llm import ModelClient, ModelRequest


_SUMMARY_FIELDS = (
    "mission_progress",
    "decisions_and_outcomes",
    "failures_and_recovery",
    "open_work",
)


class ModelHistoryCompactor:
    """Generate only semantic summary fields from projector-prepared history."""

    def __init__(
        self,
        client: ModelClient,
        *,
        summary_max_tokens: int,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.client = client
        self.summary_max_tokens = summary_max_tokens
        path = (
            Path(prompt_path)
            if prompt_path
            else Path(__file__).parent / "prompts" / "history_compaction_system.txt"
        )
        self.system_prompt = path.read_text(encoding="utf-8").strip()

    def compact(self, request: CompressionRequest) -> CompactionCallResult:
        model_request = self._model_request(request)
        started = time.monotonic()
        response = self.client.request(
            model_request,
            error_prefix="History compaction request failed",
        )
        elapsed = time.monotonic() - started
        choices = response.data.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else {}
        finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
        if finish_reason == "length":
            raise ValueError("history compactor output was truncated")
        if response.text is None:
            raise ValueError("history compactor response must contain text")
        draft = CompressionDraft.from_dict(_decode_json_object(response.text))
        return CompactionCallResult(
            draft=draft,
            usage=response.usage,
            transport_trace=response.transport_trace,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            elapsed_seconds=elapsed,
        )

    def estimate_input_tokens(self, request: CompressionRequest) -> int:
        """Estimate the complete binary-free compactor request conservatively."""

        model_request = self._model_request(request)
        encoded = json.dumps(
            {
                "messages": model_request.messages,
                "response_schema": model_request.parameters["response_format"][
                    "json_schema"
                ]["schema"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return max(1, round(len(encoded) / 3.5))

    def _model_request(self, request: CompressionRequest) -> ModelRequest:
        canonical_to_alias = request.canonical_to_alias
        summary_schema = _summary_schema(tuple(request.allowed_evidence_refs))
        payload = {
            "original_task": request.instruction,
            "current_state_hint": _translate_references(
                _without_current_identifiers(request.current_state_hint),
                canonical_to_alias,
            ),
            "previous_summary": (
                _translate_references(
                    request.previous_summary.to_dict(), canonical_to_alias
                )
                if request.previous_summary is not None
                else None
            ),
            # Chronological array order carries the only window information the
            # model needs. Internal window IDs and transcript indices stay out
            # of the semantic request.
            "compactable_windows": [
                _translate_references(list(window.events), canonical_to_alias)
                for window in request.windows
            ],
            "verified_facts": _translate_references(
                request.deterministic_facts, canonical_to_alias
            ),
        }
        return ModelRequest(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            parameters={
                "temperature": 0,
                "max_tokens": self.summary_max_tokens,
                # Structured consolidation does not need a second hidden
                # reasoning budget. The provider default consumed thousands
                # of reasoning tokens and could exceed the 300-second deadline.
                "thinking": {"type": "disabled"},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "uav_history_compaction",
                        "strict": True,
                        "schema": summary_schema,
                    },
                },
            },
        )


def _summary_schema(aliases: tuple[str, ...]) -> dict[str, Any]:
    """Bind evidence_refs to exactly the aliases available in this request."""

    evidence_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string", "enum": sorted(aliases)},
    }
    item_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "evidence_refs": evidence_schema,
        },
        "required": ["summary", "evidence_refs"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            name: {"type": "array", "items": item_schema}
            for name in _SUMMARY_FIELDS
        },
        "required": list(_SUMMARY_FIELDS),
        "additionalProperties": False,
    }


def _translate_references(value: Any, mapping: dict[str, str]) -> Any:
    """Translate reference occurrences without cascading replacement values."""

    if not mapping:
        return value
    pattern = re.compile(
        "|".join(re.escape(item) for item in sorted(mapping, key=len, reverse=True))
    )

    def translate(item: Any) -> Any:
        if isinstance(item, str):
            return pattern.sub(lambda match: mapping[match.group(0)], item)
        if isinstance(item, dict):
            return {key: translate(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [translate(child) for child in item]
        return item

    return translate(value)


def _without_current_identifiers(value: Any) -> Any:
    """Keep CURRENT useful as a hint without making its IDs citable history."""

    if isinstance(value, dict):
        return {
            key: _without_current_identifiers(item)
            for key, item in value.items()
            if key not in {"checkpoint_id", "observation_id", "frame_id"}
        }
    if isinstance(value, list):
        return [_without_current_identifiers(item) for item in value]
    return value


def _decode_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:].lstrip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("history compactor response must be a JSON object")
    return value
