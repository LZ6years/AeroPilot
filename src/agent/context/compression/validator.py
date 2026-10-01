"""Hard runtime gates for model-authored history summaries."""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.context.compression.contracts import (
    CompressionDraft,
    CompressionRequest,
    CompressionValidationResult,
)


_FORBIDDEN_KEYS = frozenset(
    {"completed", "current_pose", "current_observation_id", "current_state"}
)
_COMPLETION_CLAIM = re.compile(
    r"\b(?:task|mission)\s+(?:is\s+)?(?:complete|completed|done)\b"
    r"|任务已完成|已完成任务",
    re.IGNORECASE,
)


class CompressionValidator:
    """Validate structure and provenance without pretending to judge semantics."""

    def __init__(self, summary_max_tokens: int) -> None:
        self.summary_max_tokens = summary_max_tokens

    def validate(
        self, draft: CompressionDraft, request: CompressionRequest
    ) -> CompressionValidationResult:
        errors: list[str] = []
        encoded = json.dumps(draft.to_dict(), ensure_ascii=False).encode("utf-8")
        if len(encoded) / 3.5 > self.summary_max_tokens:
            errors.append("summary_exceeds_token_budget")
        for section_name, items in draft.to_dict().items():
            for index, item in enumerate(items):
                unknown = set(item) - {"summary", "evidence_refs"}
                if unknown:
                    errors.append(f"{section_name}[{index}]_unknown_fields")
                summary = item.get("summary")
                if not isinstance(summary, str) or not summary.strip():
                    errors.append(f"{section_name}[{index}]_summary_invalid")
                elif _COMPLETION_CLAIM.search(summary):
                    errors.append(f"{section_name}[{index}]_completion_claim")
                if _contains_forbidden_key(item):
                    errors.append(f"{section_name}[{index}]_forbidden_state")
                refs = item.get("evidence_refs", [])
                if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                    errors.append(f"{section_name}[{index}]_refs_invalid")
                    continue
                for ref in sorted(set(refs) - request.allowed_evidence_refs):
                    errors.append(f"{section_name}[{index}]_unknown_ref:{ref}")
        unique_errors = tuple(dict.fromkeys(errors))
        normalized = None if unique_errors else _canonicalize_draft(draft, request)
        return CompressionValidationResult(
            not unique_errors,
            unique_errors,
            normalized,
        )

def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(_FORBIDDEN_KEYS & value.keys()) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _canonicalize_draft(
    draft: CompressionDraft, request: CompressionRequest
) -> CompressionDraft:
    """Resolve request-local aliases before a Capsule or audit can retain them."""

    mapping = request.alias_to_canonical
    if not mapping:
        return draft
    pattern = re.compile(
        "|".join(re.escape(item) for item in sorted(mapping, key=len, reverse=True))
    )

    def translate(value: Any) -> Any:
        if isinstance(value, str):
            return pattern.sub(lambda match: mapping[match.group(0)], value)
        if isinstance(value, dict):
            return {key: translate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [translate(item) for item in value]
        return value

    return CompressionDraft.from_dict(translate(draft.to_dict()))
