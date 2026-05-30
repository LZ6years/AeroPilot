"""User-facing messages produced by FlyAgent model calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_AGENT_MESSAGE_TYPES = {"progress", "question", "final", "warning"}
VALID_AGENT_MESSAGE_CONFIDENCE = {"high", "medium", "low"}
RAW_THINKING_FIELDS = {"thinking", "raw_thinking", "chain_of_thought", "reasoning"}


@dataclass(frozen=True)
class AgentMessage:
    """A concise, explainable message that can be shown to the human operator."""

    type: str
    content: str
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"
    requires_user_reply: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        payload = dict(data)
        forbidden = sorted(set(payload) & RAW_THINKING_FIELDS)
        if forbidden:
            raise ValueError("AgentMessage must not expose raw thinking fields: " + ", ".join(forbidden))

        message_type = str(payload.get("type", "")).strip()
        if message_type not in VALID_AGENT_MESSAGE_TYPES:
            raise ValueError(f"Unsupported AgentMessage type: {message_type}")

        confidence = str(payload.get("confidence", "medium")).strip() or "medium"
        if confidence not in VALID_AGENT_MESSAGE_CONFIDENCE:
            raise ValueError(f"Unsupported AgentMessage confidence: {confidence}")

        content = str(payload.get("content", "")).strip()
        if not content:
            raise ValueError("AgentMessage content must not be empty")

        evidence_refs = _normalize_evidence_refs(payload.get("evidence_refs", []))

        return cls(
            type=message_type,
            content=content,
            evidence_refs=[dict(item) for item in evidence_refs],
            confidence=confidence,
            requires_user_reply=bool(payload.get("requires_user_reply", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "confidence": self.confidence,
            "requires_user_reply": self.requires_user_reply,
        }


def _normalize_evidence_refs(value: Any) -> list[dict[str, Any]]:
    """Accept model-friendly evidence refs while keeping the persisted shape stable.

    The public AgentMessage schema stores evidence refs as objects. Real model
    responses sometimes provide lightweight string refs such as frame paths; those
    should not abort an otherwise valid runtime step, so we wrap them into a
    neutral object form before persistence.
    """

    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    refs: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            refs.append(dict(item))
            continue
        ref = str(item).strip()
        if ref:
            refs.append({"kind": "model_reference", "ref": ref})
    return refs
