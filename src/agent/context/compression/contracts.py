"""Pure contracts for task-local history compression.

The durable transcript and Runtime never appear in these types. A projector
adapts immutable event snapshots into a ``CompressionRequest``; a stateless
compactor returns model-authored text; and the main planner thread alone may
commit a validated section into ``HistoryProjectionState``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class HistoryCompressionConfig:
    """Validated request-projection and background-compactor limits."""

    enabled: bool = True
    prefetch_trigger_tokens: int = 48_000
    soft_trigger_tokens: int = 64_000
    target_tokens: int = 32_000
    max_compactor_input_tokens: int = 32_000
    summary_max_tokens: int = 4_096
    keep_recent_decision_windows: int = 4
    compactor_timeout_sec: float = 300.0
    compactor_timeout_retries: int = 0
    compactor_connection_retries: int = 1
    max_request_images: int = 20
    max_tool_evidence_images: int = 4
    keep_recent_observations: int = 3
    pin_instruction_start: bool = True
    pin_current: bool = True

    def __post_init__(self) -> None:
        for name in ("enabled", "pin_instruction_start", "pin_current"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        positive = (
            "prefetch_trigger_tokens",
            "soft_trigger_tokens",
            "target_tokens",
            "max_compactor_input_tokens",
            "summary_max_tokens",
            "keep_recent_decision_windows",
            "max_request_images",
            "keep_recent_observations",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.target_tokens < self.prefetch_trigger_tokens <= self.soft_trigger_tokens:
            raise ValueError(
                "token thresholds must satisfy target < prefetch <= soft trigger"
            )
        if self.summary_max_tokens > 4_096:
            raise ValueError("summary_max_tokens must not exceed 4096")
        if (
            isinstance(self.compactor_timeout_sec, bool)
            or not isinstance(self.compactor_timeout_sec, (int, float))
            or self.compactor_timeout_sec <= 0
        ):
            raise ValueError("compactor_timeout_sec must be positive")
        for name in ("compactor_timeout_retries", "compactor_connection_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            type(self.max_tool_evidence_images) is not int
            or self.max_tool_evidence_images < 0
        ):
            raise ValueError("max_tool_evidence_images must be a non-negative integer")
        if self.max_tool_evidence_images > self.max_request_images:
            raise ValueError("max_tool_evidence_images must not exceed max_request_images")
        minimum_observations = (
            self.keep_recent_observations
            + int(self.pin_instruction_start)
            + int(self.pin_current)
        )
        if self.max_request_images < minimum_observations:
            raise ValueError("max_request_images is too small for pinned media")

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "HistoryCompressionConfig":
        root = value or {}
        if not isinstance(root, dict):
            raise ValueError("history_projection config must be an object")
        unknown = set(root) - {"compression", "media"}
        if unknown:
            raise ValueError(f"unknown history_projection fields: {sorted(unknown)}")
        compression = root.get("compression") or {}
        media = root.get("media") or {}
        if not isinstance(compression, dict) or not isinstance(media, dict):
            raise ValueError("compression and media config must be objects")
        allowed_compression = {
            "enabled", "prefetch_trigger_tokens", "soft_trigger_tokens",
            "target_tokens", "max_compactor_input_tokens", "summary_max_tokens",
            "keep_recent_decision_windows", "compactor_timeout_sec",
            "compactor_timeout_retries", "compactor_connection_retries",
        }
        allowed_media = {
            "max_request_images", "max_tool_evidence_images",
            "keep_recent_observations",
            "pin_instruction_start", "pin_current",
        }
        if set(compression) - allowed_compression:
            raise ValueError("unknown history compression option")
        if set(media) - allowed_media:
            raise ValueError("unknown history media option")
        return cls(**compression, **media)


@dataclass(frozen=True)
class CompressionWindow:
    """One complete assistant/tool/checkpoint unit prepared by the projector."""

    window_id: str
    start_event_index: int
    end_event_index: int
    events: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "events", tuple(copy.deepcopy(dict(item)) for item in self.events)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_event_index": self.start_event_index,
            "end_event_index": self.end_event_index,
            "events": copy.deepcopy(list(self.events)),
        }


@dataclass(frozen=True)
class EvidenceCatalogEntry:
    """One request-local alias for a durable canonical evidence reference.

    Aliases are model-facing only. Canonical references remain the identities
    stored in the Transcript, accepted Capsule, and human audit artifacts.
    """

    alias: str
    canonical_ref: str
    kind: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.alias, self.canonical_ref, self.kind)
        ):
            raise ValueError("evidence catalog fields must be non-empty strings")


@dataclass(frozen=True)
class HistoryMediaEntry:
    """Binary-free identity for media that may outlive its raw history event."""

    event_index: int
    evidence_ref: str
    observation_id: str
    frame_id: str
    media_ref: dict[str, Any]
    source_kind: str
    checkpoint_reason: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("history media event_index must be a non-negative integer")
        if self.source_kind not in {"observation", "tool_evidence"}:
            raise ValueError("history media source_kind is invalid")
        if any(
            not isinstance(value, str) or not value
            for value in (self.evidence_ref, self.observation_id, self.frame_id)
        ):
            raise ValueError("history media identity fields must be non-empty strings")
        object.__setattr__(self, "media_ref", copy.deepcopy(dict(self.media_ref)))

    @property
    def key(self) -> str:
        return (
            f"{self.observation_id}:"
            f"{self.media_ref.get('sha256') or self.media_ref.get('ref') or ''}"
        )


_SUMMARY_FIELDS = (
    "mission_progress",
    "decisions_and_outcomes",
    "failures_and_recovery",
)


@dataclass(frozen=True)
class CompressionDraft:
    """Model-authored semantic summary before projector-owned validation."""

    mission_progress: tuple[dict[str, Any], ...] = ()
    decisions_and_outcomes: tuple[dict[str, Any], ...] = ()
    failures_and_recovery: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in _SUMMARY_FIELDS:
            object.__setattr__(
                self, name, tuple(copy.deepcopy(dict(item)) for item in getattr(self, name))
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CompressionDraft":
        if not isinstance(value, dict) or set(value) != set(_SUMMARY_FIELDS):
            raise ValueError("compression draft has an invalid top-level schema")
        fields: dict[str, tuple[dict[str, Any], ...]] = {}
        for name in _SUMMARY_FIELDS:
            items = value[name]
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise ValueError(f"compression draft field {name} must be an object list")
            fields[name] = tuple(items)
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {name: copy.deepcopy(list(getattr(self, name))) for name in _SUMMARY_FIELDS}


@dataclass(frozen=True)
class CompressionRequest:
    """Self-contained, image-free semantic input with a request-local catalog.

    Projector-prepared windows and facts retain canonical references internally.
    ``ModelHistoryCompactor`` translates them to aliases at the final
    model-facing boundary, and the validator translates accepted output back.
    """

    source_digest: str
    instruction: str
    windows: tuple[CompressionWindow, ...]
    evidence_catalog: tuple[EvidenceCatalogEntry, ...]
    current_state_hint: dict[str, Any]
    deterministic_facts: dict[str, Any]
    previous_summary: CompressionDraft | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(copy.deepcopy(self.windows)))
        object.__setattr__(self, "evidence_catalog", tuple(self.evidence_catalog))
        object.__setattr__(self, "current_state_hint", copy.deepcopy(self.current_state_hint))
        object.__setattr__(self, "deterministic_facts", copy.deepcopy(self.deterministic_facts))
        aliases = [entry.alias for entry in self.evidence_catalog]
        canonical_refs = [entry.canonical_ref for entry in self.evidence_catalog]
        if len(aliases) != len(set(aliases)):
            raise ValueError("evidence catalog aliases must be unique")
        if len(canonical_refs) != len(set(canonical_refs)):
            raise ValueError("evidence catalog canonical refs must be unique")

    @property
    def allowed_evidence_refs(self) -> frozenset[str]:
        """Return the model-facing alias allowlist for compatibility helpers."""

        return frozenset(entry.alias for entry in self.evidence_catalog)

    @property
    def alias_to_canonical(self) -> dict[str, str]:
        return {entry.alias: entry.canonical_ref for entry in self.evidence_catalog}

    @property
    def canonical_to_alias(self) -> dict[str, str]:
        return {entry.canonical_ref: entry.alias for entry in self.evidence_catalog}


@dataclass(frozen=True)
class CompactionCallResult:
    """Model result plus transport-only audit returned by the worker."""

    draft: CompressionDraft
    usage: dict[str, Any] | None = None
    transport_trace: dict[str, Any] | None = None
    finish_reason: str | None = None
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", copy.deepcopy(self.usage))
        object.__setattr__(self, "transport_trace", copy.deepcopy(self.transport_trace))


@dataclass(frozen=True)
class CompressionCandidate:
    """Projector-owned boundary metadata staged beside one background request."""

    request: CompressionRequest
    through_event_index: int
    base_generation: int
    deterministic_facts: dict[str, Any]
    exact_loaded_knowledge: tuple[dict[str, Any], ...] = ()
    prefix_media: tuple[HistoryMediaEntry, ...] = ()
    estimated_input_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "deterministic_facts", copy.deepcopy(self.deterministic_facts))
        object.__setattr__(
            self,
            "exact_loaded_knowledge",
            tuple(copy.deepcopy(dict(item)) for item in self.exact_loaded_knowledge),
        )
        object.__setattr__(self, "prefix_media", tuple(self.prefix_media))

    @property
    def target_generation(self) -> int:
        return self.base_generation + 1


@dataclass(frozen=True)
class CompressedHistorySection:
    """One validated text-only replacement for a contiguous event prefix."""

    generation: int
    through_event_index: int
    source_digest: str
    summary: CompressionDraft
    deterministic_facts: dict[str, Any]
    exact_loaded_knowledge: tuple[dict[str, Any], ...] = ()
    prefix_media: tuple[HistoryMediaEntry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "deterministic_facts", copy.deepcopy(self.deterministic_facts))
        object.__setattr__(
            self,
            "exact_loaded_knowledge",
            tuple(copy.deepcopy(dict(item)) for item in self.exact_loaded_knowledge),
        )
        object.__setattr__(self, "prefix_media", tuple(self.prefix_media))


@dataclass(frozen=True)
class CompressionValidationResult:
    accepted: bool
    errors: tuple[str, ...] = ()
    normalized_draft: CompressionDraft | None = None


@dataclass
class HistoryProjectionState:
    """Mutable request-view state owned by exactly one TaskSession."""

    section: CompressedHistorySection | None = None
    compacted_through_event_index: int = 0
    capsule_generation: int = 0
    source_digest: str | None = None
    last_estimated_tokens: int = 0
    last_actual_prompt_tokens: int | None = None
    calibration_ratio: float = 1.0


class HistoryCompactor(Protocol):
    """Replaceable semantic compressor; implementations must not mutate input."""

    def compact(self, request: CompressionRequest) -> CompactionCallResult: ...

    def estimate_input_tokens(self, request: CompressionRequest) -> int: ...
