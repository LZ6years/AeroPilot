"""Versioned contracts shared by context modules, Agent projection and Runtime.

The model-facing types in this file deliberately contain no provider payloads,
Runtime services or plugin discovery.  Context modules produce final compact
model fields plus media references; human diagnostics travel in a separate
``FragmentAudit`` so request assembly never needs module-name special cases.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


INSTRUCTION_ENVELOPE_SCHEMA = "instruction_envelope_v2"
STATE_CHECKPOINT_SCHEMA = "state_checkpoint_v2"
TRANSCRIPT_SCHEMA = "agent_transcript_v2"
HISTORY_PROJECTION_SCHEMA = "history_projection_v2"

CheckpointReason = Literal[
    "instruction_start",
    "physical_window_completed",
    "plan_updated",
    "knowledge_loaded",
    "task_terminal",
]


class MissionStateView(Protocol):
    """Read-only state surface consumed by :class:`ModelContextBuilder`.

    Memory owns the transformation into model context, but it must not depend
    on the Agent's concrete mutable ``MissionState`` implementation. Runtime
    passes that implementation structurally through this narrow protocol.
    """

    run_id: str
    user_instruction: str
    protocol_context: str
    task_plan: Any
    loaded_knowledge: list[dict[str, Any]]
    motion_summary: dict[str, Any]
    observation_modalities: tuple[str, ...]
    latest_observation: dict[str, Any]
    hover_required: bool
    yaw_scan_offset_deg: float
    yaw_scan_extent_deg: float
    yaw_scan_turn_count: int
    yaw_scan_recharge_count: int
    physical_window_count: int
CHECKPOINT_REASONS = frozenset(
    {
        "instruction_start",
        "physical_window_completed",
        "plan_updated",
        "knowledge_loaded",
        "task_terminal",
    }
)


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


@dataclass(frozen=True)
class HistoryImageCandidate:
    """Binary-free observation identity supplied to pure history selectors."""

    observation_id: str
    sequence: int
    frame_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MediaRef:
    """Durable identity for provider media; binary payloads are forbidden.

    ``sha256`` identifies source bytes.  The hydrator separately records the
    digest of the normalized provider payload in request audit data.
    Runtime-persisted ``ref`` values use POSIX paths relative to the owning run;
    absolute refs are reserved for base-less standalone diagnostics and are not
    part of durable Context V2 artifacts.
    """

    ref: str
    sha256: str
    mime_type: str
    observation_id: str
    frame_id: str
    role: str
    usage: str = ""
    width: int | None = None
    height: int | None = None
    display_width: int | None = None
    display_height: int | None = None
    color_bar_height: int = 0
    source_frame_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        digest = self.sha256.strip().lower()
        if not self.ref:
            raise ValueError("MediaRef.ref must not be empty")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("MediaRef.sha256 must be a lowercase SHA-256 digest")
        if not self.observation_id or not self.frame_id or not self.role:
            raise ValueError("MediaRef requires observation_id, frame_id and role")
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ref": self.ref,
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "observation_id": self.observation_id,
            "frame_id": self.frame_id,
            "role": self.role,
            "usage": self.usage,
            "color_bar_height": self.color_bar_height,
        }
        for key in ("width", "height", "display_width", "display_height"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.source_frame_ids:
            data["source_frame_ids"] = list(self.source_frame_ids)
        return data

    def to_model_input(self) -> dict[str, Any]:
        return {"type": "image", "media_ref": self.to_dict()}


@dataclass(frozen=True)
class PreparedMedia:
    """Transient provider-ready media produced after history selection."""

    ref: MediaRef
    data_url: str
    provider_sha256: str


class DerivedMediaWriter(Protocol):
    """Narrow capability injected into modules that create derived images."""

    def write_png(self, observation_id: str, data: bytes) -> tuple[str, str]:
        """Return the persisted path and SHA-256 digest."""


@dataclass(frozen=True)
class ContextFragment:
    """One module's complete model-facing output.

    Modules own these fields directly.  Agent projection may select media but
    must not reinterpret or strip module-specific content.
    """

    name: str
    model_content: dict[str, Any] = field(default_factory=dict)
    model_inputs: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_content", _clone(dict(self.model_content)))
        object.__setattr__(
            self,
            "model_inputs",
            tuple(_clone(dict(item)) for item in self.model_inputs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_content": _clone(self.model_content),
            "model_inputs": [_clone(item) for item in self.model_inputs],
        }


@dataclass(frozen=True)
class FragmentAudit:
    """Human-facing provenance for one fragment, never provider context."""

    name: str
    source: str
    confidence: str = "medium"
    warnings: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    build_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "artifact_refs": list(self.artifact_refs),
            "metadata": _clone(self.metadata),
            "build_duration_ms": round(float(self.build_duration_ms), 3),
        }


@dataclass(frozen=True)
class ContextModuleResult:
    """Paired model output and human audit returned by a context module."""

    fragment: ContextFragment
    audit: FragmentAudit


@dataclass(frozen=True)
class TaskContextInput:
    """Dynamic instruction state supplied to modules; no conversation history."""

    active_instruction: str
    protocol_context: str = ""
    task_plan: dict[str, Any] | None = None
    loaded_knowledge: list[dict[str, Any]] = field(default_factory=list)
    motion_summary: dict[str, Any] = field(default_factory=dict)
    safety_constraints: dict[str, Any] = field(default_factory=dict)
    observation_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemorySourceBundle:
    flight_memory: Any
    map_memory: Any = None
    run_memory: Any = None


@dataclass(frozen=True)
class InstructionEnvelope:
    """Instruction-scoped stable context emitted exactly once."""

    instruction_id: str | None
    protocol_context: str
    identity: dict[str, Any]
    user_preferences: dict[str, Any]
    knowledge_catalog: tuple[dict[str, Any], ...]
    static_module_guidance: tuple[dict[str, Any], ...]
    schema_version: str = INSTRUCTION_ENVELOPE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instruction_id": self.instruction_id,
            "protocol_context": self.protocol_context,
            "identity": _clone(self.identity),
            "user_preferences": _clone(self.user_preferences),
            "knowledge_catalog": [_clone(item) for item in self.knowledge_catalog],
            "static_module_guidance": [
                _clone(item) for item in self.static_module_guidance
            ],
        }


@dataclass(frozen=True)
class StateCheckpoint:
    """Immutable, binary-free dynamic state published to one transcript event."""

    checkpoint_id: str
    reason: CheckpointReason
    observation_id: str | None
    task_plan: dict[str, Any] | None
    loaded_knowledge: tuple[dict[str, Any], ...]
    fragments: tuple[ContextFragment, ...]
    fingerprint: str
    schema_version: str = STATE_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.reason not in CHECKPOINT_REASONS:
            raise ValueError(f"unsupported checkpoint reason: {self.reason}")

    def model_payload(self) -> dict[str, Any]:
        return {
            "task_plan": _clone(self.task_plan),
            "loaded_knowledge": [_clone(item) for item in self.loaded_knowledge],
            "fragments": [fragment.to_dict() for fragment in self.fragments],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "reason": self.reason,
            "observation_id": self.observation_id,
            **self.model_payload(),
            "fingerprint": self.fingerprint,
        }


def checkpoint_fingerprint(
    observation_id: str | None,
    task_plan: dict[str, Any] | None,
    loaded_knowledge: tuple[dict[str, Any], ...],
    fragments: tuple[ContextFragment, ...],
) -> str:
    """Hash every model-visible dynamic field, excluding publication metadata."""

    payload = {
        "observation_id": observation_id,
        "task_plan": task_plan,
        "loaded_knowledge": list(loaded_knowledge),
        "fragments": [fragment.to_dict() for fragment in fragments],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_binary_free(value: Any, *, path: str = "$") -> None:
    """Reject inline provider media before a checkpoint becomes durable."""

    if isinstance(value, str) and value.startswith("data:image/"):
        raise ValueError(f"inline image data is forbidden in V2 context at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_binary_free(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_binary_free(item, path=f"{path}[{index}]")
