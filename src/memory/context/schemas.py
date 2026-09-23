"""Shared schemas for memory modules and planner context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class HistoryImageCandidate:
    """Binary-free observation identity passed to pure module selectors.

    Sequence is first-occurrence order within one active instruction, not time.
    Frame IDs describe the owning fragment; selectors never receive tool events.
    """

    observation_id: str
    sequence: int
    frame_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BudgetSpec:
    priority: int
    tokens_estimate: int = 0
    max_items: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextFragment:
    name: str
    enabled: bool
    source: str
    content: dict[str, Any] = field(default_factory=dict)
    model_inputs: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[dict[str, Any]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"
    artifact_refs: list[str] = field(default_factory=list)
    budget: BudgetSpec = field(default_factory=lambda: BudgetSpec(priority=50))
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["budget"] = self.budget.to_dict()
        return data


@dataclass(frozen=True)
class TaskContextInput:
    """Current instruction snapshot; no phase, history or tool execution objects."""

    active_instruction: str
    protocol_context: str = ""
    task_plan: dict[str, Any] | None = None
    loaded_knowledge: list[dict[str, Any]] = field(default_factory=list)
    motion_summary: dict[str, Any] = field(default_factory=dict)
    safety_constraints: dict[str, Any] = field(default_factory=dict)
    observation_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemorySourceBundle:
    flight_memory: Any
    map_memory: Any = None
    run_memory: Any = None
