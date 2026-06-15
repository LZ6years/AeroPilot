"""Shared schemas for memory modules and planner context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
class StageContextInput:
    active_instruction: str
    semantic_plan: dict[str, Any] = field(default_factory=dict)
    current_phase: dict[str, Any] | None = None
    semantic_goal: dict[str, Any] = field(default_factory=dict)
    safety_constraints: dict[str, Any] = field(default_factory=dict)
    allowed_action_constraints: dict[str, Any] = field(default_factory=dict)
    observation_policy: dict[str, Any] = field(default_factory=dict)
    phase_execution_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_instruction": self.active_instruction,
            "semantic_plan": dict(self.semantic_plan),
            "current_phase": dict(self.current_phase) if self.current_phase is not None else None,
            "semantic_goal": dict(self.semantic_goal),
            "safety_constraints": dict(self.safety_constraints),
            "allowed_action_constraints": dict(self.allowed_action_constraints),
            "observation_policy": dict(self.observation_policy),
            "phase_execution_summary": dict(self.phase_execution_summary),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemorySourceBundle:
    flight_memory: Any
    map_memory: Any = None
    run_memory: Any = None


@dataclass(frozen=True)
class PlannerContext:
    active_instruction: str
    current_phase: dict[str, Any] | None
    semantic_goal: dict[str, Any]
    fragments: list[ContextFragment]
    disabled_fragments: list[str] = field(default_factory=list)
    context_schema: dict[str, Any] = field(default_factory=dict)
    rules: dict[str, Any] = field(default_factory=dict)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    safety_constraints: dict[str, Any] = field(default_factory=dict)
    allowed_action_constraints: dict[str, Any] = field(default_factory=dict)
    observation_policy: dict[str, Any] = field(default_factory=dict)
    phase_execution_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_instruction": self.active_instruction,
            "current_phase": self.current_phase,
            "semantic_goal": self.semantic_goal,
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "disabled_fragments": list(self.disabled_fragments),
            "context_schema": dict(self.context_schema),
            "rules": dict(self.rules),
            "user_preferences": dict(self.user_preferences),
            "safety_constraints": dict(self.safety_constraints),
            "allowed_action_constraints": dict(self.allowed_action_constraints),
            "observation_policy": dict(self.observation_policy),
            "phase_execution_summary": dict(self.phase_execution_summary),
            "metadata": dict(self.metadata),
        }
