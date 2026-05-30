"""Schemas for semantic task parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticTarget:
    description: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticTarget":
        return cls(
            description=str(data.get("description", "")).strip(),
            attributes=dict(data.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticGoal:
    target: SemanticTarget
    final_state: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticGoal":
        return cls(
            target=SemanticTarget.from_dict(dict(data.get("target", {}))),
            final_state=str(data.get("final_state", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target.to_dict(), "final_state": self.final_state}


@dataclass(frozen=True)
class SemanticPhase:
    index: int
    total: int
    name: str
    meaning: str
    completion_criteria: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int | None = None, total: int | None = None) -> "SemanticPhase":
        phase_index = index if index is not None else data.get("index", 0)
        phase_total = total if total is not None else data.get("total", 0)
        return cls(
            index=int(phase_index),
            total=int(phase_total),
            name=str(data.get("name", "")).strip(),
            meaning=str(data.get("meaning", "")).strip(),
            completion_criteria=_criteria_from_dict(data.get("completion_criteria")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _criteria_from_dict(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [dict(value)]
    return []


@dataclass(frozen=True)
class SemanticPlan:
    intent: str
    goal: SemanticGoal
    phases: list[SemanticPhase]
    uncertainties: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticPlan":
        return cls(
            intent=str(data.get("intent", "")).strip(),
            goal=SemanticGoal.from_dict(dict(data.get("goal", {}))),
            phases=[
                SemanticPhase.from_dict(item, index=index, total=len(data.get("phases", [])))
                for index, item in enumerate(data.get("phases", []), start=1)
            ],
            uncertainties=[str(item).strip() for item in data.get("uncertainties", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "goal": self.goal.to_dict(),
            "phases": [phase.to_dict() for phase in self.phases],
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True)
class TaskParseResult:
    instruction: str
    parser_mode: str
    plan: SemanticPlan
    validation_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "parser_mode": self.parser_mode,
            "validation_ok": self.validation_ok,
            "plan": self.plan.to_dict(),
        }
