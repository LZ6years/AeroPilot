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

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int | None = None, total: int | None = None) -> "SemanticPhase":
        return cls(
            index=int(data.get("index", index or 0)),
            total=int(data.get("total", total or 0)),
            name=str(data.get("name", "")).strip(),
            meaning=str(data.get("meaning", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
