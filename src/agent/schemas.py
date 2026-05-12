"""Schemas for semantic task parsing and planning."""

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


@dataclass(frozen=True)
class PlannedStep:
    kind: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=lambda: ["rgb", "depth", "pose"])
    reason: str = ""

    @classmethod
    def tool_step(cls, tool: str, args: dict[str, Any] | None = None, reason: str = "") -> "PlannedStep":
        return cls(kind="tool", tool=tool, args=dict(args or {}), reason=reason)

    @classmethod
    def observe_step(
        cls, modalities: list[str] | None = None, reason: str = "refresh raw observation"
    ) -> "PlannedStep":
        return cls(kind="observe", modalities=list(modalities or ["rgb", "depth", "pose"]), reason=reason)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannedStep":
        kind = str(data.get("kind", "")).strip()
        if kind == "observe":
            return cls.observe_step(
                modalities=[str(item) for item in data.get("modalities", ["rgb", "depth", "pose"])],
                reason=str(data.get("reason", "refresh raw observation")),
            )
        if kind == "tool":
            return cls.tool_step(
                str(data.get("tool", "")).strip(),
                dict(data.get("args", {})),
                str(data.get("reason", "")),
            )
        raise ValueError(f"Unsupported planned step kind: {kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tool": self.tool,
            "args": dict(self.args),
            "modalities": list(self.modalities),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ShortHorizonPlan:
    phase: str
    steps: list[PlannedStep]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShortHorizonPlan":
        return cls(
            phase=str(data.get("phase", "")).strip(),
            steps=[PlannedStep.from_dict(dict(item)) for item in data.get("steps", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "steps": [step.to_dict() for step in self.steps]}
