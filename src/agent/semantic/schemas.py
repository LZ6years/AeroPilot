"""Schemas for semantic task parsing.

The semantic plan is a compact model-facing contract. It describes what each
phase means and how completion should be judged, without encoding primitive UAV
tools or forcing every visual/spatial task into fixed criterion enums.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskGoal:
    target: str
    final_state: str
    target_attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskGoal":
        return cls(
            target=str(data.get("target", "")).strip(),
            target_attributes=dict(data.get("target_attributes", {})),
            final_state=str(data.get("final_state", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticPhase:
    index: int
    total: int
    name: str
    goal: str
    done_when: list[str] = field(default_factory=list)
    continue_when: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int | None = None, total: int | None = None) -> "SemanticPhase":
        phase_index = index if index is not None else data.get("index", 0)
        phase_total = total if total is not None else data.get("total", 0)
        return cls(
            index=int(phase_index),
            total=int(phase_total),
            name=str(data.get("name", "")).strip(),
            goal=str(data.get("goal", "")).strip(),
            done_when=_string_list(data.get("done_when")),
            continue_when=_string_list(data.get("continue_when")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticPlan:
    intent: str
    task_goal: TaskGoal
    phases: list[SemanticPhase]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticPlan":
        phases = data.get("phases", [])
        return cls(
            intent=str(data.get("intent", "")).strip(),
            task_goal=TaskGoal.from_dict(dict(data.get("task_goal", {}))),
            phases=[
                SemanticPhase.from_dict(item, index=index, total=len(phases))
                for index, item in enumerate(phases, start=1)
                if isinstance(item, dict)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "task_goal": self.task_goal.to_dict(),
            "phases": [phase.to_dict() for phase in self.phases],
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
