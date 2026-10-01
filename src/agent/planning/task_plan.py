"""Optional task-list state; no model, execution, or completion authority.

Updates return independent immutable snapshots. Runtime owns persistence and
applies a returned snapshot only after validating the whole tool turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PLAN_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})


@dataclass(frozen=True)
class TaskPlanStep:
    id: str
    description: str
    status: str
    evidence_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskPlanStep":
        required = {"id", "description", "status", "evidence_refs"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(
                "plan step requires only id, description, status and evidence_refs"
            )
        if any(
            not isinstance(value[key], str) or not value[key].strip()
            for key in ("id", "description", "status")
        ):
            raise ValueError("plan step fields must be non-empty strings")
        if value["status"] not in PLAN_STATUSES:
            raise ValueError("unsupported plan step status")
        refs = value["evidence_refs"]
        if (
            not isinstance(refs, list)
            or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
            or len(set(refs)) != len(refs)
        ):
            raise ValueError("plan step evidence_refs must be unique non-empty strings")
        if value["status"] == "completed" and not refs:
            raise ValueError("completed plan steps require prior evidence_refs")
        return cls(
            value["id"].strip(),
            value["description"].strip(),
            value["status"],
            tuple(ref.strip() for ref in refs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class TaskPlan:
    """Latest advisory plan. Completed steps never imply a completed mission."""

    steps: tuple[TaskPlanStep, ...] = ()
    revision: int = 0
    explanation: str = ""

    def updated(self, payload: dict[str, Any]) -> "TaskPlan":
        if not isinstance(payload, dict) or set(payload) != {"explanation", "steps"}:
            raise ValueError("plan update requires only explanation and steps")
        explanation = payload["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("plan update requires a non-empty explanation")
        if not isinstance(payload["steps"], list):
            raise ValueError("plan steps must be a list")
        steps = tuple(TaskPlanStep.from_dict(item) for item in payload["steps"])
        ids = [step.id for step in steps]
        if len(set(ids)) != len(ids):
            raise ValueError("plan step IDs must be unique")
        if sum(step.status == "in_progress" for step in steps) > 1:
            raise ValueError("at most one plan step may be in_progress")
        if not {step.id for step in self.steps}.issubset(ids):
            raise ValueError("existing steps cannot disappear; mark them cancelled instead")
        # Corrections and reordering are intentional model decisions. Never
        # infer semantic equivalence of descriptions or silently retain steps.
        return TaskPlan(steps, self.revision + 1, explanation.strip())

    def to_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "explanation": self.explanation,
                "steps": [step.to_dict() for step in self.steps]}
