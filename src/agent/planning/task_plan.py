"""Optional task-list state; no model, execution, or completion authority.

Updates return independent immutable snapshots. Runtime owns persistence and
applies a returned snapshot only after validating the whole tool turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.tools.validation import ToolArgumentError


PLAN_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})


@dataclass(frozen=True)
class TaskPlanStep:
    id: str
    description: str
    status: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskPlanStep":
        required = {"id", "description", "status"}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(
                "plan step requires only id, description and status"
            )
        if any(
            not isinstance(value[key], str) or not value[key].strip()
            for key in ("id", "description", "status")
        ):
            raise ValueError("plan step fields must be non-empty strings")
        if value["status"] not in PLAN_STATUSES:
            raise ValueError("unsupported plan step status")
        return cls(
            value["id"].strip(),
            value["description"].strip(),
            value["status"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
        }


@dataclass(frozen=True)
class TaskPlan:
    """Latest advisory plan. Completed steps never imply a completed mission."""

    steps: tuple[TaskPlanStep, ...] = ()
    revision: int = 0
    explanation: str = ""

    def updated(self, payload: dict[str, Any]) -> "TaskPlan":
        """Build a candidate snapshot before Runtime commits any state.

        Full snapshots retain existing identities. Patches address only existing
        steps and are all merged before validating the single active milestone;
        completing one step and activating another is therefore order-independent.
        """
        if not isinstance(payload, dict) or set(payload) != {"operation", "explanation", "steps"}:
            raise ToolArgumentError("plan update requires only operation, explanation and steps",
                                    code="plan_update_invalid", path="args")
        operation = payload["operation"]
        if not isinstance(operation, str) or operation not in {"replace", "patch"}:
            raise ToolArgumentError("plan operation must be replace or patch",
                                    code="plan_update_invalid", path="args.operation")
        explanation = payload["explanation"]
        if explanation is None:
            explanation = self.explanation
        elif (
            not isinstance(explanation, str) or not explanation.strip()
        ):
            raise ToolArgumentError("plan explanation must be non-empty or null",
                                    code="plan_update_invalid", path="args.explanation")
        if operation == "replace":
            if not isinstance(payload["steps"], list):
                raise ValueError("plan steps must be a list")
            steps = tuple(TaskPlanStep.from_dict(item) for item in payload["steps"])
        else:
            updates = payload["steps"]
            if not isinstance(updates, list) or not updates:
                raise ValueError("plan updates must be a non-empty list")
            candidates = {step.id: step.to_dict() for step in self.steps}
            seen: set[str] = set()
            for index, update in enumerate(updates):
                if (
                    not isinstance(update, dict)
                    or set(update) != {"id", "description", "status"}
                    or all(update[key] is None for key in ("description", "status"))
                ):
                    raise ToolArgumentError("patch step requires id, description and status; at least one non-null change",
                                            code="plan_update_invalid", path=f"args.steps[{index}]")
                identity = update["id"]
                if not isinstance(identity, str) or not identity.strip():
                    raise ValueError("plan step ID must be a non-empty string")
                identity = identity.strip()
                if identity in seen:
                    raise ValueError("plan update IDs must be unique")
                if identity not in candidates:
                    raise ValueError(f"unknown plan step ID: {identity!r}")
                seen.add(identity)
                candidates[identity].update({key: value for key, value in update.items() if value is not None})
                candidates[identity]["id"] = identity
            steps = tuple(TaskPlanStep.from_dict(candidates[step.id]) for step in self.steps)
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

    def changed_step_ids(self, previous: "TaskPlan") -> list[str]:
        """Return changed identities in saved order, including moved milestones."""
        old = {step.id: (index, step) for index, step in enumerate(previous.steps)}
        return [
            step.id for index, step in enumerate(self.steps)
            if old.get(step.id) != (index, step)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"revision": self.revision, "explanation": self.explanation,
                "steps": [step.to_dict() for step in self.steps]}
