"""Apply an advisory plan update without executing or completing a task."""

from __future__ import annotations

from typing import Any, Protocol

from src.agent.planning.task_plan import TaskPlan


class PlanState(Protocol):
    task_plan: TaskPlan | None
    completed: bool
    stopped: bool


def validate_plan_update(
    state: PlanState,
    payload: dict[str, Any],
) -> tuple[TaskPlan, TaskPlan]:
    """Return the previous and candidate plans without mutating mission state."""

    if state.completed or state.stopped:
        raise ValueError("cannot update a terminal task")
    previous = state.task_plan or TaskPlan()
    updated = previous.updated(payload)
    return previous, updated


def apply_plan_update(
    state: PlanState,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate before mutation; the caller persists the returned revision."""

    previous, updated = validate_plan_update(state, payload)
    state.task_plan = updated
    return {"previous_revision": previous.revision, "plan": updated.to_dict(),
            "changed_step_ids": updated.changed_step_ids(previous)}
