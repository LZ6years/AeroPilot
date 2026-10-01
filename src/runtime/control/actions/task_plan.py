"""Apply an advisory plan update without executing or completing a task."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from src.agent.planning.task_plan import TaskPlan
from src.agent.transcript import ToolResult, TranscriptEvent
from src.tools.registry import ToolRegistry


class PlanState(Protocol):
    task_plan: TaskPlan | None
    completed: bool
    stopped: bool


def plan_evidence_refs(
    events: Iterable[TranscriptEvent], registry: ToolRegistry,
) -> frozenset[str]:
    """Return prior successful result identities usable by plan revisions.

    This validates provenance only. Runtime deliberately does not judge whether
    a cited observation or result semantically proves a step. Control and
    lifecycle results are excluded because they describe orchestration rather
    than sensed or executed work.
    """

    refs: set[str] = set()
    for event in events:
        if event.kind == "user_instruction":
            refs.clear()
        elif event.kind == "state_checkpoint":
            observation_id = getattr(event.payload, "observation_id", None)
            if isinstance(observation_id, str) and observation_id:
                refs.add(observation_id)
        elif event.kind == "tool_result":
            result: ToolResult = event.payload
            if result.status not in {"ok", "collision"}:
                continue
            try:
                effect = registry.get(result.tool_name).runtime.effect
            except KeyError:
                continue
            if effect not in {"physical", "read_only"}:
                continue
            refs.add(result.tool_call_id)
            if result.result_ref:
                refs.add(result.result_ref)
    return frozenset(refs)


def validate_plan_update(
    state: PlanState,
    payload: dict[str, Any],
    *,
    available_evidence: frozenset[str],
) -> tuple[TaskPlan, TaskPlan]:
    """Return the previous and candidate plans without mutating mission state."""

    if state.completed or state.stopped:
        raise ValueError("cannot update a terminal task")
    previous = state.task_plan or TaskPlan()
    updated = previous.updated(payload)
    cited = {ref for step in updated.steps for ref in step.evidence_refs}
    missing = sorted(cited - available_evidence)
    if missing:
        examples = sorted(available_evidence)[:5]
        raise ValueError(
            "plan evidence_refs must cite evidence available before this turn; "
            f"invalid references: {missing[:5]!r}; accepted examples: {examples!r}"
        )
    return previous, updated


def apply_plan_update(
    state: PlanState,
    payload: dict[str, Any],
    *,
    available_evidence: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Validate before mutation; the caller persists the returned revision."""

    previous, updated = validate_plan_update(
        state, payload, available_evidence=available_evidence
    )
    state.task_plan = updated
    return {"previous_revision": previous.revision, "plan": updated.to_dict()}
