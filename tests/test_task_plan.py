"""Advisory plan revisions preserve history identities without executing work."""

import pytest

from src.agent.planning.task_plan import TaskPlan


def step(id="one", status="pending", description="Reach the gate"):
    return {"id": id, "description": description, "status": status}


def test_plan_updates_are_isolated_and_do_not_complete_mission():
    original = TaskPlan()
    payload = {"explanation": "Start", "steps": [step(status="in_progress")]}
    first = original.updated(payload)
    payload["steps"][0]["description"] = "external mutation"
    assert original.revision == 0 and not original.steps
    assert first.steps[0].description == "Reach the gate"
    completed = first.updated({"explanation": "Observed arrival", "steps": [step(status="completed")]})
    assert completed.revision == 2
    assert not hasattr(completed, "completed")
    output = completed.to_dict()
    output["steps"].clear()
    assert len(completed.steps) == 1


@pytest.mark.parametrize("steps", [
    [step(), step()],
    [step(status="in_progress"), step("two", "in_progress")],
    [step(status="invented")], [step(description=" ")],
    [{**step(), "extra": True}], ["not a step"],
])
def test_invalid_steps_are_rejected(steps):
    with pytest.raises(ValueError):
        TaskPlan().updated({"explanation": "test", "steps": steps})


def test_preserve_ids_allow_explicit_cancellation_and_correction():
    first = TaskPlan().updated({"explanation": "test", "steps": [step(status="completed")]})
    with pytest.raises(ValueError, match="disappear"):
        first.updated({"explanation": "delete", "steps": []})
    reopened = first.updated({"explanation": "Earlier arrival was mistaken", "steps": [step()]})
    cancelled = reopened.updated({"explanation": "Superseded", "steps": [step(status="cancelled")]})
    assert cancelled.steps[0].status == "cancelled"
    with pytest.raises(ValueError, match="explanation"):
        cancelled.updated({"explanation": " ", "steps": [step()]})


def test_empty_plan_is_valid_and_never_mandatory():
    assert TaskPlan().updated({"explanation": "Simple task", "steps": []}).steps == ()
