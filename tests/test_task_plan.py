"""Advisory plan revisions preserve history identities without executing work."""

import pytest

from src.agent.planning.task_plan import TaskPlan


def step(id="one", status="pending", description="Reach the gate"):
    return {
        "id": id,
        "description": description,
        "status": status,
    }


def test_plan_updates_are_isolated_and_do_not_complete_mission():
    original = TaskPlan()
    payload = {"operation": "replace", "explanation": "Start", "steps": [step(status="in_progress")]}
    first = original.updated(payload)
    payload["steps"][0]["description"] = "external mutation"
    assert original.revision == 0 and not original.steps
    assert first.steps[0].description == "Reach the gate"
    completed = first.updated({"operation": "replace", "explanation": "Observed arrival", "steps": [step(status="completed")]})
    assert completed.revision == 2
    assert not hasattr(completed, "completed")
    output = completed.to_dict()
    output["steps"].clear()
    assert len(completed.steps) == 1


@pytest.mark.parametrize("steps", [
    [step(), step()],
    [step(status="in_progress"), step("two", "in_progress")],
    [step(status="invented")], [step(description=" ")],
    [{**step(), "evidence_refs": []}],
    [{**step(), "extra": True}], ["not a step"],
])
def test_invalid_steps_are_rejected(steps):
    with pytest.raises(ValueError):
        TaskPlan().updated({"operation": "replace", "explanation": "test", "steps": steps})


def test_preserve_ids_allow_explicit_cancellation_and_correction():
    first = TaskPlan().updated({"operation": "replace", "explanation": "test", "steps": [step(status="completed")]})
    with pytest.raises(ValueError, match="disappear"):
        first.updated({"operation": "replace", "explanation": "delete", "steps": []})
    reopened = first.updated({"operation": "replace", "explanation": "Earlier arrival was mistaken", "steps": [step()]})
    cancelled = reopened.updated({"operation": "replace", "explanation": "Superseded", "steps": [step(status="cancelled")]})
    assert cancelled.steps[0].status == "cancelled"
    with pytest.raises(ValueError, match="explanation"):
        cancelled.updated({"operation": "replace", "explanation": " ", "steps": [step()]})


def test_empty_plan_is_valid_and_never_mandatory():
    assert TaskPlan().updated({"operation": "replace", "explanation": "Simple task", "steps": []}).steps == ()


@pytest.mark.parametrize("payload", [
    {"explanation": None, "steps": [step()]},
    {"operation": "patch", "explanation": None, "updates": []},
    {"operation": "replace", "explanation": None, "steps": [{**step(), "description": None}]},
    {"operation": "replace", "explanation": None, "steps": [{**step(), "status": None}]},
    {"operation": "replace", "explanation": None, "steps": [{**step(), "evidence_refs": []}]},
])
def test_live_contract_rejects_old_interface_and_null_replacement(payload):
    with pytest.raises(ValueError):
        TaskPlan().updated(payload)


@pytest.mark.parametrize("reverse", [False, True])
def test_incremental_progress_is_atomic_order_independent_and_preserves_fields(reverse):
    original = TaskPlan().updated({"operation": "replace", "explanation": "Route", "steps": [
        step(status="in_progress"), step("two"),
    ]})
    updates = [{"id": "one", "description": None, "status": "completed"},
               {"id": "two", "description": None, "status": "in_progress"}]
    if reverse:
        updates.reverse()
    advanced = original.updated({"operation": "patch", "explanation": None, "steps": updates})
    assert [s.status for s in original.steps] == ["in_progress", "pending"]
    assert [s.status for s in advanced.steps] == ["completed", "in_progress"]
    assert advanced.explanation == "Route" and advanced.revision == 2
    assert advanced.steps[0].description == "Reach the gate"
    assert advanced.changed_step_ids(original) == ["one", "two"]
    updates[0]["status"] = "cancelled"
    assert [s.status for s in advanced.steps] == ["completed", "in_progress"]


@pytest.mark.parametrize("payload", [
    {}, {"explanation": "Reason only"}, {"steps": [], "updates": []},
    {"operation": "patch", "explanation": None, "steps": []}, {"operation": "patch", "explanation": None, "steps": "bad"}, {"operation": "patch", "explanation": None, "steps": ["bad"]},
    {"operation": "patch", "explanation": None, "steps": [{"description": None, "status": None, "id": "one"}]}, {"operation": "patch", "explanation": None, "steps": [{"description": None, "status": "completed"}]},
    {"operation": "patch", "explanation": None, "steps": [{"description": None, "id": "missing", "status": "completed"}]},
    {"operation": "patch", "explanation": None, "steps": [{"description": None, "id": "one", "status": "completed"}, {"description": None, "id": " one ", "status": "pending"}]},
    {"operation": "patch", "explanation": None, "steps": [{"description": None, "id": "one", "status": "unknown"}]},
    {"operation": "patch", "explanation": None, "steps": [{"status": None, "id": "one", "description": " "}]},
    {"operation": "patch", "explanation": None, "steps": [{"description": None, "id": "one", "status": "completed", "evidence_refs": []}]},
    {"operation": "patch", "explanation": None, "steps": [{"description": None, "id": "two", "status": "in_progress"}]},
    {"operation": "replace", "explanation": None, "steps": [{"id": "one", "description": "x", "status": "pending"}], "extra": True},
])
def test_invalid_updates_leave_original_snapshot_intact(payload):
    plan = TaskPlan().updated({"operation": "replace", "explanation": None, "steps": [step(status="in_progress"), step("two")]})
    original = plan.to_dict()
    with pytest.raises(ValueError):
        plan.updated(payload)
    assert plan.to_dict() == original


def test_description_correction_reopening_reordering_and_noop_revision():
    first = TaskPlan().updated({"operation": "replace", "explanation": None, "steps": [step(status="completed"), step("two")]})
    corrected = first.updated({"operation": "patch", "explanation": "Arrival was mistaken", "steps": [
        {"id": "one", "description": "Confirm the gate", "status": "in_progress"},
    ]})
    reordered = corrected.updated({"operation": "replace", "explanation": None, "steps": list(reversed(corrected.to_dict()["steps"]))})
    assert reordered.changed_step_ids(corrected) == ["two", "one"]
    unchanged = reordered.updated({"operation": "patch", "explanation": None, "steps": [{"description": None, "id": "one", "status": "in_progress"}]})
    assert unchanged.revision == reordered.revision + 1
    assert unchanged.changed_step_ids(reordered) == []
    assert unchanged.explanation == "Arrival was mistaken"
