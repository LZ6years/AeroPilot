"""Tests for semantic_plan_update patch validation and background jobs."""

from __future__ import annotations

import time

import pytest

from src.agent.semantic.schemas import SemanticPhase, SemanticPlan, TaskGoal
from src.agent.semantic.plan_updater import validate_semantic_plan_patch_numeric_provenance
from src.agent.semantic.validator import (
    SemanticPlanValidationError,
    apply_semantic_plan_patch,
    validate_semantic_plan_patch,
)
from src.runtime.control.actions import SemanticPlanUpdateJob, SemanticPlanUpdateManager


def make_plan() -> SemanticPlan:
    return SemanticPlan(
        intent="test_semantic_update",
        task_goal=TaskGoal(target="target", final_state="done"),
        phases=[
            SemanticPhase(
                index=1,
                total=1,
                name="orbit_target",
                goal="orbit the target",
                done_when=["exact radius is confirmed"],
                continue_when=["exact radius is not confirmed"],
                metrics=[],
            )
        ],
    )


def test_semantic_plan_patch_updates_current_phase_contract():
    updated = apply_semantic_plan_patch(
        make_plan(),
        0,
        {
            "phase": {
                "goal": "continue a conservative orbit around the target",
                "done_when": ["target remains usable", "motion shows around-target progress"],
                "continue_when": ["target evidence is missing"],
                "metrics": [],
            }
        },
    )

    phase = updated.phases[0]
    assert phase.goal == "continue a conservative orbit around the target"
    assert phase.done_when == ["target remains usable", "motion shows around-target progress"]
    assert phase.continue_when == ["target evidence is missing"]


def test_semantic_plan_patch_can_append_generic_recovery_phase():
    updated = apply_semantic_plan_patch(
        make_plan(),
        0,
        {
            "phase": {"done_when": ["target remains usable"]},
            "append_recovery_phase": {
                "name": "recover_reference",
                "goal": "recover usable reference evidence",
                "done_when": ["reference evidence is usable again"],
                "continue_when": ["reference evidence is still missing"],
                "metrics": [],
            },
        },
    )

    assert [phase.index for phase in updated.phases] == [1, 2]
    assert [phase.total for phase in updated.phases] == [2, 2]
    assert updated.phases[1].name == "recover_reference"


def test_semantic_plan_patch_rejects_full_plan_rewrite():
    with pytest.raises(SemanticPlanValidationError, match="unsupported top-level"):
        validate_semantic_plan_patch({"phases": []})


def test_semantic_plan_patch_rejects_unsupported_phase_field():
    with pytest.raises(SemanticPlanValidationError, match="unsupported field"):
        validate_semantic_plan_patch({"phase": {"name": "new_name"}})


def test_semantic_plan_patch_rejects_primitive_tool_tokens():
    with pytest.raises(SemanticPlanValidationError, match="primitive action"):
        validate_semantic_plan_patch({"phase": {"goal": "call uav.move_body to fix the plan"}})


def test_semantic_plan_update_rejects_invented_numeric_thresholds():
    patch = validate_semantic_plan_patch(
        {
            "phase": {
                "done_when": ["target remains visible for 3 consecutive observations"],
                "continue_when": ["radius differs by more than 0.5 meters"],
            }
        }
    )

    with pytest.raises(SemanticPlanValidationError, match="unsupported numeric"):
        validate_semantic_plan_patch_numeric_provenance(
            patch,
            instruction="Circle around the person at a radius of 3.0 meters",
            current_phase={"goal": "orbit at a radius of 3.0 meters", "metrics": []},
        )


def test_semantic_plan_update_can_reuse_instruction_and_current_contract_numbers():
    patch = validate_semantic_plan_patch(
        {
            "phase": {
                "done_when": ["the requested 3.0 meter relation is supported"],
                "metrics": [
                    {
                        "name": "planar_distance",
                        "target": 5.0,
                        "unit": "m",
                        "tolerance": 1.0,
                        "correction_allowed": True,
                    }
                ],
            }
        }
    )

    result = validate_semantic_plan_patch_numeric_provenance(
        patch,
        instruction="Move 5 meters while preserving a 3.0 meter relation",
        current_phase={"metrics": [{"target": 5.0, "tolerance": 1.0}]},
    )

    assert result == patch


class FakeUpdater:
    def __init__(self, patch: dict) -> None:
        self.patch = patch
        self.contexts = []

    def update(self, context):
        self.contexts.append(context)
        return dict(self.patch)


def test_semantic_plan_update_manager_runs_background_job():
    updater = FakeUpdater({"phase": {"done_when": ["updated condition"]}})
    manager = SemanticPlanUpdateManager(updater)
    job = SemanticPlanUpdateJob(
        run_id="run",
        phase_key="1:phase",
        semantic_plan_version=0,
        request_payload={"reason": "too strict"},
        updater_context={"current_phase": {"name": "phase"}},
    )

    ok, reason = manager.submit(job)
    result = None
    for _ in range(100):
        result = manager.take_completed()
        if result is not None:
            break
        time.sleep(0.01)

    assert ok is True
    assert reason == "submitted"
    assert result is not None
    assert result.status == "completed"
    assert result.patch == {"phase": {"done_when": ["updated condition"]}}
    assert updater.contexts == [{"current_phase": {"name": "phase"}}]
