"""Tests for semantic_plan_update patch validation and background jobs."""

from __future__ import annotations

import time

import pytest

from src.agent.semantic.schemas import SemanticPhase, SemanticPlan, TaskGoal
from src.agent.semantic.plan_updater import SemanticPlanUpdater, validate_semantic_plan_patch_numeric_provenance
from src.agent.semantic.validator import (
    SemanticPlanValidationError,
    apply_semantic_plan_patch,
    validate_semantic_plan_patch,
)
from src.runtime.control.actions import SemanticPlanUpdateJob, SemanticPlanUpdateManager
from src.runtime.configs.model_config import ModelConfig


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
            current_phase={"goal": "orbit at a radius of 3.0 meters"},
        )


def test_semantic_plan_update_can_reuse_instruction_numbers_in_natural_language():
    patch = validate_semantic_plan_patch(
        {
            "phase": {
                "done_when": [
                    "the UAV is about 5 meters from its phase-start position while preserving the 3.0 meter relation"
                ],
            }
        }
    )

    result = validate_semantic_plan_patch_numeric_provenance(
        patch,
        instruction="Move 5 meters while preserving a 3.0 meter relation",
        current_phase={"goal": "preserve the requested 3.0 meter relation"},
    )

    assert result == patch


def test_semantic_plan_update_rejects_removed_metrics_patch():
    with pytest.raises(SemanticPlanValidationError, match="unsupported field"):
        validate_semantic_plan_patch({"phase": {"metrics": []}})


def test_semantic_plan_updater_messages_reuse_selected_core_skill_guidance():
    updater = SemanticPlanUpdater(
        ModelConfig(
            task_parser_mode="model",
            provider="openai_compatible",
            model_name="test-model",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )
    context = {
        "user_instruction": "Pass the landmark",
        "current_phase": {"name": "traverse_route"},
        "selected_skill_guidance": [
            {
                "name": "route_traversal_and_passage",
                "skill_text": "Use route progress evidence.",
            }
        ],
        "configured_evidence": [
            "NOW RGB: current visible appearance and image-space relations.",
            "Action history: phase-relative UAV self-motion; not target distance.",
        ],
    }

    messages = updater._build_messages(context)

    assert [message["role"] for message in messages] == [
        "system",
        "system",
        "system",
        "system",
        "system",
        "system",
        "user",
    ]
    assert messages[0]["content"] == updater.agent_rules
    assert messages[1]["content"] == updater.user_preferences
    assert "route_traversal_and_passage" in messages[2]["content"]
    assert "Use route progress evidence." in messages[2]["content"]
    assert "Potential semantic evidence configured" in messages[3]["content"]
    assert "NOW RGB:" in messages[3]["content"]
    assert messages[4]["content"] == updater.done_when_rules
    assert messages[5]["content"] == updater.system_prompt
    assert "selected_skill_guidance" not in messages[6]["content"]
    assert "configured_evidence" not in messages[6]["content"]
    assert "Pass the landmark" in messages[6]["content"]


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
