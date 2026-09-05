"""Tests for runtime control action validation."""

from __future__ import annotations

import threading

from src.agent.planning.schemas import RuntimeControlAction
from src.runtime.configs.runtime_control_config import RuntimeControlActionConfig, RuntimeControlConfig
from src.runtime.control import RuntimeControlExecutor
from src.runtime.control.actions import SemanticPlanUpdateJob
from src.runtime.control.recommendations import (
    SEMANTIC_PLAN_UPDATE_RECOMMENDATION,
    runtime_control_recommendations,
)


def make_executor(
    *,
    enabled: bool = True,
    exposed: bool = True,
    semantic_plan_updater=None,
) -> RuntimeControlExecutor:
    return RuntimeControlExecutor(
        RuntimeControlConfig(
            enabled=enabled,
            actions={
                "ask_user": RuntimeControlActionConfig(enabled=True, exposed=exposed),
                "semantic_plan_update": RuntimeControlActionConfig(enabled=True, exposed=exposed),
            },
        ),
        semantic_plan_updater=semantic_plan_updater,
    )


def test_runtime_control_executor_rejects_unknown_action():
    result = make_executor().validate(RuntimeControlAction("fake_control", "test", {}))

    assert result.status == "rejected"
    assert result.reason == "unknown_runtime_control_action"


def test_runtime_control_executor_rejects_disabled_action():
    result = make_executor(enabled=False).validate(
        RuntimeControlAction("semantic_plan_update", "phase contract blocks progress", {"reason": "too strict"})
    )

    assert result.status == "rejected"
    assert result.reason == "runtime_control_action_disabled"


def test_runtime_control_executor_rejects_unexposed_action():
    result = make_executor(exposed=False).validate(
        RuntimeControlAction("semantic_plan_update", "phase contract blocks progress", {"reason": "too strict"})
    )

    assert result.status == "rejected"
    assert result.reason == "runtime_control_action_not_exposed"


def test_runtime_control_executor_rejects_invalid_payload():
    result = make_executor().validate(RuntimeControlAction("ask_user", "need clarification", {}))

    assert result.status == "rejected"
    assert "missing required payload field" in result.reason


def test_runtime_control_executor_phase_1_validates_without_execution():
    executor = make_executor()

    result = executor.validate(
        RuntimeControlAction("semantic_plan_update", "phase contract blocks progress", {"reason": "too strict"})
    )

    assert result.status == "accepted"
    assert result.reason == "validated_for_submission"
    context = executor.model_context()
    assert context["available"] == ["ask_user", "semantic_plan_update"]
    assert context["contracts"] == {
        "ask_user": (
            "Request when user clarification is needed to avoid a wrong target, unsafe approach, "
            "or changing the task meaning."
        ),
        "semantic_plan_update": (
            "After multiple action-observation cycles, request when confidence is low that the current goal, "
            "done_when, or continue_when still fit observed progress and user intent. Name the doubtful "
            "fields and evidence; keep normal UAV steps if continuing."
        ),
    }


def test_runtime_control_executor_tracks_semantic_update_pending_status():
    class BlockingUpdater:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def update(self, context):
            self.started.set()
            self.release.wait(timeout=2.0)
            return {}

    updater = BlockingUpdater()
    executor = make_executor(semantic_plan_updater=updater)
    job = SemanticPlanUpdateJob(
        run_id="run",
        phase_key="1:phase",
        semantic_plan_version=0,
        request_payload={"reason": "too strict"},
        updater_context={"current_phase": {"name": "phase"}},
    )

    result = executor.submit_semantic_plan_update(job)

    assert result.status == "pending"
    assert updater.started.wait(timeout=1.0)
    assert executor.model_context() == {
        "available": ["ask_user", "semantic_plan_update"],
        "contracts": {
            "ask_user": (
                "Request when user clarification is needed to avoid a wrong target, unsafe approach, "
                "or changing the task meaning."
            ),
            "semantic_plan_update": (
                "After multiple action-observation cycles, request when confidence is low that the current goal, "
                "done_when, or continue_when still fit observed progress and user intent. Name the doubtful "
                "fields and evidence; keep normal UAV steps if continuing."
            ),
        },
        "status": {"semantic_plan_update": "pending"},
    }
    updater.release.set()


def test_runtime_control_executor_rejects_unconfigured_semantic_updater():
    executor = make_executor()
    job = SemanticPlanUpdateJob(
        run_id="run",
        phase_key="1:phase",
        semantic_plan_version=0,
        request_payload={"reason": "too strict"},
        updater_context={"current_phase": {"name": "phase"}},
    )

    result = executor.submit_semantic_plan_update(job)

    assert result.status == "rejected"
    assert result.reason == "semantic_plan_updater_unconfigured"


def test_runtime_control_executor_only_emits_available_contracts():
    executor = RuntimeControlExecutor(
        RuntimeControlConfig(
            enabled=True,
            actions={
                "ask_user": RuntimeControlActionConfig(enabled=True, exposed=False),
                "semantic_plan_update": RuntimeControlActionConfig(enabled=True, exposed=True),
            },
        )
    )

    context = executor.model_context()

    assert context["available"] == ["semantic_plan_update"]
    assert context["contracts"] == {
        "semantic_plan_update": (
            "After multiple action-observation cycles, request when confidence is low that the current goal, "
            "done_when, or continue_when still fit observed progress and user intent. Name the doubtful "
            "fields and evidence; keep normal UAV steps if continuing."
        )
    }
    assert "ask_user" not in context["contracts"]


def test_runtime_control_executor_recommends_semantic_contract_review_after_multiple_cycles():
    executor = RuntimeControlExecutor(
        RuntimeControlConfig(
            enabled=True,
            actions={
                "ask_user": RuntimeControlActionConfig(enabled=True, exposed=False),
                "semantic_plan_update": RuntimeControlActionConfig(enabled=True, exposed=True),
            },
        )
    )

    context = executor.model_context(
        {
            "phase_execution_summary": {
                "window_count": 4,
                "tool_facts": {"tools_executed": {"uav.hover": 2, "uav.move_body": 2}},
            },
        }
    )

    assert context["available"] == ["semantic_plan_update"]
    assert context["recommendations"] == {"semantic_plan_update": SEMANTIC_PLAN_UPDATE_RECOMMENDATION}


def test_runtime_control_recommendation_requires_available_action_and_no_pending():
    base = {
        "phase_execution_summary": {
            "window_count": 4,
            "tool_facts": {"tools_executed": {"uav.move_body": 4}},
        },
    }

    assert runtime_control_recommendations(available=[], **base) == {}
    assert (
        runtime_control_recommendations(
            available=["semantic_plan_update"],
            status={"semantic_plan_update": "pending"},
            **base,
        )
        == {}
    )
    assert (
        runtime_control_recommendations(
            available=["semantic_plan_update"],
            phase_execution_summary={
                "window_count": 3,
                "tool_facts": {"tools_executed": {"uav.move_body": 4}},
            },
        )
        == {}
    )
    assert (
        runtime_control_recommendations(
            available=["semantic_plan_update"],
            phase_execution_summary={
                "window_count": 4,
                "tool_facts": {"tools_executed": {"uav.move_body": 3}},
            },
        )
        == {}
    )
