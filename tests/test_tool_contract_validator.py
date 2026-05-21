"""Test the explicit tool contract used by model action planning."""

from __future__ import annotations

import pytest

from src.agent.planning import (
    ActionWindow,
    ShortHorizonPlan,
    ToolContractValidationError,
    ToolContractValidator,
    ToolContractViolation,
)
from src.agent.phase import ADVANCE_TO_NEXT_PHASE
from src.agent.planning.schemas import PlannedStep


def make_context() -> dict:
    return {
        "allowed_action_constraints": {
            "planner_tools": ["uav.hover", "uav.turn_relative", "uav.move_body"],
            "action_schemas": {
                "uav.hover": {"args": {}},
                "uav.turn_relative": {
                    "args": {"yaw_deg": "number"},
                    "limits": {"yaw_deg_min": -30.0, "yaw_deg_max": 30.0},
                },
                "uav.move_body": {
                    "args_one_of": [
                        {"direction": "forward|backward|left|right|up|down", "duration": "number"},
                        {"vx": "number", "vy": "number", "vz": "number", "duration": "number"},
                    ],
                    "limits": {
                        "abs_vx_vy_max": 6.0,
                        "abs_vz_max": 3.0,
                        "duration_max": 8.0,
                    },
                },
            },
            "forbidden_tools": ["env.observe", "uav.get_frames", "uav.land"],
        }
    }


def make_plan(*steps: PlannedStep, phase: str = "scan_left") -> ShortHorizonPlan:
    return ShortHorizonPlan(phase, list(steps))


def assert_violation(exc_info: pytest.ExceptionInfo[ToolContractValidationError], code: str) -> dict:
    assert exc_info.value.violation.code == code
    data = exc_info.value.to_dict()
    assert data["code"] == code
    assert data["message"]
    return data


def test_tool_contract_validator_accepts_legal_tool_plan():
    plan = make_plan(
        PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": -15.0}, "scan left"),
        PlannedStep.tool_step("uav.move_body", {"direction": "forward", "duration": 1.5}, "small move"),
    )

    result = ToolContractValidator().validate(plan, "scan_left", make_context())

    assert result is plan
    assert result.phase_transition.decision == ADVANCE_TO_NEXT_PHASE
    assert result.action_window.scope == "current_phase_only"


def test_tool_contract_violation_to_dict_keeps_debug_fields():
    violation = ToolContractViolation(
        code="tool_not_allowed",
        message="Tool is not allowed in model action plan: uav.fake",
        step_index=0,
        tool="uav.fake",
        field="tool",
        expected=["uav.hover"],
        actual="uav.fake",
    )

    assert violation.to_dict() == {
        "code": "tool_not_allowed",
        "message": "Tool is not allowed in model action plan: uav.fake",
        "step_index": 0,
        "tool": "uav.fake",
        "field": "tool",
        "expected": ["uav.hover"],
        "actual": "uav.fake",
    }


def test_tool_contract_validator_rejects_observe_step():
    plan = make_plan(PlannedStep.observe_step())

    with pytest.raises(ToolContractValidationError, match="runtime perception policy") as exc_info:
        ToolContractValidator().validate(plan, "scan_left", make_context())

    data = assert_violation(exc_info, "observe_managed_by_runtime")
    assert data["field"] == "kind"
    assert data["actual"] == "observe"


def test_tool_contract_validator_rejects_forbidden_tool():
    plan = make_plan(PlannedStep.tool_step("env.observe", {"modalities": ["rgb"]}, "bad"))

    with pytest.raises(ToolContractValidationError, match="forbidden") as exc_info:
        ToolContractValidator().validate(plan, "scan_left", make_context())

    data = assert_violation(exc_info, "forbidden_tool")
    assert data["tool"] == "env.observe"


def test_tool_contract_validator_rejects_context_tool_without_supported_schema_validator():
    context = make_context()
    context["allowed_action_constraints"]["planner_tools"].append("uav.custom")
    plan = make_plan(PlannedStep.tool_step("uav.custom", {}, "bad"))

    with pytest.raises(ToolContractValidationError, match="no validator") as exc_info:
        ToolContractValidator().validate(plan, "scan_left", context)

    assert_violation(exc_info, "tool_schema_not_supported")


@pytest.mark.parametrize(
    ("plan", "code", "message"),
    [
        (
            make_plan(PlannedStep.tool_step("uav.fake", {}, "bad")),
            "tool_not_allowed",
            "not allowed",
        ),
        (
            make_plan(PlannedStep.tool_step("uav.hover", {"duration": 1.0}, "bad")),
            "hover_args_not_empty",
            "uav.hover args must be",
        ),
        (
            make_plan(PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": -45.0}, "bad")),
            "turn_yaw_below_limit",
            "below limit",
        ),
        (
            make_plan(PlannedStep.tool_step("uav.move_body", {"x_m": 4.0, "duration_sec": 2.0}, "bad")),
            "move_body_args_schema_error",
            "uav.move_body args must be exactly",
        ),
        (
            make_plan(PlannedStep.tool_step("uav.move_body", {"direction": "diagonal", "duration": 1.0}, "bad")),
            "move_body_direction_invalid",
            "Unsupported uav.move_body direction",
        ),
        (
            make_plan(PlannedStep.tool_step("uav.move_body", {"vx": 7.0, "vy": 0.0, "vz": 0.0, "duration": 1.0}, "bad")),
            "move_body_planar_velocity_limit_exceeded",
            "planar velocity exceeds",
        ),
        (
            make_plan(PlannedStep.tool_step("uav.move_body", {"vx": 0.0, "vy": 0.0, "vz": 4.0, "duration": 1.0}, "bad")),
            "move_body_vertical_velocity_limit_exceeded",
            "vertical velocity exceeds",
        ),
        (
            make_plan(PlannedStep.tool_step("uav.move_body", {"direction": "forward", "duration": 9.0}, "bad")),
            "move_body_duration_limit_exceeded",
            "duration exceeds",
        ),
    ],
)
def test_tool_contract_validator_rejects_invalid_contract_cases(
    plan: ShortHorizonPlan,
    code: str,
    message: str,
):
    with pytest.raises(ToolContractValidationError, match=message) as exc_info:
        ToolContractValidator().validate(plan, "scan_left", make_context())

    assert_violation(exc_info, code)


def test_tool_contract_validator_rejects_phase_mismatch():
    plan = make_plan(PlannedStep.tool_step("uav.hover", {}, "hold"), phase="wrong_phase")

    with pytest.raises(ToolContractValidationError, match="phase mismatch") as exc_info:
        ToolContractValidator().validate(plan, "scan_left", make_context())

    data = assert_violation(exc_info, "phase_mismatch")
    assert data["expected"] == "scan_left"
    assert data["actual"] == "wrong_phase"


def test_tool_contract_validator_rejects_invalid_action_window_scope():
    plan = ShortHorizonPlan(
        "scan_left",
        [PlannedStep.tool_step("uav.hover", {}, "hold")],
        action_window=ActionWindow(
            intent="hold in current phase",
            scope="whole_task",
            expected_effect="stay stable",
        ),
    )

    with pytest.raises(ToolContractValidationError, match="current_phase_only") as exc_info:
        ToolContractValidator().validate(plan, "scan_left", make_context())

    assert_violation(exc_info, "action_window_scope_invalid")


def test_tool_contract_validator_rejects_empty_action_window_descriptions():
    plan = ShortHorizonPlan(
        "scan_left",
        [PlannedStep.tool_step("uav.hover", {}, "hold")],
        action_window=ActionWindow(intent="", expected_effect=""),
    )

    with pytest.raises(ToolContractValidationError, match="intent") as exc_info:
        ToolContractValidator().validate(plan, "scan_left", make_context())

    assert_violation(exc_info, "action_window_intent_missing")


def test_tool_contract_validator_rejects_multi_step_when_intermediate_observation_required():
    plan = ShortHorizonPlan(
        "scan_left",
        [
            PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": -10.0}, "scan"),
            PlannedStep.tool_step("uav.move_body", {"direction": "forward", "duration": 1.0}, "move"),
        ],
        action_window=ActionWindow(
            intent="scan then move",
            requires_intermediate_observation=True,
            expected_effect="new visual evidence is needed before the move",
        ),
    )

    with pytest.raises(ToolContractValidationError, match="exactly one step") as exc_info:
        ToolContractValidator().validate(plan, "scan_left", make_context())

    assert_violation(exc_info, "action_window_intermediate_observation_requires_single_step")
