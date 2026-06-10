"""Test model action contract repair data structures."""

from __future__ import annotations

from src.agent.planning.model_repair_planner import ModelRepairPlannerError, RepairContext
from src.agent.planning.tool_contract_validator import ToolContractViolation


def make_planner_context() -> dict:
    return {
        "active_instruction": "turn right and find the lamp",
        "current_phase": {"name": "orient_right"},
        "semantic_goal": {"final_state": "hover facing target"},
        "fragments": [{"name": "instruction"}, {"name": "pose"}],
        "allowed_action_constraints": {
            "planner_tools": ["uav.hover", "uav.turn_relative", "uav.move_body"],
            "forbidden_tools": ["env.observe"],
        },
        "observation_policy": {"mode": "automatic", "modalities": ["rgb", "depth", "pose"]},
        "safety_constraints": {"max_yaw_scan_accum_deg": 480.0},
    }


def test_repair_context_keeps_contract_fields_and_small_context_summary():
    violation = ToolContractViolation(
        code="forbidden_tool",
        message="Tool is forbidden in model action plan: env.observe",
        step_index=0,
        tool="env.observe",
        field="tool",
        actual="env.observe",
    )
    invalid_plan = {
        "phase": "orient_right",
        "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        "phase_transition": {"decision": "continue_current_phase", "reason": "bad"},
    }

    repair_context = RepairContext.from_context(
        attempt=1,
        max_attempts=3,
        phase="orient_right",
        invalid_plan=invalid_plan,
        violation=violation,
        previous_violations=[violation.to_dict()],
        planner_context=make_planner_context(),
    )
    data = repair_context.to_dict()

    assert data["attempt"] == 1
    assert data["max_attempts"] == 3
    assert data["phase"] == "orient_right"
    assert data["invalid_plan"] == invalid_plan
    assert data["violation"]["code"] == "forbidden_tool"
    assert data["allowed_action_constraints"]["planner_tools"] == [
        "uav.hover",
        "uav.turn_relative",
        "uav.move_body",
    ]
    assert data["observation_policy"]["mode"] == "automatic"
    assert "phase_transition_policy" not in data
    assert data["context_summary"]["fragment_names"] == ["instruction", "pose"]
    assert "flight_memory" not in data["context_summary"]


def test_model_repair_planner_error_to_dict_exposes_attempts_and_violations():
    error = ModelRepairPlannerError(
        "repair failed",
        initial_plan={"phase": "p", "steps": []},
        final_plan={"phase": "p", "steps": [{"kind": "observe"}]},
        repair_attempts=[{"attempt": 1, "status": "failed"}],
        violations=[{"code": "observe_managed_by_runtime"}],
        max_repair_attempts=3,
        reason="max_repair_attempts_exhausted",
    )

    data = error.to_dict()

    assert data["reason"] == "max_repair_attempts_exhausted"
    assert data["initial_plan"]["phase"] == "p"
    assert data["final_plan"]["steps"][0]["kind"] == "observe"
    assert data["repair_attempts"][0]["attempt"] == 1
    assert data["violations"][0]["code"] == "observe_managed_by_runtime"
    assert data["max_repair_attempts"] == 3
