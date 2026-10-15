"""Diagnostic boundaries retain attribution and uncertain remote execution."""
import copy

import pytest

from src.agent.transcript import AgentToolCall, ToolResult
from src.runtime.task.turn_executor import TurnExecutor
from src.tools.diagnostics import Diagnostic, action_diagnostic, exception_diagnostic, legacy_result_diagnostic
from src.tools.validation import ToolArgumentError, validate_tool_args


def test_rejected_turn_attributes_source_and_affected_without_execution():
    calls = [AgentToolCall("bad", "runtime.update_plan"), AgentToolCall("read", "knowledge.load")]
    source = Diagnostic("plan_update_invalid", "validation", parameter_path="args.steps[0]",
                        source_tool_call_id="bad", scope="call", execution="not_started").to_dict()
    results = TurnExecutor.rejected_tool_results(calls, "unknown ID", rejection=source)
    assert [r["diagnostic"]["role"] for r in results] == ["source", "affected"]
    assert all(r["diagnostic"]["execution"] == "not_started" for r in results)
    assert all(r["diagnostic"]["source_tool_call_id"] == "bad" for r in results)
    assert source.get("role") is None


def test_timeout_after_dispatch_remains_unknown():
    diagnostic = exception_diagnostic(TimeoutError("RPC timed out"), phase="physical_execution", execution="unknown")
    result = ToolResult("flight", "uav.move", "error", diagnostic=diagnostic)
    assert result.to_dict()["diagnostic"]["execution"] == "unknown"
    _, promoted = action_diagnostic({}, call_id="flight", ok=False, phase="physical_execution")
    assert promoted["execution"] == "unknown"


def test_legacy_view_does_not_change_original_or_guess_execution():
    raw = {"status": "error", "data": {"error_type": "TimeoutError"}, "message": "timeout"}
    before = copy.deepcopy(raw)
    assert legacy_result_diagnostic(raw)["execution"] == "unknown"
    assert raw == before
    raw = {"status": "rejected", "data": {"rejection": {"code": "bad", "executed": False}}}
    assert legacy_result_diagnostic(raw)["execution"] == "not_started"


@pytest.mark.parametrize("value", [None, "ok"])
def test_nullable_schema_accepts_null_or_valid_type(value):
    validate_tool_args({"value": value}, {"type": "object", "properties": {
        "value": {"type": ["string", "null"], "minLength": 1}}})


@pytest.mark.parametrize("value", [False, 1, [], {}, ""])
def test_nullable_schema_still_checks_type_and_constraints(value):
    with pytest.raises(ToolArgumentError) as caught:
        validate_tool_args({"value": value}, {"type": "object", "properties": {
            "value": {"type": ["string", "null"], "minLength": 1}}})
    assert caught.value.path == "args.value"


def test_tool_diagnostic_requires_execution_but_task_diagnostic_can_omit_it():
    assert Diagnostic("failure").to_dict() == {"code": "failure", "phase": "unknown"}
    with pytest.raises(ValueError, match="requires execution"):
        ToolResult("id", "tool", "error", diagnostic=Diagnostic("failure").to_dict())


def test_union_validates_nested_objects_and_null_enum():
    schema = {"type": ["object", "null"], "additionalProperties": False,
              "properties": {"status": {"type": ["string", "null"], "enum": ["pending", None]}},
              "required": ["status"]}
    validate_tool_args(None, schema)
    validate_tool_args({"status": None}, schema)
    for value in ({}, {"status": "invented"}, {"status": None, "extra": 1}):
        with pytest.raises(ToolArgumentError):
            validate_tool_args(value, schema)
