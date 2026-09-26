"""Built-in tool declarations are data, not a second execution registry."""

import pytest
from dataclasses import replace
from types import SimpleNamespace

from src.tools.builtins import builtin_tools
from src.tools.validation import validate_tool_args
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult, TranscriptEvent
from src.runtime.control.actions.task_plan import apply_plan_update
from src.runtime.control.actions.terminal import apply_terminal, environmental_evidence_refs
from src.runtime.control.executor import execute_builtin_control
from src.tools.registry import ToolRegistry
from src.tools.schemas import ToolSpec, ToolRuntimePolicy
from src.agent.planning.tool_contract_validator import (
    NativeTurnValidationError, read_call_signature, validate_native_turn,
)


def state():
    return SimpleNamespace(task_plan=None, completed=False, stopped=False,
                           stopped_reason=None, stop_source=None, window_count=7)


def test_builtin_declarations_have_explicit_policies():
    specs = {spec.name: spec for spec in builtin_tools()}
    assert set(specs) == {"runtime.update_plan", "knowledge.load",
                          "runtime.report_completed", "runtime.report_blocked"}
    for spec in specs.values():
        assert spec.runtime.model_exposed
        assert spec.runtime.result_timing == "immediate"
        assert spec.wire_name == spec.name.replace(".", "_")
        assert "commit" not in spec.to_provider_tool()["function"]["description"]
        if spec.runtime.effect == "runtime_control":
            assert "only call" in spec.to_provider_tool()["function"]["description"]
    assert specs["knowledge.load"].runtime.effect == "read_only"
    assert specs["knowledge.load"].runtime.batching == "parallel_safe"
    assert specs["runtime.update_plan"].runtime.batching == "single"


def test_minimal_tool_arguments_and_terminal_evidence():
    specs = {spec.name: spec for spec in builtin_tools()}
    validate_tool_args({"name": "navigation"}, specs["knowledge.load"].input_schema)
    validate_tool_args({"explanation": "No plan needed", "steps": []},
                       specs["runtime.update_plan"].input_schema)
    validate_tool_args({"reason": "Arrived", "evidence_refs": ["obs_1"]},
                       specs["runtime.report_completed"].input_schema)
    with pytest.raises(ValueError):
        validate_tool_args({"reason": "Arrived", "evidence_refs": []},
                           specs["runtime.report_completed"].input_schema)


def test_plan_and_knowledge_descriptions_explain_selection_without_new_parameters():
    specs = {spec.name: spec for spec in builtin_tools()}
    plan = specs["runtime.update_plan"]
    knowledge = specs["knowledge.load"]
    assert "before the first physical action" in plan.description
    assert "simple single-step task can act directly" in plan.description
    assert "does not execute flight" in plan.description
    assert "provides methods" in knowledge.description
    assert "not current-scene observations or measurements" in knowledge.description
    assert set(plan.input_schema["properties"]) == {"explanation", "steps"}
    assert set(knowledge.input_schema["properties"]) == {"name", "reference_paths"}
    for spec in (plan, knowledge):
        wire = spec.to_provider_tool()["function"]
        assert wire["parameters"] == spec.input_schema
        assert spec.description in wire["description"]


def test_plan_application_does_not_reset_windows_or_complete_task():
    mission = state()
    payload = {"explanation": "Observed the endpoint", "steps": [
        {"id": "go", "description": "Reach the endpoint", "status": "completed"}]}
    result = apply_plan_update(mission, payload)
    assert mission.task_plan.revision == 1
    assert mission.window_count == 7
    assert not mission.completed
    assert result["previous_revision"] == 0
    result["plan"]["steps"][0]["description"] = "mutated"
    assert mission.task_plan.steps[0].description == "Reach the endpoint"
    before = mission.task_plan
    with pytest.raises(ValueError, match="disappear"):
        apply_plan_update(mission, {"explanation": "forget", "steps": []})
    assert mission.task_plan is before


def test_terminal_accepts_observed_evidence_without_semantic_judgment():
    mission = state()
    result = apply_terminal(mission, {"reason": "At the destination", "evidence_refs": ["obs1"]},
                            completed=True, available_evidence=frozenset({"obs1"}))
    assert mission.completed and not mission.stopped
    assert result["stop_source"] == "runtime.report_completed"
    with pytest.raises(ValueError, match="terminal"):
        apply_plan_update(mission, {"explanation": "another", "steps": []})
    with pytest.raises(ValueError, match="already terminal"):
        apply_terminal(mission, {}, completed=False, available_evidence=frozenset())


def test_blocked_needs_no_plan_or_phase_assessment():
    mission = state()
    result = apply_terminal(mission, {
        "reason": "No safe route is visible", "evidence_refs": ["obs1"],
        "blocking_condition": "unsafe_to_continue", "suggested_user_action": "Clarify route",
    }, completed=False, available_evidence=frozenset({"obs1"}))
    assert mission.stopped and not mission.completed
    assert mission.task_plan is None
    assert mission.stopped_reason == "agent_reported_blocked:unsafe_to_continue"
    assert result["suggested_user_action"] == "Clarify route"


@pytest.mark.parametrize("refs", [[], ["future"], ["obs1", "obs1"], [None]])
def test_invalid_terminal_evidence_leaves_state_unchanged(refs):
    mission = state()
    before = vars(mission).copy()
    with pytest.raises(ValueError):
        apply_terminal(mission, {"reason": "Done", "evidence_refs": refs},
                       completed=True, available_evidence=frozenset({"obs1"}))
    assert vars(mission) == before


def test_evidence_requires_returned_environment_result_not_intent_or_knowledge():
    registry = ToolRegistry()
    for spec in builtin_tools():
        registry.register(spec)
    registry.register(ToolSpec.from_input_schema("sensor", "sensor", "skill_capability", "", {},
                              ToolRuntimePolicy("read_only", "immediate", "parallel_safe", True)))
    calls = [AgentToolCall("loaded", "knowledge.load"),
             AgentToolCall("planned", "runtime.update_plan"),
             AgentToolCall("seen", "sensor"), AgentToolCall("pending", "sensor"),
             AgentToolCall("failed", "sensor")]
    events = [TranscriptEvent("user_instruction", {"instruction": "go"}),
              TranscriptEvent("state_checkpoint", {"observation_id": "obs1"}),
              TranscriptEvent("assistant_turn", AssistantTurn("turn1", "look", calls))]
    events.extend(TranscriptEvent("tool_result", result) for result in [
        ToolResult("loaded", "knowledge.load", "ok", result_ref="knowledge_ref"),
        ToolResult("planned", "runtime.update_plan", "ok"),
        ToolResult("seen", "sensor", "ok", result_ref="sensor_ref"),
        ToolResult("failed", "sensor", "error"), ToolResult("orphan", "sensor", "ok"),
    ])
    assert environmental_evidence_refs(events, registry) == {"obs1", "seen", "sensor_ref"}
    events.append(TranscriptEvent("user_instruction", {"instruction": "new task"}))
    assert environmental_evidence_refs(events, registry) == set()


@pytest.fixture
def loop_registry():
    registry = ToolRegistry()
    for spec in builtin_tools():
        registry.register(spec)
    schema = {"type": "object", "properties": {"value": {"type": "number"}},
              "required": ["value"], "additionalProperties": False}
    registry.register(ToolSpec.from_input_schema("motion", "motion", "uav_primitive", "", schema,
                              ToolRuntimePolicy("physical", "after_batch_observation", "ordered", True)))
    registry.register(ToolSpec.from_input_schema("sensor", "sensor", "skill_capability", "", schema,
                              ToolRuntimePolicy("read_only", "immediate", "parallel_safe", True, 2)))
    registry.register(ToolSpec.from_input_schema("connect", "connect", "uav_primitive", "", {},
                              ToolRuntimePolicy("lifecycle", "immediate", "single", False)))
    return registry


def validate_calls(registry, calls, **kwargs):
    return validate_native_turn(AssistantTurn("turn", "", calls), registry,
                                allowed_names=set(registry.names()),
                                latest_observation={"observation_id": "obs1"}, **kwargs)


def test_direct_ordered_physical_batch_needs_no_commit_or_plan(loop_registry):
    calls = [AgentToolCall(str(i), "motion", {"value": i}) for i in range(3)]
    assert validate_calls(loop_registry, calls) == "physical"
    assert [call.args["value"] for call in calls] == [0, 1, 2]
    with pytest.raises(NativeTurnValidationError) as failure:
        validate_calls(loop_registry, calls + [AgentToolCall("extra", "motion", {"value": 4})])
    assert failure.value.diagnostic["code"] == "physical_batch_invalid"


def test_read_batch_prevalidation_includes_all_calls_and_does_not_consume_quota(loop_registry):
    history = {read_call_signature("obs1", "sensor", {"value": 0})}
    calls = [AgentToolCall(str(i), "sensor", {"value": i}) for i in (1, 2)]
    with pytest.raises(NativeTurnValidationError) as failure:
        validate_calls(loop_registry, calls, executed_read_signatures=history)
    assert failure.value.diagnostic == {
        "code": "observation_call_limit", "scope": "call", "source_tool_call_id": "2"}
    assert len(history) == 1
    assert validate_calls(loop_registry, calls) == "read_only"


def test_read_call_dedup_and_instruction_call_ids(loop_registry):
    call = AgentToolCall("one", "sensor", {"value": 1})
    with pytest.raises(NativeTurnValidationError, match="reuse"):
        validate_calls(loop_registry, [call], executed_read_signatures={
            read_call_signature("obs1", "sensor", {"value": 1})})
    with pytest.raises(NativeTurnValidationError, match="unique"):
        validate_calls(loop_registry, [call], prior_call_ids={"one"})
    assert validate_calls(loop_registry, [call], executed_read_signatures={
        read_call_signature("old_obs", "sensor", {"value": 1})}) == "read_only"


def test_control_and_physical_turns_are_separate(loop_registry):
    plan = AgentToolCall("plan", "runtime.update_plan", {"explanation": "go", "steps": []})
    complete = AgentToolCall("done", "runtime.report_completed", {"reason": "arrived", "evidence_refs": ["obs1"]})
    move = AgentToolCall("move", "motion", {"value": 1})
    assert validate_calls(loop_registry, [plan]) == "runtime_control"
    for calls, code in (([plan, complete], "control_not_exclusive"), ([plan, move], "mixed_effects")):
        with pytest.raises(NativeTurnValidationError) as failure:
            validate_calls(loop_registry, calls)
        assert failure.value.diagnostic == {"code": code, "scope": "turn"}


def test_schema_error_attributes_the_actual_bad_call(loop_registry):
    calls = [AgentToolCall("good", "motion", {"value": 1}),
             AgentToolCall("bad", "motion", {"value": "one"})]
    with pytest.raises(NativeTurnValidationError) as failure:
        validate_calls(loop_registry, calls)
    assert failure.value.diagnostic["source_tool_call_id"] == "bad"
    assert failure.value.diagnostic["parameter_path"] == "args.value"


def test_empty_and_unexposed_turns_fail_before_execution(loop_registry):
    with pytest.raises(NativeTurnValidationError) as failure:
        validate_calls(loop_registry, [])
    assert failure.value.diagnostic["code"] == "empty_turn"
    with pytest.raises(NativeTurnValidationError) as failure:
        validate_calls(loop_registry, [AgentToolCall("connect", "connect")])
    assert failure.value.diagnostic["code"] == "tool_not_exposed"


def test_control_executor_uses_registry_and_closes_invalid_request(loop_registry):
    mission = state()
    result = execute_builtin_control(AgentToolCall("plan1", "runtime.update_plan", {
        "explanation": "Short route", "steps": [
            {"id": "go", "description": "Reach the gate", "status": "in_progress"}],
    }), mission, loop_registry, ())
    assert result.status == "ok" and result.data["plan"]["revision"] == 1
    rejected = execute_builtin_control(AgentToolCall("plan2", "runtime.update_plan", {
        "explanation": "Drop old progress", "steps": [],
    }), mission, loop_registry, ())
    assert rejected.tool_call_id == "plan2" and rejected.status == "rejected"
    assert mission.task_plan.revision == 1
    assert rejected.executed_args == {}
    assert rejected.data["rejection"]["source_tool_call_id"] == "plan2"


def test_control_executor_terminal_and_circular_evidence(loop_registry):
    mission = state()
    args = {"reason": "Reached the endpoint", "evidence_refs": ["complete1"]}
    call = AgentToolCall("complete1", "runtime.report_completed", args)
    history = (TranscriptEvent("user_instruction", {"instruction": "go"}),
               TranscriptEvent("state_checkpoint", {"observation_id": "obs1"}),
               TranscriptEvent("assistant_turn", AssistantTurn("turn1", "done", [call])))
    assert execute_builtin_control(call, mission, loop_registry, history).status == "rejected"
    assert not mission.completed
    valid = AgentToolCall("complete2", "runtime.report_completed", {**args, "evidence_refs": ["obs1"]})
    assert execute_builtin_control(valid, mission, loop_registry, history).status == "ok"
    assert mission.completed and mission.stop_source == "runtime.report_completed"


def test_prior_distance_call_does_not_require_current_requery():
    from src.tools.registry import default_registry
    registry = default_registry()
    call = AgentToolCall("range", "skill.visual_grounding.locate", {
        "observation_id": "old", "query": "taxi", "include_distance": True})
    history = (
        TranscriptEvent("user_instruction", {"instruction": "pass the taxi then turn"}),
        TranscriptEvent("state_checkpoint", {"observation_id": "old"}),
        TranscriptEvent("assistant_turn", AssistantTurn("t1", "measure", [call])),
        TranscriptEvent("tool_result", ToolResult("range", call.tool_name, "ok",
            data={"observation_id": "old", "candidates": []}, requested_args=call.args)),
        TranscriptEvent("state_checkpoint", {"observation_id": "current"}),
    )
    complete = AgentToolCall("done", "runtime.report_completed", {
        "reason": "Historical passage and subsequent turn", "evidence_refs": ["old", "current"]})
    assert execute_builtin_control(complete, state(), registry, history).status == "ok"
    assert "completion_evidence" not in registry.get(call.tool_name).metadata
    invalid = AgentToolCall("bad", "runtime.report_completed", {
        "reason": "arrived", "evidence_refs": ["invented"]})
    assert execute_builtin_control(invalid, state(), registry, history).status == "rejected"
