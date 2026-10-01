"""Request-only progress guidance: timing, replay isolation and config boundaries."""

from dataclasses import asdict

import pytest

from src.agent.config import ModelConfig
from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from tests.support.planner import (
    BaselinePlanner,
    GUIDANCE_BASELINE,
    StubPlanner,
    make_transcript,
    response_with_calls,
    guidance_probe,
    treatments,
)
from tests.support.runtime import call, done, harness, move, run_loop


@pytest.mark.parametrize("index,expected", [(None, False), (1, False), (5, False),
                                           (6, True), (7, False), (10, False),
                                           (11, True), (16, True)])
def test_reminder_boundary_and_history_isolation(tmp_path, index, expected):
    planner = StubPlanner(response_with_calls(("turn", "uav_turn_relative", {"yaw_deg": 10})))
    registry = default_registry()
    transcript = make_transcript(tmp_path)
    before = transcript.to_dict()
    history = transcript.provider_messages(registry)
    planner.plan_turn(transcript, registry, decision_index=index)
    messages = planner.last_exchange["request"]["messages"]
    assert sum(m.get("content") == planner.task_progress_reminder for m in messages) == int(expected)
    if expected:
        assert messages[-1] == {"role": "system", "content": planner.task_progress_reminder}
        assert messages[-len(history)-2:-2] == history
        assert messages[-2]["content"].startswith("CURRENT TASK STATE\n")
    else:
        assert messages[-len(history)-1:-1] == history
        assert messages[-1]["content"].startswith("CURRENT TASK STATE\n")
    assert transcript.to_dict() == before
    # Rebuilding the next request cannot carry forward the previous reminder.
    planner.plan_turn(transcript, registry, decision_index=7)
    assert all(m.get("content") != planner.task_progress_reminder for m in planner.last_request["messages"])


@pytest.mark.parametrize("interval,index,expected", [(0, 6, False), (2, 3, True), (2, 4, False)])
def test_configured_interval(tmp_path, interval, index, expected):
    planner = StubPlanner(response_with_calls(("turn", "uav_turn_relative", {"yaw_deg": 10})),
                          task_progress_reminder_interval=interval)
    planner.plan_turn(make_transcript(tmp_path), default_registry(), decision_index=index)
    assert (planner.last_request["messages"][-1].get("content") == planner.task_progress_reminder) == expected


def test_config_default_env_summary_and_transport_boundary():
    assert ModelConfig.from_env({}).task_progress_reminder_interval == 5
    config = ModelConfig.from_env({"FLYAGENT_AGENT_TASK_PROGRESS_REMINDER_INTERVAL": "0"})
    assert config.redacted_summary()["task_progress_reminder_interval"] == 0
    assert "task_progress_reminder_interval" not in asdict(config.to_provider_settings())


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_interval(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        ModelConfig(task_progress_reminder_interval=value)


def test_rejected_read_and_physical_turns_share_runtime_clock(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [
        [call("bad", "knowledge.load", name="missing")],
        [call("read", "knowledge.load", name="guide")],
        [move("move")], [done()],
    ])
    run_loop(runtime, state, scene)
    assert state.completed
    assert runtime.planner.decision_indices == [1, 2, 3, 4]
    assert backend.observations == 2


def test_protocol_rejection_counts_as_a_decision(tmp_path):
    runtime, state, scene, _ = harness(tmp_path, [
        [call("bad", "uav.turn_relative", yaw_deg=999)],
        [move("move")], [done()],
    ])
    run_loop(runtime, state, scene)
    assert state.completed
    assert runtime.planner.decision_indices == [1, 2, 3]
    assert any(e.payload.status == "rejected" for e in runtime.agent_transcript.items if e.kind == "tool_result")


def test_probe_changes_only_guidance_and_preserves_runtime_schema(monkeypatch):
    import copy
    import json
    baseline = json.loads(GUIDANCE_BASELINE.read_text(encoding="utf-8"))
    from types import SimpleNamespace
    frozen_tools = copy.deepcopy(baseline['tools'])
    monkeypatch.setattr(guidance_probe, 'default_registry', lambda: SimpleNamespace(model_specs=lambda: [
        SimpleNamespace(to_provider_tool=lambda t=t: copy.deepcopy(t)) for t in frozen_tools]))
    payload = {"messages": [{"role": "system", "content": baseline["system_prompt"]},
                            {"role": "user", "content": "recorded observation"}],
               "tools": baseline["tools"]}
    turn_tool = next(t for t in payload["tools"] if t["function"]["name"] == "uav_turn_relative")
    turn_tool["function"]["parameters"]["properties"]["yaw_deg"]["maximum"] = 90
    before = copy.deepcopy(payload)
    first = treatments(payload, 1)
    sixth = treatments(payload, 6)
    assert payload == before and first["baseline"] == before
    assert first["guidance"] == first["guidance_reminder"]
    assert sixth["guidance_reminder"]["messages"][:-1] == sixth["guidance"]["messages"]
    for variant in sixth.values():
        assert [t["function"]["parameters"] for t in variant["tools"]] == [t["function"]["parameters"] for t in before["tools"]]


def test_reminder_follows_complete_tool_result_pairs(tmp_path):
    from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult
    transcript = make_transcript(tmp_path)
    transcript.append_assistant(AssistantTurn("old", "Turn", [
        AgentToolCall("old_call", "uav.turn_relative", {"yaw_deg": 10})]))
    transcript.append_tool_result(ToolResult("old_call", "uav.turn_relative", "ok"))
    planner = StubPlanner(response_with_calls(("new_call", "uav_turn_relative", {"yaw_deg": 10})))
    registry = default_registry()
    expected_history = transcript.provider_messages(registry)
    planner.plan_turn(transcript, registry, decision_index=6)
    assert planner.last_request["messages"][-len(expected_history)-2:-2] == expected_history
    assert expected_history[-1]["role"] == "tool"
    assert planner.last_request["messages"][-1]["role"] == "system"


@pytest.mark.parametrize("index", [0, -1, True, 1.5])
def test_invalid_decision_index(tmp_path, index):
    planner = StubPlanner(response_with_calls(("turn", "uav_turn_relative", {"yaw_deg": 10})))
    with pytest.raises(ValueError, match="decision_index"):
        planner.plan_turn(make_transcript(tmp_path), default_registry(), decision_index=index)
    assert planner.payloads == []


def test_simulation_baseline_preserves_actual_safety_bounds_and_original_registry(tmp_path):
    import json
    from src.uav.config import UavSafetyConfig
    planner = BaselinePlanner(ModelConfig(model_name="stub", api_key="test-key"))
    planner._request = lambda payload: response_with_calls(("turn", "uav_turn_relative", {"yaw_deg": 10}))
    registry = default_registry(
        PluginRegistry.from_config(),
        safety_config=UavSafetyConfig(max_yaw_step_deg=90),
    )
    before = [s.to_provider_tool() for s in registry.model_specs()]
    planner.plan_turn(make_transcript(tmp_path), registry, decision_index=6)
    saved = json.loads(GUIDANCE_BASELINE.read_text(encoding="utf-8"))
    actual = planner.last_request["tools"]
    assert [t["function"]["description"] for t in actual] == [t["function"]["description"] for t in saved["tools"]]
    assert [t["function"]["parameters"] for t in actual] == [t["function"]["parameters"] for t in before]
    assert before == [s.to_provider_tool() for s in registry.model_specs()]
    assert planner.last_request["messages"][-1]["content"].startswith("CURRENT TASK STATE\n")
