"""Native arguments replace ShortHorizonPlan validation.

Batch isolation/quotas are covered in builtin_control_tools. Physical limits
remain Safety tests, not a duplicate Phase validator.
"""
import pytest
from src.agent.transcript import AgentToolCall, AssistantTurn
from src.agent.planning.tool_contract_validator import NativeTurnValidationError, validate_native_turn
from src.tools.registry import default_registry


@pytest.mark.parametrize("name,args", [
    ("uav.unknown", {}), ("env.observe", {}), ("uav.land", {}),
    ("uav.hover", {}),
    ("uav.turn_relative", {"yaw_deg": "right"}),
    ("uav.move_body", {"direction": "forward", "duration": 1}),
    ("uav.move_body", {"vx": 1, "vy": 0, "vz": 0}),
])
def test_invalid_native_call_has_source_diagnostic(name, args):
    registry = default_registry()
    turn = AssistantTurn("turn", "Act", [AgentToolCall("bad", name, args)])
    with pytest.raises(NativeTurnValidationError) as failure:
        validate_native_turn(turn, registry, allowed_names={"uav.hover", "uav.turn_relative", "uav.move_body"},
                             latest_observation={"observation_id": "obs1"})
    assert failure.value.diagnostic["source_tool_call_id"] == "bad"
    assert failure.value.diagnostic["scope"] == "call"


def test_native_physical_turn_needs_neither_phase_nor_commit():
    registry = default_registry()
    turn = AssistantTurn("turn", "Turn and observe", [
        AgentToolCall("turn1", "uav.turn_relative", {"yaw_deg": -15}),
    ])
    assert validate_native_turn(turn, registry, allowed_names=set(registry.names()),
                                latest_observation={"observation_id": "obs1"}) == "physical"
