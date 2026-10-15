"""Control routing rejects unavailable tools without mutation.

Plan/terminal application and freshness live in builtin_control_tools tests.
Asynchronous semantic-update/recommendation tests retire with that protocol.
"""
import pytest
from src.agent.mission_state import MissionState
from src.agent.transcript import AgentToolCall
from src.runtime.control.executor import execute_builtin_control
from src.tools.registry import default_registry


@pytest.mark.parametrize("tool,args", [
    ("runtime.unknown", {}), ("uav.hover", {}),
    ("runtime.update_plan", {"steps": [], "updates": []}),
    ("runtime.report_completed", {"reason": "done", "evidence_refs": ["future"]}),
])
def test_control_rejection_is_closed_without_state_mutation(tool, args):
    state = MissionState("run", "task")
    before = state.to_dict()
    result = execute_builtin_control(AgentToolCall("call", tool, args), state, default_registry(), ())
    assert result.status == "rejected"
    assert result.tool_call_id == "call" and result.executed_args == {}
    assert state.to_dict() == before
