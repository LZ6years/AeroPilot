"""Validate runtime control requests without dispatching UAV tools.
"""

from __future__ import annotations

from src.agent.transcript import AgentToolCall, ToolResult, TranscriptEvent
from src.tools.registry import ToolRegistry
from src.tools.validation import validate_tool_args
from src.runtime.control.actions.task_plan import PlanState, apply_plan_update
from src.runtime.control.actions.terminal import (
    TerminalState, apply_terminal, environmental_evidence_refs,
)


def execute_builtin_control(
    call: AgentToolCall,
    state: PlanState | TerminalState,
    registry: ToolRegistry,
    history: tuple[TranscriptEvent, ...],
) -> ToolResult:
    """Execute a validated exclusive control turn using the unique ToolRegistry.

    This function owns no tool definitions, model calls, observations or cleanup.
    Runtime persists its result and performs terminal hover/observation itself.
    Handler validation precedes state mutation; expected invalid requests close
    with a rejected ToolResult, not an unpaired exception.
    """
    try:
        spec = registry.get(call.tool_name)
        if (spec.runtime.effect != "runtime_control" or not spec.runtime.model_exposed
                or spec.runtime.batching != "single" or spec.runtime.result_timing != "immediate"):
            raise ValueError("tool is not an exposed immediate control")
        validate_tool_args(call.args, spec.input_schema)
        if call.tool_name == "runtime.update_plan":
            data = apply_plan_update(state, call.args)
        elif call.tool_name in {"runtime.report_completed", "runtime.report_blocked"}:
            data = apply_terminal(
                state, call.args, completed=call.tool_name == "runtime.report_completed",
                available_evidence=environmental_evidence_refs(history, registry),
            )
        else:
            raise ValueError("no built-in control handler for this tool")
    except (KeyError, ValueError) as exc:
        return ToolResult(
            call.tool_call_id, call.tool_name, "rejected", str(exc),
            data={"rejection": {"code": "control_rejected", "scope": "call",
                                "source_tool_call_id": call.tool_call_id}},
            requested_args=dict(call.args),
        )
    return ToolResult(call.tool_call_id, call.tool_name, "ok", "Control applied.", data=data,
                      requested_args=dict(call.args), executed_args=dict(call.args))
