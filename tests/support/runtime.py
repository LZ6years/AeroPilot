"""Runtime integration fixtures shared by product-contract tests."""

from tests.test_unified_tool_loop import (
    call,
    done,
    harness,
    move,
    public_runtime,
    run_loop,
)
from src.runtime.task.session import InstructionSession


def execute_read_only(runtime, state, scene, calls, public_intent="read"):
    """Exercise the production TurnExecutor read branch without a model call."""

    session = InstructionSession(
        scene=scene,
        agent_id="uav",
        state=state,
        memory=runtime._memory_for_state(state),
        transcript=runtime.agent_transcript,
    )
    return runtime.turn_executor._execute_native_read_calls(
        session, public_intent, calls, "decision_0001"
    )

__all__ = [
    "call",
    "done",
    "execute_read_only",
    "harness",
    "move",
    "public_runtime",
    "run_loop",
]
