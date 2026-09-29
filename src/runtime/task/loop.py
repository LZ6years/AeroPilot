"""Instruction-loop orchestration and guard policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.planning.model_action_planner import AgentProtocolError
from src.runtime.task.session import InstructionSession
from src.runtime.task.turn_executor import TurnExecutor


@dataclass(frozen=True)
class TaskLoopGuard:
    max_consecutive_protocol_rejections: int

    def __post_init__(self) -> None:
        if self.max_consecutive_protocol_rejections < 1:
            raise ValueError("protocol rejection limit must be positive")

    def stop_reason(
        self, state: MissionState, *, consecutive_rejections: int
    ) -> str | None:
        if state.physical_window_count >= state.max_physical_windows:
            return "instruction_physical_window_limit"
        if state.nonphysical_round_count >= state.max_nonphysical_rounds:
            return "nonphysical_round_limit"
        if consecutive_rejections >= self.max_consecutive_protocol_rejections:
            return "consecutive_protocol_rejections"
        return None


class InstructionLoop:
    """Own planner iteration, guard checks, and turn sequencing."""

    def __init__(
        self,
        owner: Any,
        guard: TaskLoopGuard,
        turn_executor: TurnExecutor,
    ) -> None:
        self.owner = owner
        self.guard = guard
        self.turn_executor = turn_executor

    def run(self, session: InstructionSession) -> None:
        runtime = self.owner
        state = session.state
        registry = runtime.tools.registry
        allowed = self.turn_executor.allowed_tools(session.scene, registry)
        while not state.completed and not state.stopped:
            runtime._write_state(state)
            stop_reason = self.guard.stop_reason(
                state,
                consecutive_rejections=session.consecutive_rejections,
            )
            if stop_reason is not None:
                state.mark_stopped(stop_reason, "runtime_loop_guard")
                break

            decision_id = session.next_decision_id()
            runtime._decision_index = session.decision_index
            try:
                turn = runtime.planner.plan_turn(
                    runtime.agent_transcript,
                    registry,
                    allowed_tool_names=allowed,
                    decision_index=session.decision_index,
                )
            except AgentProtocolError as exc:
                runtime._capture_instruction_exchange(
                    decision_id, error_type=type(exc).__name__
                )
                if exc.rejected_turn:
                    runtime.agent_transcript.append_provider_rejection(
                        exc.rejected_turn
                    )
                runtime.agent_transcript.append_protocol_feedback(str(exc))
                state.record_nonphysical_round()
                session.consecutive_rejections += 1
                runtime.journal.write_turn(
                    "tool_turn_rejected",
                    {"decision_id": decision_id, "error": str(exc)},
                )
                continue
            except Exception as exc:
                runtime._capture_instruction_exchange(
                    decision_id, error_type=type(exc).__name__
                )
                raise

            runtime._capture_instruction_exchange(decision_id, turn=turn)
            if not self.turn_executor.call_ids_are_fresh(
                turn, session.seen_call_ids
            ):
                runtime.agent_transcript.append_provider_rejection(
                    {
                        "error": "tool call IDs must be instruction-wide unique",
                        "content_present": bool(turn.public_intent),
                        "content": turn.public_intent or None,
                        "tool_calls": [
                            call.to_dict() for call in turn.tool_calls
                        ],
                    }
                )
                runtime.agent_transcript.append_protocol_feedback(
                    "Use fresh non-empty tool_call_ids for every call."
                )
                state.record_nonphysical_round()
                session.consecutive_rejections += 1
                continue

            history_boundary = len(runtime.agent_transcript.items)
            runtime.agent_transcript.append_assistant(turn)
            self.turn_executor.execute(
                session,
                turn,
                allowed_tools=allowed,
                decision_id=decision_id,
                history_boundary=history_boundary,
            )

        if hasattr(runtime, "_current_turn_agent_messages"):
            session.latest_turn_messages = runtime.latest_turn_agent_messages()
        runtime._write_state(state)
