"""Planner/tool loop orchestration for one TaskSession."""

from __future__ import annotations

from src.agent.planning.model_action_planner import AgentProtocolError
from src.runtime.task.session import TaskSession
from src.runtime.task.turn_executor import TurnExecutor


class TaskLoop:
    """Own planner iteration, guard checks, and accepted-turn sequencing."""

    def __init__(
        self,
        planner,
        registry,
        turn_executor: TurnExecutor,
        journal,
        evidence,
        *,
        max_consecutive_protocol_rejections: int,
    ) -> None:
        if max_consecutive_protocol_rejections < 1:
            raise ValueError("protocol rejection limit must be positive")
        self.planner = planner
        self.registry = registry
        self.turn_executor = turn_executor
        self.journal = journal
        self.evidence = evidence
        self.max_consecutive_protocol_rejections = (
            max_consecutive_protocol_rejections
        )

    def run(self, session: TaskSession) -> None:
        state = session.state
        allowed = self.turn_executor.allowed_tools(session.scene, self.registry)
        while not state.completed and not state.stopped:
            if not self._control_boundary(session, "before_planning"):
                break
            current_task_state = session.current_task_state()
            self.evidence.write_state(session)
            stop_reason = self._stop_reason(session)
            if stop_reason is not None:
                state.mark_stopped(stop_reason, "runtime_loop_guard")
                break

            decision_id = session.next_decision_id()
            self.journal.event("planner_request_started", {
                "decision_id": decision_id,
                "instruction_id": session.transcript.instruction_id,
                "observation_id": session.transcript.latest_observation_id(),
            })
            try:
                turn = self.planner.plan_turn(
                    session.transcript,
                    self.registry,
                    allowed_tool_names=allowed,
                    decision_index=session.decision_index,
                    history_projection_state=session.history_projection_state,
                    history_compaction_coordinator=(
                        session.history_compaction_coordinator
                    ),
                    current_task_state=current_task_state,
                )
            except AgentProtocolError as exc:
                self.journal.capture_provider_exchange(
                    self.planner,
                    session,
                    decision_id,
                    error_type=type(exc).__name__,
                )
                if exc.rejected_turn:
                    session.transcript.append_provider_rejection(
                        exc.rejected_turn
                    )
                session.transcript.append_protocol_feedback(str(exc))
                state.record_nonphysical_round()
                session.consecutive_rejections += 1
                self.journal.write_turn(
                    "tool_turn_rejected",
                    {"decision_id": decision_id, "error": str(exc)},
                )
                continue
            except Exception as exc:
                self.journal.capture_provider_exchange(
                    self.planner,
                    session,
                    decision_id,
                    error_type=type(exc).__name__,
                )
                raise

            self.journal.capture_provider_exchange(
                self.planner, session, decision_id, turn=turn
            )
            if not self._control_boundary(session, "before_execution"):
                self.journal.event("operator_decision_discarded", {
                    "decision_id": decision_id,
                    "instruction_id": session.transcript.instruction_id,
                    "reason": "operator_stop_before_execution",
                })
                break
            if not self.turn_executor.call_ids_are_fresh(
                turn, session.seen_call_ids
            ):
                session.transcript.append_provider_rejection(
                    {
                        "error": "tool call IDs must be instruction-wide unique",
                        "content_present": bool(turn.public_intent),
                        "content": turn.public_intent or None,
                        "tool_calls": [
                            call.to_dict() for call in turn.tool_calls
                        ],
                    }
                )
                session.transcript.append_protocol_feedback(
                    "Use fresh non-empty tool_call_ids for every call."
                )
                state.record_nonphysical_round()
                session.consecutive_rejections += 1
                continue

            history_boundary = len(session.transcript.items)
            session.transcript.append_assistant(turn)
            self.turn_executor.execute(
                session,
                turn,
                allowed_tools=allowed,
                decision_id=decision_id,
                history_boundary=history_boundary,
            )
            if not state.completed and not state.stopped:
                if not self._control_boundary(session, "after_turn"):
                    break
        self.evidence.write_state(session)

    def _control_boundary(self, session: TaskSession, boundary: str) -> bool:
        """Only the flight worker acknowledges user control and changes state."""

        if session.control is None:
            return True
        def changed(payload):
            self.journal.event("operator_control", {
                **payload, "instruction_id": session.transcript.instruction_id,
            })
        if session.control.checkpoint(boundary, changed):
            return True
        session.state.mark_stopped(
            session.control.snapshot()["reason"], "operator_control"
        )
        return False

    def _stop_reason(self, session: TaskSession) -> str | None:
        state = session.state
        if state.physical_window_count >= state.max_physical_windows:
            return "instruction_physical_window_limit"
        if state.nonphysical_round_count >= state.max_nonphysical_rounds:
            return "nonphysical_round_limit"
        if (
            session.consecutive_rejections
            >= self.max_consecutive_protocol_rejections
        ):
            return "consecutive_protocol_rejections"
        return None
