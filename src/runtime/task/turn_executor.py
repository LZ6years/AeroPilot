"""Execution of one accepted provider-native tool turn."""

from __future__ import annotations

import copy
import time
from typing import Any

from src.agent.planning.schemas import PlannedStep
from src.agent.planning.tool_contract_validator import (
    NativeTurnValidationError,
    read_call_signature,
    validate_native_turn,
)
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult
from src.env_client import SceneInstance
from src.plugins.contracts.skill import (
    SKILL_ARTIFACT_REFS_KEY,
    SKILL_MODEL_INPUTS_KEY,
)
from src.runtime.control.executor import execute_builtin_control
from src.runtime.control.read_only_executor import ReadJob, ReadOnlyBatchExecutor
from src.runtime.task.outcomes import TurnOutcome
from src.runtime.task.physical_window import PhysicalWindowExecutor
from src.runtime.task.session import InstructionSession
from src.tools.registry import ToolRegistry


class TurnExecutor:
    """Validate and execute runtime-control, read-only, or physical turns."""

    def __init__(
        self, owner: Any, physical_windows: PhysicalWindowExecutor
    ) -> None:
        self.owner = owner
        self.physical_windows = physical_windows

    @staticmethod
    def allowed_tools(scene: SceneInstance, registry: ToolRegistry) -> set[str]:
        return {
            spec.name
            for spec in registry.model_specs()
            if spec.runtime.effect != "lifecycle"
            and (
                not spec.metadata.get("backend_action")
                or scene.capabilities is None
                or scene.capabilities.supports(spec.metadata["backend_action"])
            )
        }

    @staticmethod
    def call_ids_are_fresh(
        turn: AssistantTurn, seen_call_ids: set[str]
    ) -> bool:
        identities = [call.tool_call_id for call in turn.tool_calls]
        return bool(
            all(identities)
            and len(identities) == len(set(identities))
            and not (set(identities) & seen_call_ids)
        )

    def execute(
        self,
        session: InstructionSession,
        turn: AssistantTurn,
        *,
        allowed_tools: set[str],
        decision_id: str,
        history_boundary: int,
    ) -> TurnOutcome:
        """Execute one persisted AssistantTurn and update instruction state."""

        runtime = self.owner
        state = session.state
        registry = runtime.tools.registry
        ids = [call.tool_call_id for call in turn.tool_calls]
        try:
            effect = validate_native_turn(
                turn,
                registry,
                allowed_names=allowed_tools,
                latest_observation=state.latest_observation,
                prior_call_ids=session.seen_call_ids,
                executed_read_signatures=session.executed_read_signatures,
            )
        except NativeTurnValidationError as exc:
            session.seen_call_ids.update(ids)
            for record in self.rejected_tool_results(
                turn.tool_calls, str(exc), rejection=exc.diagnostic
            ):
                runtime.agent_transcript.append_tool_result(
                    ToolResult(
                        record["tool_call_id"],
                        record["tool_name"],
                        "rejected",
                        str(exc),
                        data=record["data"],
                        requested_args=record["requested_args"],
                    )
                )
            runtime.agent_transcript.append_protocol_feedback(
                str(exc) + ". No call in this turn executed."
            )
            state.record_nonphysical_round()
            session.consecutive_rejections += 1
            return TurnOutcome(kind="protocol_rejected")

        session.seen_call_ids.update(ids)
        if effect == "runtime_control":
            return self._execute_runtime_control(
                session, turn, decision_id, history_boundary
            )
        if effect == "read_only":
            return self._execute_read_only(session, turn, decision_id)
        return self._execute_physical(session, turn, decision_id)

    def _execute_runtime_control(
        self,
        session: InstructionSession,
        turn: AssistantTurn,
        decision_id: str,
        history_boundary: int,
    ) -> TurnOutcome:
        runtime = self.owner
        state = session.state
        state.record_nonphysical_round()
        history_before_turn = runtime.agent_transcript.snapshot_before(
            history_boundary
        )
        result = execute_builtin_control(
            turn.tool_calls[0], state, runtime.tools.registry, history_before_turn
        )
        runtime.agent_transcript.append_tool_result(result)
        runtime.journal.write_turn(
            "task_control_recorded",
            {"decision_id": decision_id, "tool_result": result.to_dict()},
        )
        if result.status == "rejected":
            runtime.agent_transcript.append_protocol_feedback(result.message)
            session.consecutive_rejections += 1
        else:
            session.consecutive_rejections = 0
            if turn.tool_calls[0].tool_name == "runtime.update_plan":
                runtime.checkpoints.publish(
                    state, runtime._memory_for_state(state), "plan_updated"
                )
        return TurnOutcome(
            kind="runtime_control",
            checkpoint_reason=(
                "plan_updated"
                if result.status != "rejected"
                and turn.tool_calls[0].tool_name == "runtime.update_plan"
                else None
            ),
            reset_rejections=result.status != "rejected",
        )

    def _execute_read_only(
        self,
        session: InstructionSession,
        turn: AssistantTurn,
        decision_id: str,
    ) -> TurnOutcome:
        runtime = self.owner
        state = session.state
        state.record_nonphysical_round()
        observation_id = str(
            state.latest_observation.get("observation_id") or ""
        )
        for call in turn.tool_calls:
            session.executed_read_signatures.add(
                read_call_signature(observation_id, call.tool_name, call.args)
            )
        loaded_before = copy.deepcopy(state.loaded_knowledge)
        results = self._execute_native_read_calls(
            session, turn.public_intent, turn.tool_calls, decision_id
        )
        if runtime.tools.knowledge_service is not None:
            state.loaded_knowledge = runtime.tools.knowledge_service.loaded_index()
        knowledge_changed = state.loaded_knowledge != loaded_before
        if knowledge_changed:
            runtime.checkpoints.publish(
                state, runtime._memory_for_state(state), "knowledge_loaded"
            )
        runtime.journal.write_turn(
            "read_only_turn_recorded",
            {"decision_id": decision_id, "tool_results": results},
        )
        all_ok = all(result["status"] == "ok" for result in results)
        if all_ok:
            session.consecutive_rejections = 0
        return TurnOutcome(
            kind="read_only",
            checkpoint_reason="knowledge_loaded" if knowledge_changed else None,
            reset_rejections=all_ok,
        )

    def _execute_physical(
        self,
        session: InstructionSession,
        turn: AssistantTurn,
        decision_id: str,
    ) -> TurnOutcome:
        runtime = self.owner
        state = session.state
        steps = [
            PlannedStep.tool_step(
                call.tool_name, dict(call.args), turn.public_intent
            )
            for call in turn.tool_calls
        ]
        safety = runtime.safety.review_steps(
            steps, state, registry=runtime.tools.registry
        )
        before = copy.deepcopy(state.latest_observation)
        state.begin_physical_window()
        outcome = self.physical_windows.execute(
            session.scene,
            session.agent_id,
            state,
            steps=safety.steps,
            interventions=safety.interventions,
            native_meta={"physical_calls": turn.tool_calls},
            observation_before=before,
            decision_id=decision_id,
            physical_window_index=state.physical_window_count,
            window_index=state.physical_window_count,
        )
        state.finish_physical_window(
            outcome.observation_after, outcome.measurements
        )
        runtime._maybe_recharge_visual_search_budget(
            state, outcome.measurements
        )
        runtime.journal.write_window(
            self.physical_windows.event(
                decision_id=decision_id,
                window_index=state.physical_window_count,
                before=before,
                after=outcome.observation_after,
                status=(
                    "error"
                    if outcome.failed_call_index is not None
                    else "ok"
                ),
                ordered_steps=[
                    item["tracking"] for item in outcome.executed_steps
                ],
                measurements=outcome.measurements,
            ),
        )
        runtime.checkpoints.publish(
            state,
            runtime._memory_for_state(state),
            "physical_window_completed",
        )
        session.consecutive_rejections = 0
        return TurnOutcome(
            kind="physical",
            checkpoint_reason="physical_window_completed",
            reset_rejections=True,
        )

    @staticmethod
    def rejected_tool_results(
        calls: list[AgentToolCall],
        reason: str,
        *,
        rejection: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        diagnosis = dict(
            rejection or {"code": "invalid_turn_combination", "scope": "turn"}
        )
        source = diagnosis.get("source_tool_call_id")
        return [
            ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="rejected",
                message=reason,
                data={
                    "rejection": {
                        **diagnosis,
                        "role": (
                            (
                                "source"
                                if call.tool_call_id == source
                                else "affected"
                            )
                            if source
                            else "turn_member"
                        ),
                        "executed": False,
                    }
                },
                requested_args=dict(call.args),
            ).to_dict()
            for call in calls
        ]

    def _execute_native_read_calls(
        self,
        session: InstructionSession,
        public_intent: str,
        calls: list[AgentToolCall],
        decision_id: str,
    ) -> list[dict[str, Any]]:
        runtime = self.owner
        jobs = []
        for call in calls:
            args = copy.deepcopy(call.args)
            context = {
                "latest_observation": copy.deepcopy(
                    session.state.latest_observation
                ),
                "run_dir": str(runtime.recorder.run_dir),
                "public_intent": public_intent,
                "tool_call_id": call.tool_call_id,
            }

            def invoke(
                tool=call.tool_name, args=args, context=context
            ):
                return runtime.tools.call(
                    session.scene.scene_instance_id,
                    session.agent_id,
                    tool,
                    args,
                    runtime_context=context,
                )

            jobs.append(
                ReadJob(
                    call.tool_call_id,
                    call.tool_name,
                    runtime.tools.read_execution_hints(call.tool_name, args),
                    invoke,
                )
            )
        executor = ReadOnlyBatchExecutor(
            runtime.read_only_execution_config,
            runtime.tool_telemetry.read_event_sink(decision_id),
        )
        recorded_results: list[dict[str, Any]] = []
        ready: dict[int, Any] = {}
        received_at: dict[int, float] = {}

        def completed(index: int, result: Any) -> None:
            began = time.perf_counter()
            call = calls[index]
            runtime.tool_telemetry.read_result_event(
                "read_result_received", decision_id, call.tool_call_id
            )
            runtime.memory_writer.write_tool_completion(
                instruction_id=runtime.agent_transcript.instruction_id,
                decision_id=decision_id,
                call_index=index,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                args=call.args,
                result=result.to_dict(),
            )
            runtime.tool_telemetry.read_result_event(
                "read_result_persisted",
                decision_id,
                call.tool_call_id,
                elapsed_sec=time.perf_counter() - began,
            )
            ready[index] = result
            received_at[index] = began
            while len(recorded_results) in ready:
                next_index = len(recorded_results)
                next_call = calls[next_index]
                commit_started = time.perf_counter()
                saved = self._commit_native_read_result(
                    session.state, next_call, ready[next_index]
                )
                recorded_results.append(saved)
                del ready[next_index]
                runtime.tool_telemetry.read_result_event(
                    "read_result_committed",
                    decision_id,
                    next_call.tool_call_id,
                    elapsed_sec=time.perf_counter() - commit_started,
                    received_to_commit_sec=(
                        time.perf_counter() - received_at.pop(next_index)
                    ),
                )

        executor.execute(jobs, on_completed=completed)
        return recorded_results

    def _commit_native_read_result(
        self, state: Any, call: AgentToolCall, result: Any
    ) -> dict[str, Any]:
        runtime = self.owner
        result_data = dict(result.data)
        capability_artifact_refs = _pop_string_list(
            result_data, SKILL_ARTIFACT_REFS_KEY
        )
        capability_model_inputs = _pop_dict_list(
            result_data, SKILL_MODEL_INPUTS_KEY
        )
        data = result.to_dict()
        data["data"] = result_data
        write_result = runtime._record_action(
            state, call.tool_name, call.args, data
        )
        transcript_result = ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            status="ok" if result.ok else "error",
            message=result.message,
            data=result_data,
            artifact_refs=list(
                dict.fromkeys(
                    [
                        *capability_artifact_refs,
                        *_result_artifact_refs(result_data),
                    ]
                )
            ),
            result_ref=write_result["action_ref"].get("ref"),
            requested_args=dict(call.args),
            executed_args=dict(call.args),
            model_inputs=capability_model_inputs,
        )
        runtime.agent_transcript.append_tool_result(transcript_result)
        return transcript_result.to_dict()


def _pop_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.pop(key, [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _pop_dict_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.pop(key, [])
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _result_artifact_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"artifact_ref", "description_ref"}:
                if isinstance(item, str) and item:
                    refs.append(item)
            elif key in {"artifact_refs", "description_refs"}:
                if isinstance(item, list):
                    refs.extend(
                        child
                        for child in item
                        if isinstance(child, str) and child
                    )
            else:
                refs.extend(_result_artifact_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_result_artifact_refs(item))
    return refs
