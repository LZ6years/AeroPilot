"""Execution of one accepted provider-native tool turn."""

from __future__ import annotations

import copy
import math
import time
from typing import Any

from src.agent.planning.schemas import PlannedStep
from src.agent.planning.tool_contract_validator import (
    NativeTurnValidationError,
    validate_native_turn,
)
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult
from src.env_client import SceneInstance
from src.plugins.contracts.skill import (
    SKILL_ARTIFACT_REFS_KEY,
    SKILL_MODEL_INPUTS_KEY,
)
from src.runtime.control.executor import execute_builtin_control
from src.runtime.control.actions.terminal import environmental_evidence_refs
from src.runtime.control.actions.task_plan import validate_plan_update
from src.runtime.control.read_only_executor import ReadJob, ReadOnlyBatchExecutor
from src.runtime.task.outcomes import TurnOutcome
from src.runtime.task.physical_window import PhysicalWindowExecutor
from src.runtime.task.session import TaskSession
from src.tools.registry import ToolRegistry
from src.tools.diagnostics import Diagnostic, action_diagnostic


class TurnExecutor:
    """Validate and execute runtime-control, read-only, or physical turns."""

    def __init__(
        self,
        tools: Any,
        safety: Any,
        physical_windows: PhysicalWindowExecutor,
        evidence: Any,
        journal: Any,
        recorder: Any,
        read_only_execution_config: Any,
        tool_telemetry: Any,
    ) -> None:
        self.tools = tools
        self.safety = safety
        self.physical_windows = physical_windows
        self.evidence = evidence
        self.journal = journal
        self.recorder = recorder
        self.read_only_execution_config = read_only_execution_config
        self.tool_telemetry = tool_telemetry

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
        session: TaskSession,
        turn: AssistantTurn,
        *,
        allowed_tools: set[str],
        decision_id: str,
        history_boundary: int,
    ) -> TurnOutcome:
        state = session.state
        ids = [call.tool_call_id for call in turn.tool_calls]
        try:
            effect = validate_native_turn(
                turn,
                self.tools.registry,
                allowed_names=allowed_tools,
                latest_observation=state.latest_observation,
                prior_call_ids=session.seen_call_ids,
                executed_read_counts=session.executed_read_counts,
            )
            plan_call = next(
                (
                    call
                    for call in turn.tool_calls
                    if call.tool_name == "runtime.update_plan"
                ),
                None,
            )
            if plan_call is not None:
                try:
                    validate_plan_update(
                        state,
                        plan_call.args,
                    )
                except ValueError as exc:
                    raise NativeTurnValidationError(
                        str(exc),
                        code="plan_update_invalid",
                        call_id=plan_call.tool_call_id,
                        path=getattr(exc, "path", None),
                    ) from exc
        except NativeTurnValidationError as exc:
            session.seen_call_ids.update(ids)
            for record in self.rejected_tool_results(
                turn.tool_calls, str(exc), rejection=exc.diagnostic
            ):
                session.transcript.append_tool_result(
                    ToolResult(
                        record["tool_call_id"],
                        record["tool_name"],
                        "rejected",
                        str(exc),
                        data=record["data"],
                        diagnostic=record["diagnostic"],
                        requested_args=record["requested_args"],
                    )
                )
            session.transcript.append_protocol_feedback(
                str(exc) + ". No call in this turn executed.", exc.diagnostic
            )
            state.record_nonphysical_round()
            session.consecutive_rejections += 1
            return TurnOutcome(kind="protocol_rejected")

        session.seen_call_ids.update(ids)
        if effect == "runtime_control":
            return self._execute_runtime_control(
                session,
                turn,
                decision_id,
                history_boundary,
            )
        if effect == "read_only":
            return self._execute_read_only(session, turn, decision_id)
        if effect == "plan_read_only":
            return self._execute_plan_and_read_only(
                session,
                turn,
                decision_id,
            )
        return self._execute_physical(session, turn, decision_id)

    def _execute_runtime_control(
        self,
        session: TaskSession,
        turn: AssistantTurn,
        decision_id: str,
        history_boundary: int,
    ) -> TurnOutcome:
        state = session.state
        state.record_nonphysical_round()
        call = turn.tool_calls[0]
        # Plans validate against their latest immutable state, not full history.
        # Terminals and plugin controls still receive their pre-turn evidence.
        history = () if call.tool_name == "runtime.update_plan" else session.transcript.snapshot_before(history_boundary)
        if self.tools.registry.get(call.tool_name).kind == "skill_capability":
            # Plugin-owned local writes use the same dispatcher/evidence path
            # as reads; only built-in controls may mutate MissionState.
            action = self.tools.call(
                session.scene.scene_instance_id, session.agent_id, call.tool_name,
                copy.deepcopy(call.args),
                runtime_context=self._capability_context(session, turn.public_intent, call, history),
            )
            self.evidence.write_tool_completion(session, decision_id=decision_id,
                call_index=0, tool_call_id=call.tool_call_id, tool_name=call.tool_name,
                args=call.args, result=action.to_dict())
            result = ToolResult(**self._commit_native_read_result(session, call, action))
        else:
            result = execute_builtin_control(call, state, self.tools.registry, history)
            session.transcript.append_tool_result(result)
        self.journal.write_turn(
            "task_control_recorded",
            {"decision_id": decision_id, "tool_result": result.to_dict()},
        )
        if result.status == "rejected":
            session.transcript.append_protocol_feedback(result.message, result.diagnostic)
            session.consecutive_rejections += 1
        else:
            session.consecutive_rejections = 0
        return TurnOutcome(
            kind="runtime_control",
            reset_rejections=result.status != "rejected",
        )

    def _execute_read_only(
        self,
        session: TaskSession,
        turn: AssistantTurn,
        decision_id: str,
    ) -> TurnOutcome:
        state = session.state
        state.record_nonphysical_round()
        results = self._execute_native_read_calls(
            session, turn.public_intent, turn.tool_calls, decision_id
        )
        state.loaded_knowledge = session.knowledge_service.loaded_index()
        self.journal.write_turn(
            "read_only_turn_recorded",
            {"decision_id": decision_id, "tool_results": results},
        )
        all_ok = all(result["status"] == "ok" for result in results)
        if all_ok:
            session.consecutive_rejections = 0
        return TurnOutcome(
            kind="read_only",
            reset_rejections=all_ok,
        )

    def _execute_plan_and_read_only(
        self,
        session: TaskSession,
        turn: AssistantTurn,
        decision_id: str,
    ) -> TurnOutcome:
        """Commit one prevalidated plan, then execute its independent reads."""

        state = session.state
        state.record_nonphysical_round()
        plan_call = next(
            call
            for call in turn.tool_calls
            if call.tool_name == "runtime.update_plan"
        )
        read_calls = [
            call
            for call in turn.tool_calls
            if call.tool_name != "runtime.update_plan"
        ]
        plan_result = execute_builtin_control(
            plan_call,
            state,
            self.tools.registry,
            (),
        )
        if plan_result.status != "ok":  # defensive: prevalidation must agree
            raise RuntimeError(
                f"prevalidated plan update was not applied: {plan_result.message}"
            )

        read_results = self._execute_native_read_calls(
            session,
            turn.public_intent,
            read_calls,
            decision_id,
            append_to_transcript=False,
        )
        state.loaded_knowledge = session.knowledge_service.loaded_index()

        results_by_id = {
            plan_result.tool_call_id: plan_result,
            **{
                result["tool_call_id"]: _tool_result_from_dict(result)
                for result in read_results
            },
        }
        ordered_results = []
        for call in turn.tool_calls:
            result = results_by_id[call.tool_call_id]
            session.transcript.append_tool_result(result)
            ordered_results.append(result.to_dict())

        self.journal.write_turn(
            "plan_read_only_turn_recorded",
            {"decision_id": decision_id, "tool_results": ordered_results},
        )
        all_ok = all(result["status"] == "ok" for result in ordered_results)
        if all_ok:
            session.consecutive_rejections = 0
        return TurnOutcome(kind="plan_read_only", reset_rejections=all_ok)

    def _execute_physical(
        self,
        session: TaskSession,
        turn: AssistantTurn,
        decision_id: str,
    ) -> TurnOutcome:
        state = session.state
        steps = [
            PlannedStep.tool_step(
                call.tool_name, dict(call.args), turn.public_intent
            )
            for call in turn.tool_calls
        ]
        safety = self.safety.review_steps(
            steps, state, registry=self.tools.registry
        )
        before = copy.deepcopy(state.latest_observation)
        state.begin_physical_window()
        outcome = self.physical_windows.execute(
            session,
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
        self._maybe_recharge_visual_search_budget(session, outcome.measurements)
        self.journal.write_window(
            self.physical_windows.event(
                decision_id=decision_id,
                window_index=state.physical_window_count,
                before=before,
                after=outcome.observation_after,
                status=outcome.status,
                ordered_steps=[item["tracking"] for item in outcome.executed_steps],
                measurements=outcome.measurements,
            )
        )
        session.checkpoints.publish(
            state, session.memory, "physical_window_completed"
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
            rejection or Diagnostic("invalid_turn_combination", "validation", reason=reason,
                                    scope="turn", execution="not_started").to_dict()
        )
        source = diagnosis.get("source_tool_call_id")
        return [
            ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="rejected",
                message=reason,
                diagnostic={
                        **diagnosis,
                        "role": (
                            ("source" if call.tool_call_id == source else "affected")
                            if source
                            else "turn_member"
                        ),
                },
                requested_args=dict(call.args),
            ).to_dict()
            for call in calls
        ]

    def _execute_native_read_calls(
        self,
        session: TaskSession,
        public_intent: str,
        calls: list[AgentToolCall],
        decision_id: str,
        *,
        append_to_transcript: bool = True,
    ) -> list[dict[str, Any]]:
        jobs: list[ReadJob] = []
        history = session.transcript.snapshot() if any(
            self.tools.registry.get(call.tool_name).kind == "skill_capability" for call in calls
        ) else ()
        for call in calls:
            args = copy.deepcopy(call.args)
            context = self._capability_context(session, public_intent, call, history)

            def invoke(tool=call.tool_name, args=args, context=context):
                return self.tools.call(
                    session.scene.scene_instance_id,
                    session.agent_id,
                    tool,
                    args,
                    runtime_context=context,
                    knowledge_service=session.knowledge_service,
                )

            jobs.append(
                ReadJob(
                    call.tool_call_id,
                    call.tool_name,
                    self.tools.read_execution_hints(call.tool_name, args),
                    invoke,
                )
            )
        executor = ReadOnlyBatchExecutor(
            self.read_only_execution_config,
            self.tool_telemetry.read_event_sink(decision_id),
        )
        recorded_results: list[dict[str, Any]] = []
        ready: dict[int, Any] = {}
        received_at: dict[int, float] = {}

        def completed(index: int, result: Any) -> None:
            began = time.perf_counter()
            call = calls[index]
            # Completed attempts consume quota, including handler failures.
            policy = self.tools.registry.get(call.tool_name).runtime
            key = (policy.observation_key(call.args, session.state.latest_observation), call.tool_name)
            session.executed_read_counts[key] = session.executed_read_counts.get(key, 0) + 1
            self.tool_telemetry.read_result_event(
                "read_result_received", decision_id, call.tool_call_id
            )
            self.evidence.write_tool_completion(
                session,
                decision_id=decision_id,
                call_index=index,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                args=call.args,
                result=result.to_dict(),
            )
            self.tool_telemetry.read_result_event(
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
                    session,
                    next_call,
                    ready[next_index],
                    append_to_transcript=append_to_transcript,
                )
                recorded_results.append(saved)
                del ready[next_index]
                self.tool_telemetry.read_result_event(
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

    def _capability_context(self, session, public_intent, call, history):
        """Supply immutable task-scoped inputs, never a Runtime/State handle."""
        return {
            "latest_observation": copy.deepcopy(session.state.latest_observation),
            "run_dir": str(self.recorder.run_dir),
            "instruction_id": session.transcript.instruction_id,
            "public_intent": public_intent,
            "tool_call_id": call.tool_call_id,
            "evidence_refs": sorted(environmental_evidence_refs(history, self.tools.registry)),
        }

    def _commit_native_read_result(
        self,
        session: TaskSession,
        call: AgentToolCall,
        result: Any,
        *,
        append_to_transcript: bool = True,
    ) -> dict[str, Any]:
        result_data, diagnostic = action_diagnostic(result.data, call_id=call.tool_call_id,
            ok=result.ok, phase="read_execution", execution="unknown", reason=result.message)
        capability_artifact_refs = _pop_string_list(
            result_data, SKILL_ARTIFACT_REFS_KEY
        )
        capability_model_inputs = _pop_dict_list(
            result_data, SKILL_MODEL_INPUTS_KEY
        )
        data = result.to_dict()
        data["data"] = result_data
        write_result = self.evidence.record_action(
            session, call.tool_name, call.args, data
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
            diagnostic=diagnostic,
        )
        if append_to_transcript:
            session.transcript.append_tool_result(transcript_result)
        return transcript_result.to_dict()

    def _maybe_recharge_visual_search_budget(
        self, session: TaskSession, measurements: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            planar_m = abs(
                float(measurements.get("planar_displacement_m", 0.0) or 0.0)
            )
            vertical_m = abs(
                float(measurements.get("vertical_delta_m", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            return None
        translation_3d_m = math.hypot(planar_m, vertical_m)
        threshold_m = float(
            self.safety.safety_config.yaw_scan_recharge_distance_m
        )
        if translation_3d_m < threshold_m:
            return None
        recharge = session.state.recharge_yaw_scan()
        payload = {
            "phase_key": None,
            "window_index": session.state.physical_window_count,
            "planar_displacement_m": round(planar_m, 6),
            "vertical_displacement_m": round(vertical_m, 6),
            "translation_3d_m": round(translation_3d_m, 6),
            "recharge_distance_m": threshold_m,
            **recharge,
        }
        self.journal.event("visual_search_budget_recharged", payload)
        return payload


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


def _tool_result_from_dict(value: dict[str, Any]) -> ToolResult:
    """Recreate a deferred immutable result without losing audit fields."""

    return ToolResult(
        tool_call_id=value["tool_call_id"],
        tool_name=value["tool_name"],
        status=value["status"],
        message=value.get("message", ""),
        data=dict(value.get("data") or {}),
        artifact_refs=list(value.get("artifact_refs") or []),
        result_ref=value.get("result_ref"),
        requested_args=dict(value.get("requested_args") or {}),
        executed_args=dict(value.get("executed_args") or {}),
        safety_interventions=list(value.get("safety_interventions") or []),
        model_inputs=list(value.get("model_inputs") or []),
        diagnostic=value.get("diagnostic"),
    )
