"""Physical-window execution, measurements, and event-envelope ownership."""

from __future__ import annotations

from typing import Any, Callable

from src.agent.planning.schemas import PlannedStep
from src.agent.transcript import ToolResult
from src.env_client import ActionResult, SceneInstance
from src.env_client.errors import SceneRecoveryRequired
from src.runtime.task.outcomes import PhysicalWindowOutcome
from src.uav import pose_measurements


PHYSICAL_TRACE_SCHEMA_VERSION = "physical_window_v2"


class PhysicalWindowExecutor:
    """Execute one Safety-reviewed physical window and capture its evidence."""

    def __init__(
        self,
        owner: Any,
        pose_from_observation: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.owner = owner
        self.pose_from_observation = pose_from_observation

    def measurements(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, Any]:
        return pose_measurements(
            self.pose_from_observation(before),
            self.pose_from_observation(after),
        )

    @staticmethod
    def event(
        *,
        decision_id: str,
        window_index: int,
        before: dict[str, Any],
        after: dict[str, Any],
        ordered_steps: list[dict[str, Any]],
        status: str,
        measurements: dict[str, Any] | None = None,
        error: BaseException | None = None,
        stabilization: list[dict[str, Any]] | None = None,
        observation_error: BaseException | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": PHYSICAL_TRACE_SCHEMA_VERSION,
            "decision_id": decision_id,
            "phase_key": None,
            "window_index": window_index,
            "observation_before_id": before.get("observation_id"),
            "observation_after_id": after.get("observation_id"),
            "status": status,
            "ordered_steps": ordered_steps,
        }
        if measurements is not None:
            payload["measurements"] = measurements
        if error is not None:
            payload.update(error_type=type(error).__name__, error=str(error))
        if stabilization is not None:
            payload["stabilization"] = stabilization
        if observation_error is not None:
            payload["observation_error"] = {
                "error_type": type(observation_error).__name__,
                "error": str(observation_error),
            }
        elif error is not None:
            payload["observation_error"] = None
        return payload

    def execute(
        self,
        scene: SceneInstance,
        agent_id: str,
        state: Any,
        *,
        steps: list[PlannedStep],
        interventions: list[dict[str, Any]],
        native_meta: dict[str, Any],
        observation_before: dict[str, Any],
        decision_id: str,
        physical_window_index: int,
        window_index: int,
    ) -> PhysicalWindowOutcome:
        """Execute a prevalidated batch and make one automatic observation."""

        runtime = self.owner
        if not 1 <= len(steps) <= 3:
            raise ValueError("physical batch requires 1–3 steps")
        submitted_calls = native_meta.get("physical_calls", [])
        if submitted_calls and len(submitted_calls) != len(steps):
            raise ValueError("physical calls and Safety-reviewed steps must match")
        for step in steps:
            if step.kind != "tool" or not step.tool:
                raise ValueError("physical batches accept only tool steps")
            if runtime.tools.registry.get(step.tool).runtime.effect != "physical":
                raise ValueError("non-physical step in a physical batch")
            runtime._validate_step_capability(scene, step.tool)

        runtime._mark_before_last_action_window(state, observation_before)
        executed_steps: list[dict[str, Any]] = []
        native_physical_calls = list(native_meta.get("physical_calls", []))
        failed_native_index: int | None = None
        batch_error: Exception | None = None
        for index, step in enumerate(steps):
            payload = {"step_index": index, "step": step.to_dict()}
            if step.kind != "tool" or not step.tool:
                raise ValueError(
                    "runtime action plan requires tool steps; observation is automatic"
                )
            native_call = (
                native_physical_calls[index]
                if index < len(native_physical_calls)
                else None
            )
            requested_tool = (
                native_call.tool_name if native_call is not None else step.tool
            )
            relevant_interventions = [
                dict(item)
                for item in interventions
                if item.get("tool") == requested_tool
                and ("step_index" not in item or item["step_index"] == index)
            ]
            try:
                result = runtime.tool_telemetry.call_step(
                    scene_instance_id=scene.scene_instance_id,
                    agent_id=agent_id,
                    phase=None,
                    window_index=window_index,
                    step_index=index,
                    tool=step.tool,
                    args=step.args,
                    pose_before=runtime._pose_from_observation(
                        state.latest_observation
                    ),
                    runtime_context={
                        "latest_observation": dict(state.latest_observation),
                        "observation_before": dict(observation_before),
                        "run_dir": str(runtime.recorder.run_dir),
                    },
                )
            except Exception as exc:
                if not native_physical_calls:
                    raise
                result = ActionResult(
                    ok=False,
                    action=step.tool,
                    message=f"{type(exc).__name__}: {exc}",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )
            data = result.to_dict()
            write_result = runtime._record_action(state, step.tool, step.args, data)
            action_ref = write_result.get("action_ref") or {}
            pose_ref = write_result.get("pose_ref") or {}
            step_tracking = {
                "step_index": index,
                "tool_call_id": (
                    native_call.tool_call_id if native_call is not None else None
                ),
                "requested_tool": requested_tool,
                "tool": step.tool,
                "effect": runtime.tools.registry.get(step.tool).runtime.effect,
                "status": "ok" if result.ok else "error",
                "action_ref": action_ref.get("ref"),
                "result_pose_ref": pose_ref.get("ref"),
                "requested_args": dict(
                    native_call.args if native_call is not None else step.args
                ),
                "executed_args": dict(step.args),
                "safety_interventions": relevant_interventions,
            }
            if native_call is not None:
                transcript_result = ToolResult(
                    tool_call_id=native_call.tool_call_id,
                    tool_name=requested_tool,
                    status="ok" if result.ok else "error",
                    message=result.message,
                    data=dict(result.data),
                    result_ref=write_result["action_ref"].get("ref"),
                    requested_args=dict(native_call.args),
                    executed_args=dict(step.args),
                    safety_interventions=relevant_interventions,
                )
                runtime.agent_transcript.append_tool_result(transcript_result)
                self._append_native_round_tool_result(native_meta, transcript_result)
            if not result.ok:
                if result.execution_state.get("requires_scene_reset"):
                    batch_error = SceneRecoveryRequired(result.message)
                executed_steps.append(
                    {
                        **payload,
                        "tracking": step_tracking,
                        "result": data,
                        "stabilization": {
                            "status": "skipped",
                            "reason": "tool execution failed",
                        },
                    }
                )
                if not native_physical_calls:
                    raise batch_error or RuntimeError(
                        f"Action failed: {step.tool}: {result.message}"
                    )
                failed_native_index = index
                break
            try:
                stabilization = runtime.stabilizer.wait_after_step(
                    scene.scene_instance_id,
                    agent_id,
                    None,
                    step,
                    capabilities=scene.capabilities,
                    post_action_state=result.execution_state.get(
                        "post_action_state_token"
                    ),
                )
            except Exception as exc:
                batch_error = exc
                failed_native_index = index
                stabilization = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            executed_steps.append(
                {
                    **payload,
                    "tracking": step_tracking,
                    "result": data,
                    "stabilization": stabilization,
                }
            )
            runtime._event(
                "tool_step_completed",
                {
                    "decision_id": decision_id,
                    "physical_window_index": physical_window_index,
                    "step_index": index,
                    "tool": step.tool,
                    "ok": data.get("ok"),
                    "stabilization": runtime._stabilization_summary(stabilization),
                },
            )
            runtime._write_state(state)
            if batch_error is not None:
                break

        if failed_native_index is not None:
            for skipped_index, call in enumerate(
                native_physical_calls[failed_native_index + 1 :],
                start=failed_native_index + 1,
            ):
                transcript_result = ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="skipped",
                    message=(
                        "skipped because an earlier physical call or "
                        "stabilization failed"
                    ),
                    requested_args=dict(call.args),
                )
                runtime.agent_transcript.append_tool_result(transcript_result)
                self._append_native_round_tool_result(native_meta, transcript_result)
                skipped_step = (
                    steps[skipped_index] if skipped_index < len(steps) else None
                )
                if skipped_step is not None:
                    executed_steps.append(
                        {
                            "step_index": skipped_index,
                            "step": skipped_step.to_dict(),
                            "tracking": {
                                "step_index": skipped_index,
                                "tool_call_id": call.tool_call_id,
                                "requested_tool": call.tool_name,
                                "tool": skipped_step.tool,
                                "effect": runtime.tools.registry.get(
                                    skipped_step.tool
                                ).runtime.effect,
                                "status": "skipped",
                                "action_ref": None,
                                "result_pose_ref": None,
                                "requested_args": dict(call.args),
                                "executed_args": None,
                                "safety_interventions": [],
                            },
                            "result": {
                                "ok": False,
                                "status": "skipped",
                                "message": (
                                    "skipped because an earlier physical call or "
                                    "stabilization failed"
                                ),
                            },
                            "stabilization": {
                                "status": "skipped",
                                "reason": "physical call was not executed",
                            },
                        }
                    )
            if not isinstance(batch_error, SceneRecoveryRequired):
                try:
                    runtime._try_hover(scene, agent_id, state)
                except SceneRecoveryRequired as exc:
                    batch_error = exc

        observation_after: dict[str, Any] = {}
        observation_error: Exception | None = None
        try:
            if not isinstance(batch_error, SceneRecoveryRequired):
                observation_after = runtime._apply_automatic_observation_policy(
                    scene, agent_id, state
                )
        except Exception as exc:
            observation_error = exc
            if batch_error is None:
                batch_error = exc
                try:
                    runtime._try_hover(scene, agent_id, state)
                except SceneRecoveryRequired as recovery:
                    batch_error = recovery
        if batch_error is not None:
            runtime.journal.write_window(
                self.event(
                    decision_id=decision_id,
                    window_index=physical_window_index,
                    before=observation_before,
                    after=observation_after,
                    status="error",
                    ordered_steps=[item["tracking"] for item in executed_steps],
                    error=batch_error,
                    stabilization=[
                        item["stabilization"] for item in executed_steps
                    ],
                    observation_error=observation_error,
                ),
            )
            raise batch_error

        window_measurements = self.measurements(
            observation_before, observation_after
        )
        return PhysicalWindowOutcome(
            executed_steps=tuple(executed_steps),
            failed_call_index=failed_native_index,
            observation_after=observation_after,
            measurements=window_measurements,
        )

    @staticmethod
    def _append_native_round_tool_result(
        native_meta: dict[str, Any], result: ToolResult
    ) -> None:
        rounds = native_meta.get("model_rounds")
        if not isinstance(rounds, list) or not rounds:
            return
        final_round = rounds[-1]
        if not isinstance(final_round, dict):
            return
        outcome = final_round.get("runtime_outcome")
        if not isinstance(outcome, dict):
            outcome = {"status": "physical_batch_accepted", "feedback": None}
            final_round["runtime_outcome"] = outcome
        results = outcome.setdefault("tool_results", [])
        if isinstance(results, list):
            results.append(result.to_dict())
