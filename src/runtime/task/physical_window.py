"""Physical-window execution, measurements, and event-envelope ownership."""

from __future__ import annotations

from typing import Any

from src.agent.planning.schemas import PlannedStep
from src.agent.transcript import ToolResult
from src.env_client import ActionResult, SceneInstance
from src.env_client.errors import SceneRecoveryRequired
from src.runtime.task.outcomes import PhysicalWindowOutcome
from src.runtime.task.session import TaskSession
from src.uav import pose_measurements


PHYSICAL_TRACE_SCHEMA_VERSION = "physical_window_v2"


class PhysicalWindowExecutor:
    """Execute one Safety-reviewed physical transaction and capture evidence."""

    def __init__(
        self,
        tools: Any,
        tool_telemetry: Any,
        stabilizer: Any,
        evidence: Any,
        journal: Any,
        recorder: Any,
    ) -> None:
        self.tools = tools
        self.tool_telemetry = tool_telemetry
        self.stabilizer = stabilizer
        self.evidence = evidence
        self.journal = journal
        self.recorder = recorder

    @staticmethod
    def measurements(
        before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, Any]:
        return pose_measurements(
            _pose_from_observation(before),
            _pose_from_observation(after),
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
        session: TaskSession,
        *,
        steps: list[PlannedStep],
        interventions: list[dict[str, Any]],
        native_meta: dict[str, Any],
        observation_before: dict[str, Any],
        decision_id: str,
        physical_window_index: int,
        window_index: int,
    ) -> PhysicalWindowOutcome:
        """Execute a 1–3 call physical batch and take one after-observation."""

        scene = session.scene
        state = session.state
        if not 1 <= len(steps) <= 3:
            raise ValueError("physical batch requires 1–3 steps")
        submitted_calls = native_meta.get("physical_calls", [])
        if submitted_calls and len(submitted_calls) != len(steps):
            raise ValueError("physical calls and Safety-reviewed steps must match")
        for step in steps:
            if step.kind != "tool" or not step.tool:
                raise ValueError("physical batches accept only tool steps")
            if self.tools.registry.get(step.tool).runtime.effect != "physical":
                raise ValueError("non-physical step in a physical batch")
            self._validate_step_capability(scene, step.tool)

        self.evidence.mark_before_last_action_window(session, observation_before)
        executed_steps: list[dict[str, Any]] = []
        native_calls = list(native_meta.get("physical_calls", []))
        interrupted_index: int | None = None
        interruption_status: str | None = None
        batch_error: Exception | None = None
        for index, step in enumerate(steps):
            payload = {"step_index": index, "step": step.to_dict()}
            native_call = native_calls[index] if index < len(native_calls) else None
            requested_tool = native_call.tool_name if native_call else step.tool
            relevant_interventions = [
                dict(item)
                for item in interventions
                if item.get("tool") == requested_tool
                and ("step_index" not in item or item["step_index"] == index)
            ]
            try:
                result = self.tool_telemetry.call_step(
                    scene_instance_id=scene.scene_instance_id,
                    agent_id=session.agent_id,
                    phase=None,
                    window_index=window_index,
                    step_index=index,
                    tool=step.tool,
                    args=step.args,
                    pose_before=_pose_from_observation(state.latest_observation),
                    runtime_context={
                        "latest_observation": dict(state.latest_observation),
                        "observation_before": dict(observation_before),
                        "run_dir": str(self.recorder.run_dir),
                    },
                )
            except Exception as exc:
                if not native_calls:
                    raise
                result = ActionResult(
                    ok=False,
                    action=step.tool,
                    message=f"{type(exc).__name__}: {exc}",
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )
            result_data = result.to_dict()
            result_status = result.outcome
            write_result = self.evidence.record_action(
                session, step.tool, step.args, result_data
            )
            action_ref = write_result.get("action_ref") or {}
            pose_ref = write_result.get("pose_ref") or {}
            tracking = {
                "step_index": index,
                "tool_call_id": native_call.tool_call_id if native_call else None,
                "requested_tool": requested_tool,
                "tool": step.tool,
                "effect": self.tools.registry.get(step.tool).runtime.effect,
                "status": result_status,
                "action_ref": action_ref.get("ref"),
                "result_pose_ref": pose_ref.get("ref"),
                "requested_args": dict(native_call.args if native_call else step.args),
                "executed_args": dict(step.args),
                "safety_interventions": relevant_interventions,
            }
            if native_call is not None:
                transcript_result = ToolResult(
                    tool_call_id=native_call.tool_call_id,
                    tool_name=requested_tool,
                    status=result_status,
                    message=result.message,
                    data=dict(result.data),
                    result_ref=action_ref.get("ref"),
                    requested_args=dict(native_call.args),
                    executed_args=dict(step.args),
                    safety_interventions=relevant_interventions,
                )
                session.transcript.append_tool_result(transcript_result)
                self._append_native_round_tool_result(native_meta, transcript_result)
            if result_status == "error":
                if result.execution_state.get("requires_scene_reset"):
                    batch_error = SceneRecoveryRequired(result.message)
                executed_steps.append(
                    {
                        **payload,
                        "tracking": tracking,
                        "result": result_data,
                        "stabilization": {
                            "status": "skipped",
                            "reason": "tool execution failed",
                        },
                    }
                )
                if not native_calls:
                    raise batch_error or RuntimeError(
                        f"Action failed: {step.tool}: {result.message}"
                    )
                interrupted_index = index
                interruption_status = "error"
                break
            try:
                stabilization = self.stabilizer.wait_after_step(
                    scene.scene_instance_id,
                    session.agent_id,
                    None,
                    step,
                    capabilities=scene.capabilities,
                    post_action_state=result.execution_state.get(
                        "post_action_state_token"
                    ),
                )
            except Exception as exc:
                batch_error = exc
                interrupted_index = index
                interruption_status = "error"
                stabilization = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            executed_steps.append(
                {
                    **payload,
                    "tracking": tracking,
                    "result": result_data,
                    "stabilization": stabilization,
                }
            )
            self.journal.event(
                "tool_step_completed",
                {
                    "decision_id": decision_id,
                    "physical_window_index": physical_window_index,
                    "step_index": index,
                    "tool": step.tool,
                    "ok": result_data.get("ok"),
                    "stabilization": _stabilization_summary(stabilization),
                },
            )
            self.evidence.write_state(session)
            if batch_error is not None:
                break
            if result_status == "collision":
                interrupted_index = index
                interruption_status = "collision"
                break

        if interrupted_index is not None:
            skipped_message = (
                "skipped because an earlier physical call collided"
                if interruption_status == "collision"
                else "skipped because an earlier physical call or stabilization failed"
            )
            for skipped_index, call in enumerate(
                native_calls[interrupted_index + 1 :],
                start=interrupted_index + 1,
            ):
                transcript_result = ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    status="skipped",
                    message=skipped_message,
                    requested_args=dict(call.args),
                )
                session.transcript.append_tool_result(transcript_result)
                self._append_native_round_tool_result(native_meta, transcript_result)
                skipped_step = steps[skipped_index] if skipped_index < len(steps) else None
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
                                "effect": self.tools.registry.get(
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
                                "message": skipped_message,
                            },
                            "stabilization": {
                                "status": "skipped",
                                "reason": "physical call was not executed",
                            },
                        }
                    )
            if (
                interruption_status != "collision"
                and not isinstance(batch_error, SceneRecoveryRequired)
            ):
                try:
                    self._try_hover(session)
                except SceneRecoveryRequired as exc:
                    batch_error = exc

        observation_after: dict[str, Any] = {}
        observation_error: Exception | None = None
        try:
            if not isinstance(batch_error, SceneRecoveryRequired):
                observation_after = self._automatic_observation(session)
        except Exception as exc:
            observation_error = exc
            if batch_error is None:
                batch_error = exc
                try:
                    self._try_hover(session)
                except SceneRecoveryRequired as recovery:
                    batch_error = recovery
        if batch_error is not None:
            self.journal.write_window(
                self.event(
                    decision_id=decision_id,
                    window_index=physical_window_index,
                    before=observation_before,
                    after=observation_after,
                    status="error",
                    ordered_steps=[item["tracking"] for item in executed_steps],
                    error=batch_error,
                    stabilization=[item["stabilization"] for item in executed_steps],
                    observation_error=observation_error,
                )
            )
            raise batch_error

        return PhysicalWindowOutcome(
            executed_steps=tuple(executed_steps),
            status=interruption_status or "ok",
            interrupted_call_index=interrupted_index,
            observation_after=observation_after,
            measurements=self.measurements(observation_before, observation_after),
        )

    def _automatic_observation(self, session: TaskSession) -> dict[str, Any]:
        modalities = list(session.state.observation_modalities)
        source = "automatic_physical_window"
        self.journal.debug_event(
            "automatic_observation_policy_applied",
            {
                "mode": "automatic",
                "modalities": modalities,
                "sample_timing": "after_each_physical_window",
            },
        )
        observation = self.evidence.record_observation(session, modalities, source)
        self.journal.debug_event("automatic_observation_updated", observation)
        self.journal.event(
            "observation_updated",
            self.evidence.observation_summary(observation, source),
        )
        self.evidence.write_state(session)
        return observation

    def _try_hover(self, session: TaskSession) -> None:
        scene = session.scene
        if scene.capabilities and not scene.capabilities.supports("hover"):
            self.journal.debug_event(
                "safety_hover_skipped", {"reason": "backend_capability"}
            )
            return
        try:
            result = self.tools.call(
                scene.scene_instance_id, session.agent_id, "uav.hover", {}
            )
            self.evidence.record_action(session, "uav.hover", {}, result.to_dict())
            self.journal.debug_event("safety_hover_completed", result.to_dict())
            if result.execution_state.get("requires_scene_reset"):
                raise SceneRecoveryRequired(result.message)
        except SceneRecoveryRequired:
            raise
        except Exception as exc:
            self.journal.debug_event(
                "safety_hover_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )

    def _validate_step_capability(
        self, scene: SceneInstance, tool_name: str
    ) -> None:
        spec = self.tools.registry.get(tool_name)
        action = spec.metadata.get("backend_action")
        if not isinstance(action, str) or not action or scene.capabilities is None:
            return
        if scene.capabilities.supports(action):
            return
        payload = {
            "tool": tool_name,
            "action": action,
            "protocol": scene.capabilities.protocol,
            "control_mode": scene.capabilities.control_mode,
            "supported_actions": list(scene.capabilities.supported_actions),
        }
        self.journal.debug_event("runtime_action_capability_rejected", payload)
        raise RuntimeError(
            f"{scene.capabilities.protocol}/{scene.capabilities.control_mode} "
            f"does not support {action}"
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


def _pose_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    pose = observation.get("pose") if isinstance(observation, dict) else None
    return dict(pose) if isinstance(pose, dict) else {}


def _stabilization_summary(stabilization: dict | None) -> dict[str, Any]:
    data = stabilization if isinstance(stabilization, dict) else {}
    return {
        "status": data.get("status"),
        "elapsed_sec": data.get("elapsed_sec"),
        "last_velocity_norm_mps": data.get("last_velocity_norm_mps"),
        "threshold": data.get("threshold"),
        "reason": data.get("reason"),
    }
