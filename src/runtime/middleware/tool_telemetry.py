"""Record debug telemetry around runtime tool execution.

The middleware is intentionally observational: it emits started/completed/failed
debug events and returns or raises exactly what the wrapped tool call does. It
does not update MissionState, write memory records, enforce safety, or retry
actions; TaskRuntime owns those behaviors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.env_client.schemas import ActionResult
from src.tools.dispatcher import ToolDispatcher


class ToolTelemetryMiddleware:
    """Wrap ToolDispatcher calls with runtime debug events."""

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        debug_event: Callable[[str, dict[str, Any]], None],
    ) -> None:
        self.dispatcher = dispatcher
        self.debug_event = debug_event

    def call_step(
        self,
        *,
        scene_instance_id: str,
        agent_id: str,
        phase: str,
        window_index: int,
        step_index: int,
        tool: str,
        args: dict[str, Any],
        pose_before: dict[str, Any] | None,
        runtime_context: dict[str, Any] | None = None,
    ) -> ActionResult:
        """Dispatch one tool step and emit telemetry without changing behavior."""

        base_payload = {
            "scene_instance_id": scene_instance_id,
            "agent_id": agent_id,
            "phase": phase,
            "window_index": window_index,
            "step_index": step_index,
            "tool": tool,
            "args": dict(args),
        }
        if pose_before:
            base_payload["pose_before"] = dict(pose_before)
        self.debug_event("tool_call_started", base_payload)
        try:
            result = self.dispatcher.call(
                scene_instance_id,
                agent_id,
                tool,
                args,
                runtime_context=runtime_context,
            )
        except Exception as exc:
            self.debug_event(
                "tool_call_failed",
                {
                    **base_payload,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

        result_data = _result_to_dict(result)
        completed_payload = {
            **base_payload,
            "result": result_data,
        }
        pose_after = _pose_from_result(result_data)
        if pose_after:
            completed_payload["pose_after"] = pose_after
        self.debug_event("tool_call_completed", completed_payload)
        return result


def _result_to_dict(result: Any) -> dict[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        return dict(data) if isinstance(data, dict) else {"value": data}
    if isinstance(result, dict):
        return dict(result)
    return {"value": repr(result)}


def _pose_from_result(result_data: dict[str, Any]) -> dict[str, Any]:
    data = result_data.get("data")
    if isinstance(data, dict) and isinstance(data.get("pose"), dict):
        return dict(data["pose"])
    if isinstance(result_data.get("pose"), dict):
        return dict(result_data["pose"])
    return {}
