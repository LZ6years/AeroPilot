"""Tests for runtime tool-call telemetry middleware."""

from __future__ import annotations

import pytest

from src.env_client.schemas import ActionResult
from src.runtime.middleware import ToolTelemetryMiddleware


class FakeDispatcher:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = []

    def call(self, scene_instance_id, agent_id, tool_name, args, runtime_context=None):
        self.calls.append(
            {
                "scene_instance_id": scene_instance_id,
                "agent_id": agent_id,
                "tool_name": tool_name,
                "args": dict(args),
                "runtime_context": dict(runtime_context or {}),
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


def test_tool_telemetry_records_success_and_returns_original_result():
    events: list[tuple[str, dict]] = []
    result = ActionResult(True, "turn_relative", "turned", {"pose": {"yaw_deg": 15.0}})
    dispatcher = FakeDispatcher(result=result)
    middleware = ToolTelemetryMiddleware(dispatcher, lambda name, payload: events.append((name, payload)))

    returned = middleware.call_step(
        scene_instance_id="scene-1",
        agent_id="uav_1",
        phase="turn_phase",
        window_index=2,
        step_index=0,
        tool="uav.turn_relative",
        args={"yaw_deg": 15.0},
        pose_before={"yaw_deg": 0.0},
        runtime_context={"latest_observation": {"pose": {"yaw_deg": 0.0}}},
    )

    assert returned is result
    assert dispatcher.calls == [
        {
            "scene_instance_id": "scene-1",
            "agent_id": "uav_1",
            "tool_name": "uav.turn_relative",
            "args": {"yaw_deg": 15.0},
            "runtime_context": {"latest_observation": {"pose": {"yaw_deg": 0.0}}},
        }
    ]
    assert [name for name, _ in events] == ["tool_call_started", "tool_call_completed"]
    assert events[0][1]["pose_before"] == {"yaw_deg": 0.0}
    assert events[1][1]["result"]["ok"] is True
    assert events[1][1]["pose_after"] == {"yaw_deg": 15.0}


def test_tool_telemetry_records_failure_and_reraises():
    events: list[tuple[str, dict]] = []
    dispatcher = FakeDispatcher(error=RuntimeError("simulator unavailable"))
    middleware = ToolTelemetryMiddleware(dispatcher, lambda name, payload: events.append((name, payload)))

    with pytest.raises(RuntimeError, match="simulator unavailable"):
        middleware.call_step(
            scene_instance_id="scene-1",
            agent_id="uav_1",
            phase="turn_phase",
            window_index=2,
            step_index=1,
            tool="uav.hover",
            args={},
            pose_before=None,
            runtime_context={"phase": "turn_phase"},
        )

    assert dispatcher.calls[0]["tool_name"] == "uav.hover"
    assert dispatcher.calls[0]["runtime_context"] == {"phase": "turn_phase"}
    assert [name for name, _ in events] == ["tool_call_started", "tool_call_failed"]
    assert events[1][1]["error_type"] == "RuntimeError"
    assert events[1][1]["error"] == "simulator unavailable"
