"""Tests for runtime tool-call telemetry middleware."""

from __future__ import annotations

import pytest

from src.env_client.schemas import ActionResult
from src.runtime.middleware import ToolTelemetryMiddleware


def test_tool_telemetry_records_success_and_returns_original_result():
    events: list[tuple[str, dict]] = []
    middleware = ToolTelemetryMiddleware(lambda name, payload: events.append((name, payload)))
    result = ActionResult(True, "turn_relative", "turned", {"pose": {"yaw_deg": 15.0}})

    returned = middleware.run_tool_step(
        scene_instance_id="scene-1",
        agent_id="uav_1",
        phase="turn_phase",
        window_index=2,
        step_index=0,
        tool="uav.turn_relative",
        args={"yaw_deg": 15.0},
        pose_before={"yaw_deg": 0.0},
        call_fn=lambda: result,
    )

    assert returned is result
    assert [name for name, _ in events] == ["tool_call_started", "tool_call_completed"]
    assert events[0][1]["pose_before"] == {"yaw_deg": 0.0}
    assert events[1][1]["result"]["ok"] is True
    assert events[1][1]["pose_after"] == {"yaw_deg": 15.0}


def test_tool_telemetry_records_failure_and_reraises():
    events: list[tuple[str, dict]] = []
    middleware = ToolTelemetryMiddleware(lambda name, payload: events.append((name, payload)))

    def explode():
        raise RuntimeError("simulator unavailable")

    with pytest.raises(RuntimeError, match="simulator unavailable"):
        middleware.run_tool_step(
            scene_instance_id="scene-1",
            agent_id="uav_1",
            phase="turn_phase",
            window_index=2,
            step_index=1,
            tool="uav.hover",
            args={},
            pose_before=None,
            call_fn=explode,
        )

    assert [name for name, _ in events] == ["tool_call_started", "tool_call_failed"]
    assert events[1][1]["error_type"] == "RuntimeError"
    assert events[1][1]["error"] == "simulator unavailable"
