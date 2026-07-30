"""Test runtime UAV stabilization policy."""

from __future__ import annotations

from src.agent.planning.schemas import PlannedStep
from src.env_client.schemas import ActionResult, Observation, SceneCapabilities
from src.runtime.control import UavStabilizer
from src.runtime.configs.stabilization_config import UavStabilizationConfig


class Recorder:
    def __init__(self) -> None:
        self.events = []

    def event(self, name, payload):
        self.events.append({"event": name, "payload": payload})


class StubTools:
    def __init__(self, velocities):
        self.velocities = list(velocities)
        self.calls = []
        self.observations = 0

    def call(self, scene_instance_id, agent_id, tool_name, args):
        self.calls.append((tool_name, args))
        return ActionResult(True, tool_name.removeprefix("uav."), "ok", {})

    def observe(self, scene_instance_id, agent_id, modalities):
        self.observations += 1
        velocity = self.velocities.pop(0) if self.velocities else None
        pose = {"position": {"x": 0.0, "y": 0.0, "z": -2.0}}
        if velocity is not None:
            pose["velocity"] = velocity
        return Observation(scene_instance_id, agent_id, modalities, {"pose": pose})


def capabilities(actions, modalities=("pose",)):
    return SceneCapabilities(
        protocol="test",
        control_mode="test",
        supported_actions=tuple(actions),
        observation_modalities=tuple(modalities),
        requires_takeoff=False,
        supports_soft_reset=False,
    )


def test_stabilizer_skips_non_motion_tool():
    recorder = Recorder()
    tools = StubTools([])
    stabilizer = UavStabilizer(tools, recorder, UavStabilizationConfig(poll_interval_sec=0.0))

    result = stabilizer.wait_after_step("scene", "uav_1", "phase", PlannedStep.tool_step("uav.hover", {}, "hover"))

    assert result["status"] == "skipped"
    assert result["reason"] == "non_motion_tool"
    assert tools.observations == 0


def test_stabilizer_completes_when_velocity_is_below_threshold():
    recorder = Recorder()
    tools = StubTools([{"x": 0.3, "y": 0.0, "z": 0.0}, {"x": 0.05, "y": 0.0, "z": 0.0}])
    stabilizer = UavStabilizer(tools, recorder, UavStabilizationConfig(poll_interval_sec=0.0))

    result = stabilizer.wait_after_step("scene", "uav_1", "phase", PlannedStep.tool_step("uav.move_body", {}, "move"))

    assert result["status"] == "completed"
    assert result["poll_count"] == 2
    assert tools.calls[0] == ("uav.move_body", {"vx": 0.0, "vy": 0.0, "vz": 0.0, "duration": 2.0})
    assert result["hover_after_brake"] is False
    assert result["hover_result"] is None
    assert "uav_stabilization_brake_requested" in [event["event"] for event in recorder.events]
    assert "uav_stabilization_completed" in [event["event"] for event in recorder.events]


def test_stabilizer_can_hover_after_brake_when_enabled():
    tools = StubTools([{"x": 0.05, "y": 0.0, "z": 0.0}])
    stabilizer = UavStabilizer(
        tools,
        Recorder(),
        UavStabilizationConfig(poll_interval_sec=0.0, hover_after_brake=True),
    )

    result = stabilizer.wait_after_step("scene", "uav_1", "phase", PlannedStep.tool_step("uav.move_body", {}, "move"))

    assert result["status"] == "completed"
    assert [call[0] for call in tools.calls] == ["uav.move_body", "uav.hover"]
    assert result["hover_result"]["ok"] is True


def test_stabilizer_skips_brake_when_duration_is_zero():
    tools = StubTools([{"x": 0.05, "y": 0.0, "z": 0.0}])
    stabilizer = UavStabilizer(
        tools,
        Recorder(),
        UavStabilizationConfig(poll_interval_sec=0.0, zero_velocity_brake_duration_sec=0.0),
    )

    result = stabilizer.wait_after_step("scene", "uav_1", "phase", PlannedStep.tool_step("uav.move_body", {}, "move"))

    assert result["status"] == "completed"
    assert tools.calls == []
    assert result["brake_result"] is None


def test_stabilizer_times_out_without_raising():
    tools = StubTools([{"x": 1.0, "y": 0.0, "z": 0.0}] * 10000)
    stabilizer = UavStabilizer(
        tools,
        Recorder(),
        UavStabilizationConfig(timeout_sec=0.01, poll_interval_sec=0.0, hover_after_brake=False),
    )

    result = stabilizer.wait_after_step("scene", "uav_1", "phase", PlannedStep.tool_step("uav.turn_relative", {}, "turn"))

    assert result["status"] == "timeout"
    assert result["last_velocity_norm_mps"] == 1.0


def test_stabilizer_skips_when_velocity_is_missing():
    tools = StubTools([None])
    stabilizer = UavStabilizer(tools, Recorder(), UavStabilizationConfig(poll_interval_sec=0.0))

    result = stabilizer.wait_after_step("scene", "uav_1", "phase", PlannedStep.tool_step("uav.move_body", {}, "move"))

    assert result["status"] == "skipped"
    assert result["reason"] == "velocity_missing"


def test_stabilizer_skips_unsupported_brake_and_hover():
    tools = StubTools([{"x": 0.0, "y": 0.0, "z": 0.0}])
    stabilizer = UavStabilizer(
        tools,
        Recorder(),
        UavStabilizationConfig(poll_interval_sec=0.0, hover_after_brake=True),
    )

    result = stabilizer.wait_after_step(
        "scene",
        "uav_1",
        "phase",
        PlannedStep.tool_step("uav.turn_relative", {}, "turn"),
        capabilities=capabilities(("turn_relative",)),
    )

    assert result["status"] == "completed"
    assert result["brake_result"]["status"] == "skipped"
    assert result["brake_result"]["reason"] == "move_body_unsupported"
    assert result["hover_result"]["status"] == "skipped"
    assert result["hover_result"]["reason"] == "hover_unsupported"
    assert tools.calls == []


def test_stabilizer_skips_polling_without_pose_observation():
    tools = StubTools([{"x": 0.0, "y": 0.0, "z": 0.0}])
    stabilizer = UavStabilizer(
        tools,
        Recorder(),
        UavStabilizationConfig(poll_interval_sec=0.0),
    )

    result = stabilizer.wait_after_step(
        "scene",
        "uav_1",
        "phase",
        PlannedStep.tool_step("uav.move_body", {}, "move"),
        capabilities=capabilities(("move_body",), modalities=("rgb",)),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "pose_observation_unsupported"
    assert tools.observations == 0
    assert tools.calls == []
