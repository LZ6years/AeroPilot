"""Verify one-use scene evidence, failure retention, and the opt-in RPC saving."""

from types import SimpleNamespace
import pytest

from src.adapters.uav_primitives import PrimitiveResult
from src.adapters.airsim.performance import AirSimPerformance, MeasuredAirSimClient
from src.sim_server.backends.airsim_backend import AirSimBackend
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.control.stabilization import UavStabilizer
from src.agent.planning.schemas import PlannedStep
from tests.test_uav_stabilization import StubTools


def backend_receipt():
    backend = AirSimBackend()
    backend._dispatch = lambda *_: PrimitiveResult(True, execution_state={
        "verified_paused_pose": {"velocity": {"x": 0, "y": 0, "z": 0}}})
    scene = SimpleNamespace(scene_instance_id="generation-1")
    result = backend.execute_action(scene, "uav", "move_body", {})
    return backend, scene, result.execution_state["post_action_state_token"]


@pytest.mark.parametrize("invalidate", ["write", "reset", "stop", "scene", "agent", "used"])
def test_receipt_cannot_survive_control_boundaries(invalidate):
    backend, scene, token = backend_receipt()
    scene_id, agent = scene.scene_instance_id, "uav"
    if invalidate == "write":
        backend._dispatch = lambda *_: (_ for _ in ()).throw(ConnectionError("disconnected"))
        with pytest.raises(ConnectionError):
            backend.execute_action(scene, agent, "set_pose", {})
    elif invalidate == "reset":
        backend.reset(scene)
    elif invalidate == "stop":
        backend.stop(scene)
    elif invalidate == "scene":
        scene_id = "generation-2"
    elif invalidate == "agent":
        agent = "another"
    else:
        assert backend.consume_post_action_state(scene_id, agent, token)
    assert backend.consume_post_action_state(scene_id, agent, token) is None


@pytest.mark.parametrize("enabled,hover,velocity,reused", [
    (True, False, 0, True), (False, False, 0, False),
    (True, True, 0, False), (True, False, float("nan"), False),
    (True, False, 1, False), (True, False, None, False),
])
def test_stabilization_reuse_is_optional_and_requires_valid_velocity(enabled, hover, velocity, reused):
    backend, scene, token = backend_receipt()
    backend._post_action_state[3]["velocity"]["x"] = velocity
    tools = StubTools([{"x": 0, "y": 0, "z": 0}])
    tools.env_client = backend
    config = UavStabilizationConfig(reuse_verified_post_action_state=enabled, hover_after_brake=hover)
    result = UavStabilizer(tools, config=config).wait_after_step(
        scene.scene_instance_id, "uav", "action", PlannedStep.tool_step("uav.move_body", {}, "move"),
        post_action_state=token)
    assert bool(result.get("reused_post_action_state")) == reused
    assert tools.observations == (0 if reused else 1)
    assert bool(tools.calls) == (not reused)


def test_default_and_roundtrip():
    assert not UavStabilizationConfig().reuse_verified_post_action_state
    assert UavStabilizationConfig.from_dict(UavStabilizationConfig().to_dict()) == UavStabilizationConfig()
    with pytest.raises(ValueError):
        UavStabilizationConfig.from_dict({"reuse_verified_post_action_state": "false"})


def test_rpc_failure_has_duration_count_and_parent(tmp_path):
    performance = AirSimPerformance(tmp_path / "timing.jsonl")
    def fail():
        raise TimeoutError("timeout")
    client = MeasuredAirSimClient(SimpleNamespace(simGetImages=fail), performance)
    with pytest.raises(TimeoutError), performance.span("observation"):
        client.simGetImages()
    result = performance.flush("observation")
    assert result["rpc_count"] == 1
    assert result["spans"][1]["parent"] == 0
    assert all(s["elapsed_sec"] >= 0 and s["error_type"] == "TimeoutError" for s in result["spans"])
    assert performance.spans == []


def test_disconnect_invalidates_read_receipt():
    backend, scene, token = backend_receipt()
    def fail():
        raise ConnectionError("disconnected")
    backend.uav = SimpleNamespace(get_pose=fail)
    with pytest.raises(ConnectionError):
        backend.get_observation(scene, "uav", ["pose"])
    assert backend.consume_post_action_state(scene.scene_instance_id, "uav", token) is None


def test_real_adapter_path_saves_five_rpc_without_losing_observation(monkeypatch):
    from src.adapters.airsim.uav import AirSimUavAdapter
    from src.env_client.simulator_client import SimulatorEnvClient
    from src.tools.dispatcher import ToolDispatcher
    from src.tools.registry import default_registry
    from tests.airsim_fakes import FakeAirSim, FakeAirSimClient, FakeImageResponse
    counts = []
    for reuse in (False, True):
        adapter = AirSimUavAdapter(control_strategy="paused_multirotor")
        raw = FakeAirSimClient()
        raw.image_responses = [FakeImageResponse(width=1, height=1, image_data_uint8=bytes([1, 2, 3]))]
        adapter._client = MeasuredAirSimClient(raw, adapter.performance)
        monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)
        backend = AirSimBackend(uav=adapter)
        scene = SimpleNamespace(scene_instance_id="scene")
        service = SimpleNamespace(
            execute_action=lambda sid, agent, name, args: backend.execute_action(scene, agent, name, args),
            get_observation=lambda sid, agent, modalities, capture_sources: backend.get_observation(scene, agent, modalities, capture_sources),
            consume_post_action_state=backend.consume_post_action_state)
        client = SimulatorEnvClient(service)
        tools = ToolDispatcher(client, default_registry())
        result = client.execute_action("scene", "uav", "move_body", {"vx": 1, "vy": 0, "vz": 0, "duration": 1})
        stable = UavStabilizer(tools, config=UavStabilizationConfig(reuse_verified_post_action_state=reuse)).wait_after_step(
            "scene", "uav", "diagnostic", PlannedStep.tool_step("uav.move_body", {}, "test"),
            post_action_state=result.execution_state["post_action_state_token"])
        observation = client.get_observation("scene", "uav", ["rgb", "pose"])
        assert result.ok and stable["status"] == "completed"
        assert "action_diagnostics" not in result.data
        assert result.execution_state["action_diagnostics"]["tracking"]
        assert observation.data["frames"]["frames"]["rgb"]["width"] == 1
        assert len(raw.move_body_calls) == 1 and len(raw.image_requests) == 1
        counts.append(adapter.performance.rpc_count_total)
    assert counts[0] - counts[1] == 5
