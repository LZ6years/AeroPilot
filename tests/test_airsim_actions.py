"""Verify action-window composition independently from the public adapter."""

import pytest

from src.adapters.airsim.actions import PausedMultirotorActionExecutor
from src.adapters.airsim.control import PausedMultirotorControlStrategy
from src.uav.frames import normalize_yaw_deg, position_distance_m
from tests.airsim_fakes import FakeAirSim, FakeAirSimClient


@pytest.mark.parametrize("error,result,kind", [
    (TimeoutError("timeout"), True, "rpc_error"),
    (None, False, "controller_timeout_or_rejected"),
])
def test_executor_reads_future_and_always_freezes(error, result, kind):
    client = FakeAirSimClient()
    client.future_error, client.future_result = error, result
    strategy = PausedMultirotorControlStrategy("localhost", 1, "Drone_1", lambda: FakeAirSim)
    executor = PausedMultirotorActionExecutor(
        "Drone_1", lambda: FakeAirSim, strategy.get_pose,
        strategy._freeze_at_current_pose, strategy._canonical_pose_result,
    )
    outcome = executor.move_body(client, 1, 0, 0, 1)
    assert not outcome.ok
    assert outcome.data["failure_kind"] == kind
    assert client.pause_calls[-1] is True


def test_canonical_distance_preserves_missing_and_nonfinite_measurements():
    assert position_distance_m({"x": 0, "y": 0, "z": 0},
                               {"position": {"x": 3, "y": 0, "z": 4}}) == 5
    assert position_distance_m({"x": 0}, {"x": 1}) is None
    assert position_distance_m({"x": float("nan"), "y": 0, "z": 0},
                               {"x": 1, "y": 0, "z": 0}) is None
    assert normalize_yaw_deg(181) == -179
