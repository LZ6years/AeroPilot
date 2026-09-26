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


def strategy():
    return PausedMultirotorControlStrategy("localhost", 1, "Drone_1", lambda: FakeAirSim)


def test_timeout_cancels_task_and_records_all_stage_times():
    client = FakeAirSimClient()
    client.future_error = TimeoutError("server task may still be running")
    result = strategy().move_body(client, 1, 0, 0, 1)
    diagnostics = result.data['action_diagnostics']
    assert client.cancel_calls == ['Drone_1']
    assert diagnostics['cancel'] == {'attempted': True, 'error': None}
    assert diagnostics['recovery_required'] is False
    assert result.execution_state['requires_scene_reset'] is False
    timings = diagnostics['timings_sec']
    assert all(t >= 0 for t in timings.values())
    assert timings['total'] >= sum(timings[k] for k in (
        'before_snapshot', 'action_wait', 'cancel', 'freeze', 'after_snapshot'))
    assert timings['action_wait'] >= timings['future_wait']


def test_failed_cancel_blocks_later_motion_but_retains_original_error():
    client = FakeAirSimClient()
    client.future_error = TimeoutError('original')
    client.cancel_error = RuntimeError('cancel failed')
    control = strategy()
    result = control.move_body(client, 1, 0, 0, 1)
    assert result.data['failure_kind'] == 'rpc_error'
    diagnostics = result.data['action_diagnostics']
    assert diagnostics['cancel']['error']['error'] == 'cancel failed'
    assert diagnostics['rpc']['error']['error'] == 'original'
    assert result.execution_state['requires_scene_reset'] is True
    client.future_error = None
    with pytest.raises(RuntimeError, match='recreate the scene'):
        control.move_body(client, 1, 0, 0, 1)
    assert len(client.move_body_calls) == 1
    assert client.pause_calls[-1] is True


@pytest.mark.parametrize('scale,kind', [(0.01, 'control_insufficient_progress'),
                                      (13, 'motion_discontinuity'), (0.5, None)])
def test_gross_translation_verification(scale, kind):
    client = FakeAirSimClient()
    client.action_effect_scale = scale
    control = strategy()
    result = control.move_body(client, 0, 0, 2, 3)
    assert result.data['failure_kind'] == kind
    assert result.ok == (kind is None)
    if kind == 'motion_discontinuity':
        with pytest.raises(RuntimeError, match='recreate the scene'):
            control.move_body(client, 1, 0, 0, 1)


def test_rotation_has_server_deadline_and_verifies_target_not_just_movement():
    client = FakeAirSimClient()
    client.action_effect_scale = 0.5
    result = strategy().turn_relative(client, 90)
    assert 0 < client.rotate_options['timeout_sec'] < 10
    assert client.rotate_options['margin'] == 5
    assert result.data['failure_kind'] == 'control_target_not_reached'
    assert client.cancel_calls == []  # Completed RPC, no stale task to cancel.


def test_freeze_failure_marks_scene_unusable_and_never_issues_receipt():
    control = strategy()
    client = FakeAirSimClient()
    def fail(*_):
        raise ConnectionError("freeze disconnected")
    control._actions._freeze_at_current_pose = fail
    result = control.move_body(client, 1, 0, 0, 1)
    assert not result.ok and result.execution_state['requires_scene_reset']
    assert result.execution_state['verified_paused_pose'] is None
    assert result.data['action_diagnostics']['timings_sec']['freeze'] >= 0
    held = control.move_body(client, 0, 0, 0, 1)
    assert not held.ok and held.execution_state['requires_scene_reset']
