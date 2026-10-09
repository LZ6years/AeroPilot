"""Operator control exercises real loop boundaries and complete physical batches."""

import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep

import pytest

from src.runtime.control.runtime_control import RuntimeControl
from tests.test_unified_tool_loop import public_runtime, move, done


def until(predicate, timeout=5):
    deadline = monotonic() + timeout
    while not predicate():
        if monotonic() > deadline:
            raise AssertionError("Timed out waiting for control boundary")
        sleep(.01)


def test_pause_after_provider_retains_turn_until_explicit_resume(tmp_path):
    runtime, backend = public_runtime(tmp_path, [[move("move")], [done()]])
    control = RuntimeControl()
    original = runtime.planner.plan_turn
    def planner(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(runtime.planner.contexts) == 1:
            control.request("pause")
        return result
    runtime.planner.plan_turn = planner
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(runtime.run_task, "map", "Move", control=control)
        until(lambda: control.snapshot()["status"] == "paused")
        assert not any(name == "move_body" for name, _ in backend.actions)
        assert len(runtime.planner.contexts) == 1
        control.request("resume")
        state = future.result(5)
    assert state.completed and backend.pose["x"] == 2
    assert len(runtime.planner.contexts) == 2


def test_stop_received_during_provider_discards_returned_action(tmp_path):
    runtime, backend = public_runtime(tmp_path, [[move("must_not_execute")]])
    control = RuntimeControl()
    original = runtime.planner.plan_turn
    def planner(*args, **kwargs):
        result = original(*args, **kwargs)
        control.request("stop")
        return result
    runtime.planner.plan_turn = planner
    state = runtime.run_task("map", "Move", control=control)
    assert state.stopped and not state.completed
    assert not any(name == "move_body" for name, _ in backend.actions)
    assert any(name == "operator_decision_discarded" for name, _ in runtime.recorder.events)
    assert not any(item.kind == "assistant_turn" for item in runtime.planner.transcript.items)


def test_pause_finishes_entire_batch_and_after_observation(tmp_path):
    runtime, backend = public_runtime(tmp_path, [[move("first"),move("second")],[done()]])
    control = RuntimeControl()
    execute = backend.execute_action
    def action(scene, agent, name, args):
        result = execute(scene, agent, name, args)
        if name == "move_body" and backend.pose["x"] == 2:
            control.request("pause")
        return result
    backend.execute_action = action
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(runtime.run_task, "map", "Move", control=control)
        until(lambda: control.snapshot()["status"] == "paused")
        assert backend.pose["x"] == 4
        windows = [payload for name,payload in runtime.recorder.events if name=="physical_window_recorded"]
        assert len(windows)==1 and windows[0]["observation_after_id"]
        assert len(windows[0]["ordered_steps"])==2
        control.request("stop")
        state = future.result(5)
    assert state.stopped
    assert len(runtime.planner.contexts)==1


def test_stop_latch_cannot_resume_or_pause():
    control=RuntimeControl()
    control.request("stop")
    for action in ("resume","pause"):
        with pytest.raises(ValueError):
            control.request(action)
    assert not control.checkpoint("before_execution",lambda _:None)
