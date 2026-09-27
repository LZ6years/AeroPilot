"""Read scheduling limits and ordering are tested with events, not speed guesses."""

from concurrent.futures import ThreadPoolExecutor
import threading
import json
import pytest

from src.env_client import ActionResult
from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig, load_read_only_execution_config
from src.runtime.control.read_only_executor import ReadJob, ReadOnlyBatchExecutor
from src.tools.schemas import ReadExecutionHints


def job(name, invoke, hints=None):
    return ReadJob(name, "test.read", hints or ReadExecutionHints(), invoke)


def ok(value="ok"):
    return ActionResult(True, "test.read", value)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.2, "4"])
def test_invalid_limits(value):
    for key in ("max_workers", "max_capability_calls"):
        with pytest.raises(ValueError):
            ReadOnlyExecutionConfig(**{key: value})
    with pytest.raises(ValueError):
        ReadOnlyExecutionConfig(capability_limits={"vision": value})


def test_config_default_and_explicit_file(tmp_path):
    assert ReadOnlyExecutionConfig().max_workers == 1
    limits = {"vision": 1}
    config = ReadOnlyExecutionConfig(4, 2, limits)
    limits["vision"] = 20
    assert config.capability_limit("vision") == 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_read_only_execution_config(path) == config
    path.write_text('{"unknown": 1}')
    with pytest.raises(ValueError):
        load_read_only_execution_config(path)


def test_synchronous_path_and_regular_failure(monkeypatch):
    import src.runtime.control.read_only_executor as module
    monkeypatch.setattr(module, "ThreadPoolExecutor", lambda **kw: pytest.fail("unnecessary thread pool"))
    caller = threading.get_ident()
    def call():
        assert threading.get_ident() == caller
        return ok()
    def fail():
        raise ValueError("failed read")
    results = ReadOnlyBatchExecutor().execute([job("1", fail), job("2", call)])
    assert not results[0].ok and results[1].ok
    assert ReadOnlyBatchExecutor(ReadOnlyExecutionConfig(4)).execute([job("1", call)])[0].ok


def test_capability_backlog_does_not_starve_light_reads_or_exceed_limits():
    release, knowledge, light, two_caps = (threading.Event() for _ in range(4))
    lock = threading.Lock()
    active = {"all": 0, "cap": 0}
    peak = dict(active)
    def work(kind):
        with lock:
            active["all"] += 1
            active["cap"] += int(kind == "cap")
            for key in active:
                peak[key] = max(peak[key], active[key])
            if active["cap"] == 2:
                two_caps.set()
        if kind == "knowledge":
            knowledge.set()
        if kind == "light":
            light.set()
        assert release.wait(5)
        with lock:
            active["all"] -= 1
            active["cap"] -= int(kind == "cap")
        return ok(kind)
    cap = ReadExecutionHints("capability", "vision")
    jobs = [job(str(i), lambda: work("cap"), cap) for i in range(6)]
    jobs += [job("knowledge", lambda: work("knowledge")), job("light", lambda: work("light"))]
    with ThreadPoolExecutor(1) as caller:
        future = caller.submit(ReadOnlyBatchExecutor(ReadOnlyExecutionConfig(4, 2)).execute, jobs)
        try:
            assert two_caps.wait(3) and knowledge.wait(3) and light.wait(3)
            assert peak == {"all": 4, "cap": 2}
        finally:
            release.set()
        assert len(future.result()) == 8


def test_per_capability_limit_and_same_key_fifo_with_reverse_completion():
    first_started, last_done, release = (threading.Event() for _ in range(3))
    order = []
    def first():
        order.append("first")
        first_started.set()
        assert release.wait(5)
        return ok("first")
    def second():
        order.append("second")
        return ok("second")
    def last():
        last_done.set()
        return ok("last")
    hint = ReadExecutionHints(serial_key="knowledge:same")
    with ThreadPoolExecutor(1) as caller:
        future = caller.submit(ReadOnlyBatchExecutor(ReadOnlyExecutionConfig(4)).execute,
                               [job("a", first, hint), job("b", second, hint), job("c", last)])
        try:
            assert first_started.wait(3) and last_done.wait(3)
            assert order == ["first"]
        finally:
            release.set()
        assert [r.message for r in future.result()] == ["first", "second", "last"]
        assert order == ["first", "second"]
    # A capability-specific cap can be stricter than the aggregate budget.
    entered, other_entered, release = (threading.Event() for _ in range(3))
    order = []
    def limited():
        order.append("vision")
        entered.set()
        assert release.wait(5)
        return ok()
    def other():
        other_entered.set()
        return ok()
    cap = ReadExecutionHints("capability", "vision")
    with ThreadPoolExecutor(1) as caller:
        future = caller.submit(ReadOnlyBatchExecutor(ReadOnlyExecutionConfig(4, 2, {"vision": 1})).execute,
            [job("1", limited, cap), job("2", limited, cap),
             job("3", other, ReadExecutionHints("capability", "other"))])
        try:
            assert entered.wait(3) and other_entered.wait(3)
            assert order == ["vision"]
        finally:
            release.set()
        assert len(future.result()) == 3 and order == ["vision", "vision"]


def test_fatal_exit_cancels_unscheduled_and_waits_for_started_work():
    started, release = threading.Event(), threading.Event()
    events = []
    class Fatal(BaseException):
        pass
    def held():
        started.set()
        assert release.wait(5)
        return ok()
    def fatal():
        assert started.wait(3)
        raise Fatal()
    scheduler = ReadOnlyBatchExecutor(ReadOnlyExecutionConfig(2), lambda n, p: events.append((n, p)))
    with ThreadPoolExecutor(1) as caller:
        future = caller.submit(scheduler.execute, [job("hold", held), job("fatal", fatal), job("pending", lambda: pytest.fail("ran pending"))])
        try:
            assert started.wait(3)
            # A callback proves cleanup began while the first job is still held.
            cancelled = threading.Event()
            old = scheduler.event_sink
            def sink(n, p):
                old(n, p)
                if n == "read_call_cancelled":
                    cancelled.set()
            scheduler.event_sink = sink
            # The event may have occurred before replacement; either is valid.
            assert any(n == "read_call_cancelled" for n, _ in events) or cancelled.wait(3)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(Fatal):
            future.result()
    assert any(n == "read_call_cancelled" and p["tool_call_id"] == "pending" for n, p in events)


def test_runtime_results_and_observations_remain_ordered_and_isolated(tmp_path):
    from tests.test_unified_tool_loop import harness, call
    from src.tools.schemas import ToolSpec, ToolRuntimePolicy
    runtime, state, scene, _ = harness(tmp_path, [])
    runtime.read_only_execution_config = ReadOnlyExecutionConfig(4)
    runtime.tools.registry.register(ToolSpec("test.read", "test_read", "skill_capability", "read", (),
        ToolRuntimePolicy("read_only", "immediate", "parallel_safe", True)))
    barrier = threading.Barrier(2)
    caller = threading.get_ident()
    written = []
    original_record = runtime._record_action
    def record(*a, **k):
        assert threading.get_ident() == caller
        written.append(a[2]["id"])
        return original_record(*a, **k)
    runtime._record_action = record
    def dispatch(scene_id, agent, name, args, runtime_context):
        runtime_context["latest_observation"]["nested"]["changed"] = args["id"]
        barrier.wait(timeout=3)
        assert runtime_context["latest_observation"]["nested"]["changed"] == args["id"]
        return ok(args["id"])
    runtime.tools.call = dispatch
    state.latest_observation["nested"] = {"changed": None}
    calls = [call("a", "test.read", id="a"), call("b", "test.read", id="b")]
    results = runtime._execute_native_read_calls(scene, "uav", state, "read", calls)
    assert [r["tool_call_id"] for r in results] == written == ["a", "b"]
    assert state.latest_observation["nested"] == {"changed": None}
