"""Fault injection for same-request recovery and independently draining lanes."""
import json
import threading
import time
from types import SimpleNamespace
import pytest

from src.llm import DefaultModelClient, ModelRequest, ProviderSettings, ProviderRequestError
from src.llm.recovery import RecoveryPolicy, RequestGuard, recovery_scope, service_error
from src.llm.providers.openai_compatible import decode_response
from evaluation.benchmarks.core.campaign.cleanup import CleanupChecks
from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue


class Clock:
    now = 0.
    def time(self): return self.now
    def sleep(self, value): self.now += value


class SDK:
    def __init__(self, responses):
        self.responses_list = iter(responses)
        self.calls = []
        self.closed = threading.Event()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **payload):
        self.calls.append(payload)
        result = next(self.responses_list)
        if isinstance(result, Exception): raise result
        return result
    def close(self): self.closed.set()


OK = {"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}


@pytest.mark.parametrize("failure", [TimeoutError("read"), service_error("无可用账号，请稍后重试"),
    service_error("账号并发窗口已满，请稍后重试或提高并发上限"),
    service_error("server_error · An error occurred while processing your request. You can retry your request"),
    service_error("Upstream stream ended prematurely; safe to retry"), ConnectionError("closed")])
def test_same_request_recovers_without_publishing_partial_response(failure):
    clock, trace = Clock(), {}
    sdk = SDK([failure, OK])
    client = DefaultModelClient(ProviderSettings(model_name="test"), sdk_client=sdk,
                               clock=clock.time, sleep=clock.sleep)
    with recovery_scope(RecoveryPolicy()):
        response = client.request(ModelRequest([{"role": "user", "content": "unchanged"}]), transport_trace=trace)
    assert response.text == "done"
    assert sdk.calls[0] == sdk.calls[1]
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["exception_chain"]
    assert 4 <= trace["attempts"][0]["backoff_elapsed_seconds"] <= 6


def test_common_six_attempt_budget_preserves_detail():
    clock, trace = Clock(), {}
    sdk = SDK([TimeoutError(), ConnectionError()]*3)
    client = DefaultModelClient(ProviderSettings(model_name="test"), sdk_client=sdk,
                               clock=clock.time, sleep=clock.sleep)
    with recovery_scope(RecoveryPolicy()), pytest.raises(ProviderRequestError):
        client.request(ModelRequest([]), transport_trace=trace)
    assert len(sdk.calls) == len(trace["attempts"]) == trace["error_attempt_count"] == 6


@pytest.mark.parametrize("message", ["authentication failed", "unrecognized failure"])
def test_non_retryable_service_error(message):
    sdk, trace = SDK([service_error(message)]), {}
    with recovery_scope(RecoveryPolicy()), pytest.raises(ProviderRequestError):
        DefaultModelClient(ProviderSettings(model_name="test"), sdk_client=sdk).request(ModelRequest([]), transport_trace=trace)
    assert len(sdk.calls) == 1


def test_blocked_read_closed_at_deadline_and_no_late_response():
    closed = threading.Event()
    guard = RequestGuard(RecoveryPolicy(deadline_sec=.08), time.monotonic()+.08, time.monotonic)
    start = time.monotonic()
    with pytest.raises(ProviderRequestError, match="deadline"):
        guard.run(lambda: closed.wait(5), closed.set)
    assert time.monotonic()-start < 1
    assert closed.wait(1)


def test_backoff_cancel_and_600_second_bound():
    clock = Clock()
    guard = RequestGuard(RecoveryPolicy(), 600, clock.time)
    with pytest.raises(ProviderRequestError, match="deadline") as error:
        guard.sleep(9999, clock.sleep)
    assert clock.now == 600
    assert error.value.retryable  # exhausted logical request still permits a later task attempt
    guard = RequestGuard(RecoveryPolicy(cancelled=lambda: True), 1000, clock.time)
    with pytest.raises(ProviderRequestError, match="cancelled"):
        guard.sleep(5, clock.sleep)
    assert clock.now == 600


def test_backoff_clock_advances_during_progress_without_negative_sleep():
    clock = Clock()
    # Progress persistence can finish after the remaining backoff interval.
    policy = RecoveryPolicy(progress=lambda _: clock.sleep(2))
    guard = RequestGuard(policy, 600, clock.time)
    sleeps = []
    guard.sleep(1, lambda value: sleeps.append(value))
    assert not sleeps


def test_sse_classification_requires_explicit_envelope():
    with pytest.raises(ProviderRequestError) as error:
        decode_response(b'data: {"error":{"message":"Account selection timed out"}}\n\n', stream=True)
    assert error.value.retryable
    legacy = {"error": {"error_type": "ValueError", "error": "model provider SSE error: safe to retry"}}
    assert ParallelRecoveryQueue.provider_error(legacy)["retryable"]
    assert not ParallelRecoveryQueue.infrastructure_record({"error": {"error_type": "ValueError", "error": "navigation"}})


def test_cleanup_lanes_finish_independently():
    pending = threading.Event()
    checks = CleanupChecks(budget=0)
    checks.start("A", lambda: (pending.wait(2), {"cleanup_verified": False})[1])
    checks.start("B", lambda: {"cleanup_verified": True})
    try:
        until = time.monotonic()+1
        receipt = None
        while receipt is None and time.monotonic() < until:
            receipt = checks.poll("B")
            time.sleep(.001)
        assert receipt["cleanup_verified"]
        assert checks.poll("A") is None
        assert "A" in checks.jobs
    finally:
        pending.set()
        checks.close()


def test_safe_ssh_retries_but_authentication_does_not(monkeypatch):
    from src.sim_server.remote.ssh import SshCommandRunner, SshOperationError
    import subprocess
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise subprocess.TimeoutExpired("ssh", 30)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    monkeypatch.setattr("src.sim_server.remote.ssh.subprocess.run", run)
    monkeypatch.setattr("src.sim_server.remote.ssh.time.sleep", lambda _: None)
    assert SshCommandRunner()._execute(["ssh"], "status", 30, safe=True) == "ok"
    assert len(calls) == 3
    calls.clear()
    def auth(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=255, stderr="Permission denied password=secret", stdout="")
    monkeypatch.setattr("src.sim_server.remote.ssh.subprocess.run", auth)
    with pytest.raises(SshOperationError) as exc:
        SshCommandRunner()._execute(["ssh"], "status", 30, safe=True)
    assert len(calls) == 1
    assert "secret" not in str(exc.value)
    assert not exc.value.retryable


def test_lost_start_receipt_adopts_only_identical_scene(tmp_path, monkeypatch):
    import hashlib
    from src.sim_server.remote import worker
    executable = tmp_path/"ue"
    executable.touch()
    settings = tmp_path/"settings.json"
    settings.write_text("{}")
    registry = tmp_path/"registry.json"
    spec = {"executable": str(executable), "settings_path": str(settings),
            "registry_path": str(registry), "log_path": str(tmp_path/"ue.log")}
    spec_path = tmp_path/"spec.json"
    spec_path.write_text(json.dumps(spec))
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()+b"{}").hexdigest()
    registry.write_text(json.dumps({"status": "running", "pid": 123, "pgid": 123, "spec_sha256": digest}))
    monkeypatch.setattr(worker, "_identity_matches", lambda _: True)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate start"))
    assert worker._start_scene_locked(str(spec_path))["reused"]
    settings.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="mismatched"):
        worker._start_scene_locked(str(spec_path))
    monkeypatch.setattr(worker, "_identity_matches", lambda _: False)
    monkeypatch.setattr(worker, "_live_group_members", lambda _: [123])
    with pytest.raises(RuntimeError, match="uncertain"):
        worker._start_scene_locked(str(spec_path))


@pytest.mark.parametrize("message", ["model provider SSE error: safe to retry", "sleep length must be non-negative"])
def test_historical_sse_audit_is_idempotent_and_keeps_navigation_failures(tmp_path, message):
    tasks = [{"split": "seen", "episode_id": name} for name in ("service", "navigation")]
    queue = ParallelRecoveryQueue(tmp_path/"progress.json", tasks, "fixed")
    for task in tasks:
        result = {"episode_id": task["episode_id"], "metrics": {"success": False}}
        if task["episode_id"] == "service":
            result["error"] = {"error_type": "ValueError", "error": message}
        queue.state["tasks"][queue.key(task)].update(status="scored", result=result, attempts=[{
            "number": 1, "status": "scored", "result": result, "valid_score": True, "cleanup_verified": True}])
    assert len(queue.reconcile_scored_errors()) == 1
    service = queue.state["tasks"][queue.key(tasks[0])]
    assert service["result"] is None and service["status"] == "retry_pending"
    assert service["attempts"][0]["result"]["error"]["error_type"] == "ValueError"
    assert queue.state["tasks"][queue.key(tasks[1])]["status"] == "scored"
    before = (tmp_path/"progress.json").read_bytes()
    assert queue.reconcile_scored_errors() == []
    assert (tmp_path/"progress.json").read_bytes() == before


def test_reused_tunnel_pid_never_signalled_and_open_port_blocks_cleanup(tmp_path, monkeypatch):
    from evaluation.benchmarks.core import supervision
    from src.sim_server.remote import recovery
    root = tmp_path/"runs/case"
    (root/"state").mkdir(parents=True)
    (root/"scenes/owned").mkdir(parents=True)
    (root/"state/scene_instance.json").write_text(json.dumps({"scene_instance_id": "owned", "map_id": "map",
        "process_ownership": "remote_managed", "remote_process_id": 123, "control_port": 35001}))
    (root/"scenes/owned/settings.json").write_text('{"ApiServerPort":30001}')
    monkeypatch.setattr(supervision, "load_map_config", lambda _: ([SimpleNamespace(map_id="map", connection={"mode": "remote"})], None))
    monkeypatch.setattr(supervision.psutil, "Process", lambda _: SimpleNamespace(create_time=lambda: 999,
        terminate=lambda: pytest.fail("must not terminate reused PID")))
    control = SimpleNamespace(status=lambda *a: {"pid": 123, "scene_instance_id": "owned"},
        stop=lambda *a: {"cleanup_verified": True}, probe_prepared_port=lambda *a: {"status": "unreachable"})
    owner = {"pid": 44, "created": 1, "command": ["ssh", "-L", "127.0.0.1:35001:127.0.0.1:30001"]}
    for closed in (False, True):
        monkeypatch.setattr(recovery, "_closed_local_port", lambda *a: closed)
        receipts = supervision.cleanup_owned(tmp_path/"runs", "maps", [owner], control=control)
        assert receipts[0]["obsolete_tunnel_pids"] == [44]
        assert supervision.cleanup_verified(receipts) == closed
