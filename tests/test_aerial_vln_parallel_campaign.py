"""Deterministic dual-lane ownership, recovery and artifact regression tests."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.aerial_vln import parallel_campaign as parallel


def task(identity, difficulty="low", split="val_seen"):
    return {"split": split, "episode_id": identity, "scene_id": "2", "difficulty": difficulty}


def queue(tmp_path, tasks):
    return parallel.ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")


def receipt(current, *, provider=False, cleanup=True, retryable=True, reason=None):
    return {"cleanup_verified": cleanup, "valid_score": not provider and not reason,
            "process": {"reason": reason}, "result": {
                "episode_id": current["episode_id"], "metrics": {"success": False},
                "infrastructure_error": {"stage": "provider", "error": {
                    "category": "tls", "attempts": 9, "exhaustion_reason": "deadline",
                    "retryable": retryable}} if provider else None}}


def infrastructure_receipt(current, *, cleanup=True, stage="cleanup", errors=None):
    """Match the executor's final-observation failure, retaining partial metrics."""
    value = receipt(current, cleanup=cleanup)
    value["valid_score"] = False
    value["result"]["infrastructure_error"] = {"stage": stage, "errors": errors if errors is not None else [
        {"stage": "final_observation", "error_type": "TimeoutError", "error": "Request timed out"}]}
    return value


def write(path, value):
    parallel.write_json(path, value)


def test_shared_json_read_recovers_seven_access_conflicts_without_losing_state(tmp_path, monkeypatch):
    path = tmp_path / "host_status.json"
    write(path, {"active": ["A", "B"], "pid": 42})
    original = Path.read_text
    attempts, waits = [], []
    def read(current, *args, **kwargs):
        attempts.append(current)
        if len(attempts) <= 7:
            raise PermissionError(13, "temporary sharing conflict", str(current))
        return original(current, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(parallel.time, "sleep", waits.append)
    assert parallel.read_json(path) == {"active": ["A", "B"], "pid": 42}
    assert len(attempts) == 8 and waits == list(parallel.FILE_ACCESS_BACKOFF)


def test_persistent_file_access_error_is_bounded_and_never_returns_empty_state(monkeypatch):
    attempts, waits = [], []
    def fail(*args, **kwargs):
        attempts.append(True)
        raise PermissionError(13, "persistent access denial")
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(parallel.time, "sleep", waits.append)
    with pytest.raises(PermissionError, match="persistent"):
        parallel.read_json("host_status.json")
    assert len(attempts) == 8 and sum(waits) < 2


def test_atomic_metadata_write_retries_replace_and_preserves_old_complete_json(tmp_path, monkeypatch):
    path = tmp_path / "commands/A.json"
    write(path, {"id": "old"})
    replace = parallel.os.replace
    calls = []
    def locked(source, destination):
        calls.append(True)
        if len(calls) <= 2:
            assert json.loads(path.read_text(encoding="utf-8")) == {"id": "old"}
            raise PermissionError(13, "replacement sharing conflict")
        return replace(source, destination)
    monkeypatch.setattr(parallel.os, "replace", locked)
    monkeypatch.setattr(parallel.time, "sleep", lambda _: None)
    write(path, {"id": "new"})
    assert len(calls) == 3 and json.loads(path.read_text(encoding="utf-8")) == {"id": "new"}
    assert not list(path.parent.glob("*.tmp"))


def test_queue_save_retries_without_claiming_a_second_attempt(tmp_path, monkeypatch):
    current = task("one")
    q = parallel.ParallelRecoveryQueue(tmp_path / "progress.json", [current], "frozen",
                                       save_retry=parallel.retry_file_access)
    calls = []
    original = parallel.os.replace
    def locked(source, destination):
        calls.append(True)
        if len(calls) <= 2:
            raise PermissionError(13, "progress temporarily open")
        return original(source, destination)
    monkeypatch.setattr(parallel.os, "replace", locked)
    monkeypatch.setattr(parallel.time, "sleep", lambda _: None)
    q.claim([current], "A")
    state = json.loads(q.path.read_text(encoding="utf-8"))
    assert len(state["tasks"]["val_seen/one"]["attempts"]) == 1 and len(calls) == 3


def test_missing_partial_and_other_io_failures_keep_distinct_semantics(tmp_path, monkeypatch):
    path = tmp_path / "missing.json"
    assert parallel.read_json(path) == {}
    path.write_text("{", encoding="utf-8")
    assert parallel.read_json(path) == {}
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(OSError(28, "disk failure")))
    monkeypatch.setattr(parallel.time, "sleep", lambda _: pytest.fail("unrelated disk errors must not retry"))
    with pytest.raises(OSError, match="disk failure"):
        parallel.read_json(path)


def test_native_diagnostic_bytes_do_not_mask_actual_host_failure(tmp_path):
    path = tmp_path / 'stderr.log'
    raw = b'Loading weights: \xa8\x80\nCUDA out of memory\n'
    path.write_bytes(raw)
    text = parallel.read_process_diagnostic(path)
    assert 'CUDA out of memory' in text and '\\xa8\\x80' in text
    assert path.read_bytes() == raw


def test_exit_status_replaces_stale_running_lanes_with_final_queue_state(tmp_path):
    current = task("one")
    q = queue(tmp_path, [current])
    q.claim([current], "A")
    q.accept(current, 1, receipt(current))
    q.pause("manual_interrupt")
    write(tmp_path / "parallel/status.json", {"status": "running", "lanes": {"A": current}})
    parallel.write_exit_status(tmp_path, q, workers=2, gate=True, host_pid=10)
    status = parallel.read_json(tmp_path / "parallel/status.json")
    exit_state = parallel.read_json(tmp_path / "parallel/exit.json")
    assert status["status"] == exit_state["status"] == "paused"
    assert status["reason"] == exit_state["reason"] == "manual_interrupt"
    assert status["counts"] == {"scored": 1} and status["lanes"] == {}


def test_image_campaign_report_failure_does_not_stop_or_modify_queue(tmp_path, monkeypatch):
    current = task("one")
    q = queue(tmp_path, [current])
    q.claim([current], "A")
    write(tmp_path / "manifest.json", {"experiment": "image_budget"})
    before = q.path.read_bytes()
    monkeypatch.setattr(parallel, "_render_reports", lambda *args: (_ for _ in ()).throw(TypeError("bad audit field")))
    assert parallel.render_reports(tmp_path, q) is None
    assert q.path.read_bytes() == before
    assert parallel.read_json(tmp_path / "parallel/report_error.json")["pending"] is True


def test_storage_forecast_uses_future_logging_profile_not_old_full_debug_sizes(tmp_path):
    current = task("one")
    q = queue(tmp_path, [current])
    q.state["tasks"][q.key(current)]["attempts"] = [{"retained_bytes": 3 * 1024**3, "logging_profile": "full"}]
    write(tmp_path / "parallel/execution_overlay.json", {"logging_profile": "paper_process_metrics_v1"})
    assert parallel.storage_budget(tmp_path, q, [current], 2)
    check = parallel.read_json(tmp_path / "storage_check.json")
    assert check["calibration_pending"] and check["observed_cases"] == 0
    assert check["required_bytes"] == 10 * 1024**3


def test_reporting_extension_only_replaces_verified_evaluation_readers(tmp_path, monkeypatch):
    import hashlib
    import sys
    import evaluation.experiments as package
    import evaluation.experiments.compression_cost as old_cost
    import evaluation.experiments.image_budget_report as old_report
    from src.llm import client as original_client
    hashes = {}
    for filename, old in (("compression_cost.py", old_cost), ("image_budget_report.py", old_report)):
        name = "evaluation.experiments." + Path(filename).stem
        monkeypatch.setitem(sys.modules, name, old)
        monkeypatch.setattr(package, Path(filename).stem, old)
        path = tmp_path / filename
        path.write_text("REPORT_TEST_MARKER = True\n", encoding="utf-8")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    overlay = {"controller_script": str(tmp_path / "parallel_campaign.py"), "source_hashes": hashes}
    parallel.install_report_readers(overlay, {"experiment": "image_budget"})
    assert sys.modules["evaluation.experiments.compression_cost"].REPORT_TEST_MARKER
    assert sys.modules["src.llm.client"] is original_client
    (tmp_path / "compression_cost.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Unverified"):
        parallel.install_report_readers(overlay, {"experiment": "image_budget"})


def test_two_lanes_claim_different_tasks_and_reject_duplicate_lane(tmp_path):
    tasks = [task("one"), task("two"), task("three")]
    q = queue(tmp_path, tasks)
    first = q.claim(tasks, "A")
    second = q.claim(tasks, "B")
    assert first[0] == tasks[0] and second[0] == tasks[1]
    assert json.loads(q.path.read_text())["tasks"]["val_seen/one"]["status"] == "running"
    with pytest.raises(ValueError, match="lane"):
        q.claim(tasks, "A")
    q.accept(*[first[0], first[1]["number"]], receipt(tasks[0]))
    assert q.claim(tasks, "A")[0] == tasks[2]


@pytest.mark.parametrize("unfinished", ["pending", "running", "retry_pending", "retry_exhausted"])
def test_budget_stage_waits_for_all_valid_scores_including_retries(tmp_path, unfinished):
    tasks = [{**task(identity), "dataset": "aerial_vln", "image_budget": budget}
             for identity in ("one", "two") for budget in (30, 10, 20)]
    manifest = {"experiment": "image_budget", "image_budgets": [10, 20, 30]}
    q = queue(tmp_path, tasks)
    for value in q.state["tasks"].values():
        value["status"] = "scored"
    current = next(t for t in tasks if t["image_budget"] == 10)
    q.state["tasks"][q.key(current)]["status"] = unfinished
    budget, candidates = parallel.image_budget_stage(manifest, tasks, q)
    assert budget == 10 and {t["image_budget"] for t in candidates} == {10}
    q.state["tasks"][q.key(current)]["status"] = "scored"
    later = next(t for t in tasks if t["image_budget"] == 20)
    q.state["tasks"][q.key(later)]["status"] = "pending"
    assert parallel.image_budget_stage(manifest, tasks, q)[0] == 20


def test_budget_local_pilot_unlocks_only_its_cohort_and_preserves_other_scores(tmp_path):
    tasks = [{**task(str(index)), "dataset": "aerial_vln", "image_budget": budget}
             for budget in (10, 20, 30) for index in range(5)]
    manifest = {"experiment": "image_budget", "image_budgets": [10, 20, 30]}
    q = queue(tmp_path, tasks)
    # Completed measurements from the previous interleaved schedule survive.
    q.state["tasks"][q.key(tasks[5])]["status"] = "scored"
    budget, cohort = parallel.image_budget_stage(manifest, tasks, q)
    trial = cohort[:4]
    assert parallel.stage_candidates(cohort, trial, False, q) == trial[:2]
    for t in trial:
        q.state["tasks"][q.key(t)]["status"] = "scored"
    assert parallel.stage_candidates(cohort, trial, True, q) == cohort
    remaining, _ = q.claim(cohort, "A")
    assert remaining == cohort[4] and remaining["image_budget"] == 10
    q.accept(remaining, 1, receipt(remaining))
    assert parallel.image_budget_stage(manifest, tasks, q)[0] == 20
    assert q.state["tasks"][q.key(tasks[5])]["status"] == "scored"
    assert parallel.technical_check_path(tmp_path, 10) != parallel.technical_check_path(tmp_path, 20)


def test_legacy_campaign_keeps_manifest_order(tmp_path):
    tasks = [task("one"), task("two")]
    q = queue(tmp_path, tasks)
    assert parallel.image_budget_stage({}, tasks, q) == (None, tasks)
    assert parallel.technical_check_path(tmp_path, None) == tmp_path / "parallel/trial_check.json"


def test_workers_cannot_mutate_or_save_progress(tmp_path):
    q = queue(tmp_path, [task("one")])
    with ThreadPoolExecutor(1) as pool:
        for operation in (q.save, lambda: q.claim([task("one")], "A")):
            with pytest.raises(RuntimeError, match="controller"):
                pool.submit(operation).result()
    assert q.state["tasks"]["val_seen/one"]["status"] == "pending"


def test_tls_failure_defers_while_other_lane_scores_then_retry_once(tmp_path):
    tasks = [task("one"), task("two"), task("three")]
    q = queue(tmp_path, tasks)
    one, a = q.claim(tasks, "A")
    two, b = q.claim(tasks, "B")
    assert q.accept(one, a["number"], receipt(one, provider=True)) == "deferred"
    assert q.accept(two, b["number"], receipt(two)) == "scored"
    three, c = q.claim(tasks, "A")
    assert three == tasks[2]
    q.accept(three, c["number"], receipt(three))
    retry, attempt = q.claim(tasks, "B")
    assert retry == one and attempt["number"] == 2
    assert q.accept(one, 2, receipt(one)) == "scored"
    assert q.accept(one, 1, receipt(one, provider=True)) == "duplicate"
    assert q.state["tasks"]["val_seen/one"]["result"]["metrics"]["success"] is False
    assert q.claim(tasks, "A") is None


def test_exhausted_tls_trajectory_never_enters_sr(tmp_path):
    current = task("one")
    q = queue(tmp_path, [current])
    for index in range(3):
        _, attempt = q.claim([current], "A")
        q.accept(current, attempt["number"], receipt(current, provider=True))
    item = q.state["tasks"]["val_seen/one"]
    assert item["result"] is None and item["status"] == "retry_exhausted"
    assert len(item["attempts"]) == 3
    assert item["attempts"][-1]["result"]["infrastructure_error"]["error"]["attempts"] == 9
    assert q.claim([current], "B") is None


@pytest.mark.parametrize("flat", [True, False])
def test_wrapped_freeze_timeout_defers_and_healthy_lane_keeps_result(tmp_path, flat):
    tasks = [task("timeout"), task("healthy")]
    q = queue(tmp_path, tasks)
    failed, a = q.claim(tasks, "A")
    healthy, b = q.claim(tasks, "B")
    error = {"error_type": "SceneRecoveryRequired", "error": "AirSim freeze failed: Request timed out"}
    proof = infrastructure_receipt(failed, stage="scene_control", errors=[error])
    if flat:
        proof["result"]["infrastructure_error"] = dict(error)
    raw = json.dumps(proof, sort_keys=True)
    assert q.accept(failed, a["number"], proof) == "deferred"
    assert json.dumps(proof, sort_keys=True) == raw
    assert q.state["tasks"][q.key(failed)]["result"] is None
    assert q.accept(healthy, b["number"], receipt(healthy)) == "scored"
    # Old paused receipts become recoverable using the identical classifier.
    q.pause("infrastructure_failure", task=failed, number=a["number"])
    assert q.recoverable_pause(q.state)
    assert q.state["tasks"][q.key(healthy)]["status"] == "scored"


@pytest.mark.parametrize("message,cleanup,retryable", [
    ("AirSim freeze failed: Request timed out", False, True),
    ("AirSim freeze failed: Request timed out", True, False),
    ("AirSim freeze failed: unknown control state", True, True),
    ("Request timed out", True, True),
    ("AirSim freeze failed: permission denied", True, True),
])
def test_scene_reset_is_not_a_blanket_retry_permission(message, cleanup, retryable):
    error = {"error_type": "SceneRecoveryRequired", "error": message, "retryable": retryable}
    value = infrastructure_receipt(task("case"), cleanup=cleanup, stage="scene_control", errors=[error])
    assert not parallel.ParallelRecoveryQueue.retryable_infrastructure(value)


def test_wrapped_scene_timeout_keeps_original_three_attempt_limit(tmp_path):
    current = task("timeout")
    q = queue(tmp_path, [current])
    for number in range(1, 4):
        _, attempt = q.claim([current], "A")
        value = infrastructure_receipt(current, stage="scene_control", errors=[{
            "error_type": "SceneRecoveryRequired", "error": "AirSim freeze failed: Request timed out"}])
        assert q.accept(current, number, value) == "deferred"
    assert q.state["tasks"][q.key(current)]["status"] == "retry_exhausted"
    assert q.claim([current], "A") is None


def test_two_final_observation_timeouts_defer_without_interrupting_other_lane_then_score_once(tmp_path):
    tasks = [task("one"), task("two"), task("three")]
    q = queue(tmp_path, tasks)
    one, first = q.claim(tasks, "A")
    two, other = q.claim(tasks, "B")
    assert q.accept(one, first["number"], infrastructure_receipt(one)) == "deferred"
    assert q.state["status"] != "paused" and q.state["tasks"][q.key(two)]["status"] == "running"
    assert q.state["tasks"][q.key(one)]["result"] is None
    assert q.accept(two, other["number"], receipt(two)) == "scored"
    three, next_attempt = q.claim(tasks, "A")
    assert three == tasks[2]
    q.accept(three, next_attempt["number"], receipt(three))
    retry, second = q.claim(tasks, "B")
    assert retry == one and second["number"] == 2
    assert q.accept(retry, 2, infrastructure_receipt(retry)) == "deferred"
    retry, third = q.claim(tasks, "B")
    assert third["number"] == 3
    success = receipt(retry)
    success["result"]["metrics"]["success"] = True
    assert q.accept(retry, 3, success) == "scored"
    assert q.accept(one, 1, infrastructure_receipt(one)) == "duplicate"
    assert sum(x["status"] == "scored" for x in q.state["tasks"].values()) == 3
    assert len(q.state["tasks"][q.key(one)]["attempts"]) == 3


def test_three_cleanup_timeouts_exhaust_only_that_episode_while_other_work_continues(tmp_path):
    one, other = task("one"), task("other")
    q = queue(tmp_path, [one, other])
    for number in range(1, 4):
        _, attempt = q.claim([one], "A")
        assert attempt["number"] == number
        assert q.accept(one, number, infrastructure_receipt(one)) == "deferred"
    assert q.state["tasks"][q.key(one)]["status"] == "retry_exhausted"
    assert q.state["tasks"][q.key(one)]["result"] is None and q.state["status"] != "paused"
    assert q.claim([one, other], "A")[0] == other


@pytest.mark.parametrize('shape', ['single', 'list', 'both', 'legacy', 'flat', 'flat_cleanup'])
def test_infrastructure_error_shapes_share_the_same_transport_classifier(shape):
    value = infrastructure_receipt(task('one'))
    error = {'error_type': 'ConnectionResetError', 'error': 'connection reset'}
    infrastructure = value['result']['infrastructure_error']
    if shape == 'single':
        infrastructure.pop('errors')
        infrastructure['error'] = error
    elif shape == 'list':
        infrastructure['errors'].append(error)
    elif shape == 'both':
        infrastructure['error'] = error
    elif shape == 'legacy':
        infrastructure.pop('errors')
        value['result']['error'] = error
    elif shape == 'flat':
        value['result']['infrastructure_error'] = error
    else:
        value['result']['infrastructure_error'] = {'error_type': 'CleanupError', 'error': 'cleanup failed', 'details': [error]}
    before = json.dumps(value, sort_keys=True)
    assert parallel.ParallelRecoveryQueue.retryable_infrastructure(value)
    assert json.dumps(value, sort_keys=True) == before


def test_flat_scene_timeout_defers_without_stopping_other_lane_or_reading_text_as_http_metadata(tmp_path):
    tasks = [task('travel'), task('aerial')]
    q = queue(tmp_path, tasks)
    _, attempt = q.claim(tasks, 'A')
    q.claim(tasks, 'B')
    failed = infrastructure_receipt(tasks[0])
    failed['result']['infrastructure_error'] = {'error_type': 'TimeoutExpired', 'error': 'SSH probe timed out after 30 seconds'}
    original = json.dumps(failed['result'], sort_keys=True)
    assert q.accept(tasks[0], attempt['number'], failed) == 'deferred'
    assert q.state['tasks'][q.key(tasks[1])]['status'] == 'running'
    assert q.state['tasks'][q.key(tasks[0])]['result'] is None
    assert q.provider_error(failed['result']) == {}
    assert json.dumps(failed['result'], sort_keys=True) == original


@pytest.mark.parametrize('kind', ['ValueError', 'AuthenticationError', 'SceneRecoveryRequired'])
def test_flat_unknown_errors_still_pause_instead_of_becoming_valid_scores(tmp_path, kind):
    current = task('one')
    q = queue(tmp_path, [current])
    q.claim([current], 'A')
    failed = infrastructure_receipt(current)
    failed['result']['infrastructure_error'] = {'error_type': kind, 'error': 'cannot classify'}
    assert q.accept(current, 1, failed) == 'pause'
    assert q.state['tasks'][q.key(current)]['result'] is None


def test_provider_metadata_requires_provider_stage_and_handles_flat_provider_receipts():
    cls = parallel.ParallelRecoveryQueue
    error = {'error_type': 'ProviderRequestError', 'status_code': 429, 'retryable': True}
    assert cls.provider_error({'infrastructure_error': {'stage': 'provider', 'error': error}}) == error
    assert cls.provider_error({'infrastructure_error': error}) == error
    assert cls.provider_error({'infrastructure_error': {'error_type': 'TimeoutExpired', 'error': 'SSH text'}}) == {}
    assert cls.provider_error({'infrastructure_error': {'stage': 'provider', 'error': 'malformed'}}) == {}
    assert cls.provider_error({'infrastructure_error': 'malformed'}) == {}


@pytest.mark.parametrize('fault', ['unknown', 'authentication', 'mixed', 'missing', 'malformed',
                                  'cleanup_unverified', 'retry_forbidden', 'provider'])
def test_unverified_or_unknown_infrastructure_cannot_be_classified_as_recoverable(tmp_path, fault):
    current = task('one')
    value = infrastructure_receipt(current)
    infrastructure = value['result']['infrastructure_error']
    if fault in {'unknown', 'authentication'}:
        infrastructure['errors'] = [{'error_type': 'ValueError' if fault == 'unknown' else 'AuthenticationError'}]
    elif fault == 'mixed':
        infrastructure['error'] = {'error_type': 'ValueError'}
    elif fault == 'missing':
        infrastructure['errors'] = []
    elif fault == 'malformed':
        infrastructure['errors'].append('not an error receipt')
    elif fault == 'cleanup_unverified':
        value['cleanup_verified'] = False
    elif fault == 'retry_forbidden':
        infrastructure['errors'][0]['retryable'] = False
    else:
        infrastructure['stage'] = 'provider'
    assert not parallel.ParallelRecoveryQueue.retryable_infrastructure(value)
    if fault != 'provider':
        q = queue(tmp_path, [current])
        q.claim([current], 'A')
        assert q.accept(current, 1, value) == 'pause'
        assert q.state['pause_cause']['task_key'] == q.key(current)
        assert not q.recoverable_pause(q.state)


def test_pause_cause_survives_finalization_and_legacy_progress_remains_readable(tmp_path):
    current = task('one')
    q = queue(tmp_path, [current])
    q.claim([current], 'A')
    failed = infrastructure_receipt(current, errors=[{'error_type': 'ValueError'}])
    q.accept(current, 1, failed)
    cause = dict(q.state['pause_cause'])
    q.pause('infrastructure_failure')
    restored = queue(tmp_path, [current])
    assert restored.state['pause_cause'] == cause and not restored.recoverable_pause(restored.state)
    restored.state.pop('pause_cause')
    restored.save()
    legacy = queue(tmp_path, [current])
    assert legacy.state['tasks'] == q.state['tasks'] and not legacy.recoverable_pause(legacy.state)
    legacy.pause('manual_interrupt')
    assert 'pause_cause' not in legacy.state


def test_cleanup_failure_preserves_other_valid_score_and_untouched_work(tmp_path):
    tasks = [task("one"), task("two"), task("three")]
    q = queue(tmp_path, tasks)
    q.claim(tasks, "A")
    q.claim(tasks, "B")
    assert q.accept(tasks[0], 1, receipt(tasks[0], cleanup=False)) == "pause"
    assert q.accept(tasks[1], 1, receipt(tasks[1])) == "scored"
    assert q.state["status"] == "paused" and q.state["reason"] == "cleanup_unverified"
    assert q.state["tasks"]["val_seen/three"]["status"] == "pending"


def test_crash_recovery_adopts_complete_result_and_defers_partial_task(tmp_path):
    tasks = [task("one"), task("two"), task("three")]
    q = queue(tmp_path, tasks)
    q.claim(tasks, "A")
    q.claim(tasks, "B")
    restored = queue(tmp_path, tasks)
    assert restored.recover_leases(lambda current, _: receipt(
        current, reason="host_interrupted" if current == tasks[1] else None))
    assert restored.state["tasks"]["val_seen/one"]["status"] == "scored"
    assert restored.state["tasks"]["val_seen/two"]["status"] == "retry_pending"
    assert restored.state["tasks"]["val_seen/three"]["status"] == "pending"


def test_recovery_checks_every_owned_lane_even_when_first_cleanup_fails(tmp_path):
    tasks = [task("one"), task("two")]
    q = queue(tmp_path, tasks)
    q.claim(tasks, "A")
    q.claim(tasks, "B")
    inspected = []
    def recover(current, attempt):
        inspected.append(current["episode_id"])
        return receipt(current, cleanup=current["episode_id"] != "one")
    assert not q.recover_leases(recover)
    assert inspected == ["one", "two"]
    assert q.state["tasks"]["val_seen/two"]["status"] == "scored"
    assert q.state["reason"] == "interrupted_cleanup_unverified"


def test_deferred_unverified_cleanup_is_rechecked_before_any_dispatch(tmp_path):
    current = task("one")
    q = queue(tmp_path, [current])
    q.claim([current], "A")
    q.accept(current, 1, receipt(current, cleanup=False))
    assert q.recover_leases(lambda *args: receipt(current))
    assert q.state["tasks"]["val_seen/one"]["status"] == "scored"
    assert q.claim([current], "B") is None


def test_receipt_identity_and_reserved_metadata_are_checked(tmp_path):
    one, two = task("one"), task("two")
    q = queue(tmp_path, [one, two])
    with pytest.raises(ValueError, match="ownership"):
        q.claim([one], "A", metadata={"number": 10})
    q.claim([one], "A")
    with pytest.raises(ValueError, match="identity"):
        q.accept(one, 1, receipt(two))
    with pytest.raises(ValueError, match="current attempt"):
        q.accept(one, 2, receipt(one))


def test_provider_outage_requires_service_check_and_auth_pauses(tmp_path):
    tasks = [task(str(index)) for index in range(4)]
    q = queue(tmp_path, tasks)
    actions = []
    for current in tasks[:3]:
        leased, attempt = q.claim(tasks, "A")
        actions.append(q.accept(leased, attempt["number"], receipt(leased, provider=True)))
    assert actions == ["deferred", "deferred", "service_check"]
    leased, attempt = q.claim(tasks, "B")
    assert q.accept(leased, attempt["number"], receipt(leased, provider=True, retryable=False)) == "pause"
    assert q.state["reason"] == "non_retryable_provider_error"


def catalog():
    return {"port_policy": {"control_port_start": 35001, "max_scene_num": 8},
            "maps": [{"map_id": "map", "connection": {"mode": "remote"},
                      "airsim": {"port": 30001, "camera_name": "front_0"},
                      "launch": {"args": ["-RenderOffscreen", "-GraphicsAdapter=0"],
                                 "existing_process_policy": "reject"}}]}


def gpu_binding(index):
    device_uuid = "GPU-00000000-0000-0000-0000-%012d" % index
    return {"remote_gpu": index, "uuid": device_uuid, "pci_bus_id": "00000000:86:00.0",
            "visible_vulkan_uuids": [device_uuid], "vulkan_index": 0,
            "environment": {"VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
                            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
                            "CUDA_VISIBLE_DEVICES": device_uuid}}


def test_lane_configs_change_only_ports_and_allowed_gpu_without_mutating_base():
    original = catalog()
    a, b = (parallel.lane_catalog(original, lane, gpu_binding(index))
            for lane, index in (("A", 0), ("B", 2)))
    assert original == catalog()
    assert a["port_policy"]["control_port_start"] == 35001
    assert b["port_policy"]["control_port_start"] == 35002
    assert a["maps"][0]["airsim"]["port"] == 30001
    assert b["maps"][0]["airsim"]["port"] == 30002
    assert b["maps"][0]["launch"]["args"] == ["-RenderOffscreen", "-GraphicsAdapter=0"]
    assert b["maps"][0]["launch"]["env"]["CUDA_VISIBLE_DEVICES"] == gpu_binding(2)["uuid"]
    assert b["maps"][0]["airsim"]["camera_name"] == "front_0"
    assert "-GraphicsAdapter=1" not in json.dumps((a, b))


def test_cached_models_require_complete_existing_weights_and_record_hashes(tmp_path):
    settings = {"grounding_model": "org/dino", "segmentation_model": "org/sam"}
    for model in settings.values():
        directory = tmp_path / ("models--" + model.replace("/", "--"))
        (directory / "refs").mkdir(parents=True)
        (directory / "refs/main").write_text("revision", encoding="utf-8")
        snapshot = directory / "snapshots/revision"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"unchanged-weights")
    records = parallel.cached_models(settings, tmp_path)
    assert len(records) == 2 and records[0]["revision"] == "revision"
    assert records[0]["files"]["model.safetensors"] == hashlib.sha256(b"unchanged-weights").hexdigest()
    (Path(records[1]["snapshot"]) / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        parallel.cached_models(settings, tmp_path)


def test_trial_uses_unscored_seen_and_covers_all_difficulties(tmp_path):
    tasks = [task("already"), task("low"), task("medium", "medium"),
             task("high", "high"), task("high2", "high")]
    q = queue(tmp_path, tasks)
    q.claim(tasks, "A")
    q.accept(tasks[0], 1, receipt(tasks[0]))
    selected = parallel.trial_tasks(tasks, q.state)
    assert [current["episode_id"] for current in selected] == ["low", "medium", "high", "high2"]


def test_extension_freeze_does_not_edit_base_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "discover_gpu_bindings", lambda _: {"A": gpu_binding(0), "B": gpu_binding(2)})
    base = tmp_path / "source/maps.json"
    write(base, catalog())
    content = base.read_bytes()
    manifest = {"fingerprint": "frozen", "configs": {
        "val_seen": {"map_config_path": str(base), "report_profile": "paper_process"}}}
    recovery = {"execution_source": str(tmp_path / "recovery/source"), "recovery_revision": "audited",
                "recovery_policy": {"max_attempts": 6, "deadline_sec": 600}}
    write(tmp_path / "parallel/execution_overlay.json", recovery)
    overlay = parallel.prepare_extension(tmp_path, manifest)
    assert all(overlay[key] == value for key, value in recovery.items())
    parallel.verify_overlay(overlay, manifest)
    assert base.read_bytes() == content
    assert overlay["lanes"]["B"]["remote_gpu"] == 2
    assert overlay["forbidden_remote_gpus"] == [1]
    Path(overlay["lanes"]["B"]["map_config_path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        parallel.verify_overlay(overlay, manifest)


def test_gpu_binding_uses_uuid_mapping_instead_of_nvidia_index():
    binding = gpu_binding(2)
    binding["visible_vulkan_uuids"] = [gpu_binding(0)["uuid"], binding["uuid"]]
    binding["vulkan_index"] = 1
    config = parallel.lane_catalog(catalog(), "B", binding)
    assert config["maps"][0]["launch"]["args"][-1] == "-GraphicsAdapter=1"
    assert config["maps"][0]["launch"]["env"]["CUDA_VISIBLE_DEVICES"] == binding["uuid"]
    binding["vulkan_index"] = 0
    with pytest.raises(ValueError, match="binding"):
        parallel.lane_catalog(catalog(), "B", binding)


def test_gpu_guard_rejects_gpu_one_and_preserves_allowed_display_context():
    assert not parallel.gpu_violation({"owned": [{"lane": "B", "observed_gpus": [0, 2]}]})
    assert parallel.gpu_violation({"owned": [{"lane": "B", "observed_gpus": [0, 1]}]})
    with pytest.raises(ValueError, match="binding"):
        parallel.validate_gpu_binding(gpu_binding(1), 1)


def test_watchdog_deadline_is_independent_of_recent_progress():
    request = {"started": 0, "progress_at": 7199}
    assert parallel.watchdog_reason(request, 7201, 7200, 900) == "episode_deadline"
    request["progress_at"] = 0
    assert parallel.watchdog_reason(request, 901, 7200, 900) == "no_progress"
    assert parallel.watchdog_reason(request, 100, 7200, 900) is None


def test_pid_reuse_never_kills_another_process(monkeypatch):
    killed = []
    process = SimpleNamespace(create_time=lambda: 3, cmdline=lambda: ["other"],
                              kill=lambda: killed.append(True), wait=lambda **_: None)
    monkeypatch.setattr(parallel.psutil, "Process", lambda _: process)
    parallel.kill_owned({"pid": 10, "created": 2, "command": ["ours"]})
    assert not killed
    parallel.kill_owned({"pid": 10, "created": 3, "command": ["other"]})
    assert killed == [True]


def scored_trial(tmp_path):
    tasks = [task("one"), task("two")]
    q = queue(tmp_path, tasks)
    for lane, current in zip(("A", "B"), tasks):
        q.claim(tasks, lane)
        q.accept(current, 1, receipt(current))
        directory = parallel._attempt_dir(tmp_path, current, 1)
        page = directory / "evaluation/cases" / current["episode_id"] / "review.html"
        page.parent.mkdir(parents=True)
        (page.parent / "image.png").write_bytes(b"image")
        page.write_text('<html><img src="image.png"><pre>full tool results</pre></html>', encoding="utf-8")
        run = directory / "runs/run"
        (run / "memory").mkdir(parents=True)
        (run / "memory/agent_transcript.jsonl").write_text(json.dumps({
            "kind": "user_instruction", "payload": {"instruction": current["episode_id"]}}) + "\n", encoding="utf-8")
        q.state["tasks"][q.key(current)]["result"]["run_dir"] = str(run)
        write(directory / "thread_evidence.json", {"episode_id": current["episode_id"],
            "instruction_sha256": hashlib.sha256(current["episode_id"].encode()).hexdigest()})
    snapshots = [{"host_pid": 10, "services": [{"service": 11, "grounding_model": 12, "segmentation_model": 13}]}] * 2
    gpus = [{"owned": [{"lane": "A", "observed_gpus": [0]}, {"lane": "B", "observed_gpus": [2]}]}]
    return q, tasks, snapshots, gpus


def test_technical_gate_ignores_navigation_sr_checks_gpu_and_original_images(tmp_path):
    q, tasks, snapshots, gpus = scored_trial(tmp_path)
    result = parallel.validate_trial(tmp_path, q, tasks, snapshots, gpu_samples=gpus)
    assert result["passed"] and result["image_references"] == 2
    assert result["navigation_sr_is_gate"] is False
    gpus[0]["owned"][1]["observed_gpus"] = [1]
    result = parallel.validate_trial(tmp_path, q, tasks, snapshots, gpu_samples=gpus)
    assert not result["passed"] and "forbidden_remote_gpu" in result["errors"]


@pytest.mark.parametrize("fault", ["model_copy", "model_change", "foreign_image", "base64", "cleanup"])
def test_trial_rejects_isolation_and_resource_failures(tmp_path, fault):
    q, tasks, snapshots, gpus = scored_trial(tmp_path)
    snapshots = json.loads(json.dumps(snapshots))
    if fault == "model_copy":
        snapshots[-1]["services"].append(dict(snapshots[-1]["services"][0]))
    elif fault == "model_change":
        snapshots[-1]["services"][0]["grounding_model"] = 999
    elif fault in {"foreign_image", "base64"}:
        page = parallel._attempt_dir(tmp_path, tasks[0], 1) / "evaluation/cases/one/review.html"
        page.write_text('<img src="data:image/png;base64,AA">' if fault == "base64"
                        else '<img src="../../../../other.png">', encoding="utf-8")
    else:
        q.state["tasks"]["val_seen/one"]["attempts"][-1]["cleanup_verified"] = False
    assert not parallel.validate_trial(tmp_path, q, tasks, snapshots, gpu_samples=gpus)["passed"]


def test_manual_pause_interrupts_service_backoff_without_provider_request(tmp_path, monkeypatch):
    q = queue(tmp_path, [task("one")])
    (tmp_path / "PAUSE").write_text("pause", encoding="utf-8")
    monkeypatch.setattr(parallel, "provider_probe", lambda *a, **k: pytest.fail("must not request"))
    assert not parallel.service_recovery(tmp_path, {}, q)
    assert q.state["reason"] == "manual_interrupt"


def test_service_outage_uses_bounded_backoff_and_keeps_pending(tmp_path, monkeypatch):
    current = task("one")
    q = queue(tmp_path, [current])
    clock, probes = [0], []
    monkeypatch.setattr(parallel.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(parallel.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(parallel, "provider_probe", lambda *a, **k: probes.append(clock[0]) or False)
    manifest = {"configs": {"val_seen": {"env_path": "unused"}}}
    assert not parallel.service_recovery(tmp_path, manifest, q)
    assert probes == [30, 150, 450]
    assert q.state["reason"] == "provider_unavailable"
    assert q.state["tasks"]["val_seen/one"]["status"] == "pending"


def test_completed_boundary_never_arms_for_live_or_partial_provider_child(tmp_path):
    current = task("one")
    q = queue(tmp_path, [current])
    q.claim([current], "A")
    directory = parallel._attempt_dir(tmp_path, current, 1)
    write(directory / "process.json", {"status": "running"})
    write(directory / "result.json", receipt(current)["result"])
    assert parallel.completed_boundary(tmp_path, q.state) is None
    write(directory / "process.json", {"exit_code": 0})
    assert parallel.completed_boundary(tmp_path, q.state)["task"] == current
    write(directory / "result.json", receipt(current, provider=True)["result"])
    assert parallel.completed_boundary(tmp_path, q.state) is None


@pytest.mark.parametrize("terminal_provider_errors", [False, True])
def test_report_failure_keeps_score_and_depth_without_repeating_flight(tmp_path, monkeypatch, terminal_provider_errors):
    current = task("one")
    directory = parallel._attempt_dir(tmp_path, current, 1)
    depth = directory / "runs/run/artifacts/depth/data.npy"
    depth.parent.mkdir(parents=True)
    depth.write_bytes(b"depth")
    write(directory / "result.json", {**receipt(current)["result"], "infrastructure_error": None})
    proof = [{"remote_stop": {"cleanup_verified": True}, "endpoint": {"status": "unreachable"}}]
    monkeypatch.setattr(parallel, "cleanup_owned", lambda *a, **k: proof)
    monkeypatch.setattr(parallel, "write_process_review", lambda *a, **k: (_ for _ in ()).throw(ValueError("report")))
    manifest = {"terminal_provider_errors": terminal_provider_errors,
                "configs": {"val_seen": {"map_config_path": "unused", "report_profile": "paper_process"}}}
    done = parallel.finish_attempt(tmp_path, manifest, current, {"number": 1}, {"descendants": []})
    assert done["valid_score"] and done["cleanup_verified"] and depth.exists()
    assert done["result"]["report_error"]["error"] == "report"
    q = queue(tmp_path, [current])
    q.claim([current], "A")
    assert q.accept(current, 1, done) == "scored"
    assert q.claim([current], "B") is None


@pytest.mark.parametrize('launched', [False, True])
def test_startup_cleanup_selects_proof_from_persisted_launch_spec(tmp_path, monkeypatch, launched):
    current = task('startup')
    directory = parallel._attempt_dir(tmp_path, current, 1)
    scene = directory / 'runs/run/scenes/scene_unique'
    scene.mkdir(parents=True)
    if launched:
        write(scene / 'remote-scene-spec.json', {})
    write(directory / 'result.json', {'episode_id': current['episode_id'], 'infrastructure_error': {
        'stage': 'scene_setup', 'error': {'error_type': 'TimeoutExpired'}}})
    monkeypatch.setattr(parallel, 'cleanup_owned', lambda *args: [])
    package = SimpleNamespace(map_id='map')
    monkeypatch.setattr(parallel, 'load_map_config', lambda *args: ([package], None))
    calls = []
    def proof(kind):
        def verify(*args, **kwargs):
            calls.append((kind, kwargs))
            return {'verified': True, 'kind': kind}
        return verify
    monkeypatch.setattr(parallel, '_extension', lambda *args: SimpleNamespace(
        verify_failed_launch=proof('launched'), verify_unlaunched_scene=proof('unlaunched')))
    manifest = {'configs': {'val_seen': {'map_config_path': 'unused', 'scene_map': {'2': 'map'}}}}
    done = parallel.finish_attempt(tmp_path, manifest, current, {'number': 1, 'local_port': 35002},
                                   {'descendants': []})
    assert done['cleanup_verified'] and not done['valid_score']
    expected = {'completed_attempt': True, 'local_endpoint': ('127.0.0.1', 35002)}
    if not launched:
        expected['recheck_cleanup'] = True
    assert calls == [('launched' if launched else 'unlaunched', expected)]


def test_offline_report_repair_never_claims_a_scored_task(tmp_path, monkeypatch):
    current = task("one")
    q = queue(tmp_path, [current])
    q.claim([current], "A")
    done = receipt(current)
    directory = parallel._attempt_dir(tmp_path, current, 1)
    done["result"].update(run_dir=str(directory / "runs/run"), report_error={"error": "initial"})
    q.accept(current, 1, done)
    calls = []
    monkeypatch.setattr(parallel, "write_process_review",
                        lambda *a, **k: calls.append((a, k)) or {"html": "review.html"})
    manifest = {"configs": {"val_seen": {"report_profile": "paper_process"}}}
    parallel.repair_pending_reports(tmp_path, manifest, q)
    assert len(calls) == 1
    assert "report_error" not in q.state["tasks"]["val_seen/one"]["result"]
    assert q.claim([current], "B") is None


def test_finished_threads_have_independent_configs_and_result_directories(tmp_path, monkeypatch):
    calls = []
    manifest = {"configs": {"val_seen": {
        "dataset_root": "unused", "split": "val_seen", "map_config_path": "unused",
        "scene_map": {"2": "map"}, "report_profile": "paper_process", "execution_mode": "managed",
        "enable_action_planner_debug": False}}}
    class Runner:
        def __init__(self, config):
            self.config = config
            calls.append(config)
        def prepare_evaluation(self, **_):
            episode = self.config.episode_ids[0]
            return SimpleNamespace(cases=[SimpleNamespace(episode_id=episode, instruction=episode)],
                selection=SimpleNamespace(assignment_by_episode={episode: None}), provenance=None)
        def run_case(self, case, directory, **_):
            return {"episode_id": case.episode_id, "metrics": {"success": False}}
    from evaluation.benchmarks.aerial_vln import campaign_adapter
    monkeypatch.setattr(campaign_adapter, "AerialVlnBenchmarkRunner", Runner)
    requests = [{"task": task(identity), "attempt": {"number": 1}, "lane": lane,
                 "map_config_path": lane + ".json"} for lane, identity in (("A", "one"), ("B", "two"))]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda request: parallel.execute_thread(tmp_path, manifest, request, {}), requests))
    assert {result["episode_id"] for result in results} == {"one", "two"}
    assert len({config.run_root for config in calls}) == 2
    assert {config.map_config_path for config in calls} == {"A.json", "B.json"}
    assert all(not config.generate_html_review and config.report_profile == "paper_process" for config in calls)


def test_trial_rejects_history_from_another_instruction(tmp_path):
    q, tasks, snapshots, gpus = scored_trial(tmp_path)
    path = Path(q.state["tasks"]["val_seen/one"]["result"]["run_dir"]) / "memory/agent_transcript.jsonl"
    path.write_text(json.dumps({"kind": "user_instruction", "payload": {"instruction": "other"}}), encoding="utf-8")
    check = parallel.validate_trial(tmp_path, q, tasks, snapshots, gpu_samples=gpus)
    assert not check["passed"] and "transcript_instruction:val_seen/one" in check["errors"]


def test_coverage_check_preserves_unseen_and_rejects_missing_tasks(tmp_path):
    tasks = [task("one"), task("unseen", split="val_unseen")]
    q = queue(tmp_path, tasks)
    manifest = {"tasks": tasks, "counts": {"val_seen": 1, "val_unseen": 1}}
    parallel.coverage_check(tmp_path, manifest, q)
    assert q.state["tasks"]["val_unseen/unseen"]["status"] == "pending"
    del q.state["tasks"]["val_unseen/unseen"]
    with pytest.raises(ValueError, match="coverage"):
        parallel.coverage_check(tmp_path, manifest, q)
