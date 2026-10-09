"""Diagnostic hooks must preserve calls, isolate threads and survive failures."""

from contextlib import ExitStack
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tests.manual.task_profile import TaskProfile, install_profile, summarize
from tests.manual.airsim_task_profile_probe import (
    ProfileRunner, can_continue, pair_cases, partial_case, verify_frozen,
)


def test_nested_accounting_and_exception_preserved(tmp_path):
    tick = [0.]
    profile = TaskProfile(tmp_path / "time.jsonl", clock=lambda: tick[0])
    failure = RuntimeError("do not record my secret body")
    with profile.span("root"):
        tick[0] = 2
        with pytest.raises(RuntimeError) as caught, profile.span("child"):
            tick[0] = 6
            raise failure
        tick[0] = 10
    assert caught.value is failure
    result = summarize(profile.path)
    stages = next(iter(result["threads"].values()))
    assert stages["root"]["inclusive_sec"] == 10
    assert stages["root"]["exclusive_sec"] == 6
    assert stages["child"]["exclusive_sec"] == 4
    assert stages["child"]["errors"] == 1
    assert "secret" not in profile.path.read_text()


def test_worker_time_is_separate_and_native_exit_span_is_missing(tmp_path):
    profile = TaskProfile(tmp_path / "time.jsonl")
    def worker():
        with profile.span("disk"):
            pass
    with profile.span("main"):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    profile.emit({"event": "begin", "id": 100, "parent": None, "thread": 999, "name": "crashed", "time": 1})
    with profile.path.open("a") as stream:
        stream.write('{"partial":')
    report = summarize(profile.path)
    assert len(report["threads"]) == 2
    assert next(row for row in report["spans"] if row["name"] == "disk")["parent"] is None
    assert report["incomplete"][0]["name"] == "crashed"
    assert report["malformed_lines"] == 1
    assert not summarize(tmp_path / "missing")["available"]


def test_descriptor_and_hooks_restore_even_on_failure(tmp_path):
    profile = TaskProfile(tmp_path / "time.jsonl")
    class Fake:
        @staticmethod
        def method(value):
            return value
    marker = object()
    with ExitStack() as stack:
        profile.wrap(stack, Fake, "method", "static")
        assert Fake.method(marker) is marker and Fake().method(marker) is marker
    from src.tools.dispatcher import ToolDispatcher
    original = ToolDispatcher.call
    with pytest.raises(ValueError), install_profile(profile):
        assert ToolDispatcher.call is not original
        raise ValueError()
    assert ToolDispatcher.call is original


def test_reuse_changes_only_experimental_stabilizer(tmp_path, monkeypatch):
    from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner
    from src.runtime.configs.stabilization_config import UavStabilizationConfig
    default = UavStabilizationConfig()
    monkeypatch.setattr(TravelUavBenchmarkRunner, "_make_runtime",
                        lambda *a: SimpleNamespace(stabilizer=SimpleNamespace(config=default)))
    runner = ProfileRunner.__new__(ProfileRunner)
    runner.reuse = True
    from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
    runner.read_only_execution_config = ReadOnlyExecutionConfig()
    runtime = runner._make_runtime(tmp_path, None)
    assert runtime.stabilizer.config.reuse_verified_post_action_state
    expected = default.to_dict() | {"reuse_verified_post_action_state": True}
    assert runtime.stabilizer.config.to_dict() == expected
    assert not default.reuse_verified_post_action_state


def test_call_context_does_not_leak_after_synchronous_batch(tmp_path):
    from src.runtime.control.read_only_executor import ReadOnlyBatchExecutor, ReadJob
    from src.tools.schemas import ReadExecutionHints
    from src.env_client import ActionResult
    profile = TaskProfile(tmp_path / "timing.jsonl")
    with install_profile(profile):
        job = ReadJob("read1", "test.read", ReadExecutionHints(), lambda: ActionResult(True, "test.read", "ok"))
        ReadOnlyBatchExecutor().execute([job])
        assert profile.call_id.get() is None
    spans = summarize(profile.path)["spans"]
    assert next(s for s in spans if s["name"] == "runtime.read_job")["tool_call_id"] == "read1"


def test_pair_identity_and_pause_gate():
    original = {"benchmark": "travel", "scene": "ModernCityMap", "case_id": "same"}
    cases = pair_cases({"cases": [{}] * 12 + [original]})
    assert [c["reuse"] for c in cases] == [False, True]
    assert all(c["case_id"] == "same" for c in cases) and "reuse" not in original
    concurrent = pair_cases({"cases": [{}] * 12 + [original]}, "read-only-concurrency")
    assert [c["read_workers"] for c in concurrent] == [1, 4]
    assert not any(c["reuse"] for c in concurrent)
    with pytest.raises(ValueError):
        pair_cases({"cases": [{}] * 12 + [{**original, "scene": "different"}]})
    status = {"exit_code": 0, "cleanup": {"verified": True}}
    assert can_continue(status, {"status": "ok", "completed": False})
    assert not can_continue(status, {})
    assert not can_continue(status, {"infrastructure_error": {"type": "SceneRecoveryRequired"}})
    for changed in ({"exit_code": -1}, {"cleanup": {"verified": False}}, {"review_error": "broken image"}):
        assert not can_continue(status | changed, {"status": "ok"})


def test_offline_gate_requires_all_conditions_and_valid_device_counts(tmp_path):
    import json
    from tests.manual.airsim_task_profile_probe import offline_ready
    rows = [{"sample": s, "workers": w, "exit_code": 0, "completed_batches": 5, "started_calls": 10,
             "incomplete_spans": 0, "peak_active_device": {"cpu": 1}, "remaining_active_device": {"cpu": 0}}
            for s in range(1, 5) for w in (1, 2, 4)]
    path = tmp_path / "summary.json"
    def save(state="passed"):
        path.write_text(json.dumps({"state": state, "error": None, "conditions": rows}))
    save()
    assert offline_ready(tmp_path)
    save("paused")
    assert not offline_ready(tmp_path)
    rows[0]["peak_active_device"] = {"cpu": 2}
    save()
    assert not offline_ready(tmp_path)
    rows.pop()
    save()
    assert not offline_ready(tmp_path)


def test_offline_parent_exit_blocks_scene_start(tmp_path, monkeypatch):
    import json
    import psutil
    from tests.manual.airsim_task_profile_probe import wait_offline
    (tmp_path / "parent.json").write_text(json.dumps({"pid": 99999, "created": 0}))
    (tmp_path / "summary.json").write_text(json.dumps({"state": "running"}))
    def missing(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil, "Process", missing)
    with pytest.raises(RuntimeError, match="without passing"):
        wait_offline(tmp_path, tmp_path / "gate.json")


def test_existing_process_wait_does_not_follow_reused_pid(tmp_path, monkeypatch):
    import psutil
    from tests.manual.airsim_task_profile_probe import wait_existing_process
    responses = iter([SimpleNamespace(create_time=lambda: 1), SimpleNamespace(create_time=lambda: 2)])
    monkeypatch.setattr(psutil, "Process", lambda pid: next(responses))
    wait_existing_process(123, tmp_path / "gate.json")


def test_frozen_source_tamper_blocks_next_task(tmp_path, monkeypatch):
    import tests.manual.airsim_task_profile_probe as probe
    path = tmp_path / "code.py"
    path.write_text("original")
    manifest = {"source_snapshot": str(tmp_path), "snapshot_hashes": {path.name: probe.sha(path)},
                "env_path": "not_read", "model_config": {"model_name": "unchanged"}}
    monkeypatch.setattr(probe, "load_model_config", lambda _: SimpleNamespace(redacted_summary=lambda: manifest["model_config"]))
    verify_frozen(manifest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="Frozen executable"):
        verify_frozen(manifest)


def test_early_exit_exports_no_score_both_pages(tmp_path):
    from tests.manual.export_benchmark_review import export_case, validate_bundle
    from tests.manual.airsim_task_profile_probe import refresh_index
    source = partial_case(tmp_path, {"case_id": "example"})
    review = tmp_path / "review"
    review.mkdir()
    refresh_index(tmp_path, review, [], state="paused")
    receipt = export_case(source, review, "01")
    assert receipt["incomplete"]
    assert "无最终评分" in (review / receipt["review"]).read_text(encoding="utf-8")
    assert (review / receipt["audit"]).is_file()
    assert validate_bundle(review)["max_path_length"] <= 240
