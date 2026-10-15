"""Frozen hard cohort preparation and read-only native progress projection."""
import pytest

from tests.manual import aerial_vln_hard10_parallel_probe as hard
from tests.manual import aerial_vln_plan_context_probe as probe
from evaluation.benchmarks.aerial_vln.parallel_recovery import ParallelRecoveryQueue


def prepared(tmp_path):
    tasks = [{"split": "val_seen", "episode_id": cid} for cid in probe.read(hard.HARD_CASES)]
    manifest = {"tasks": tasks, "config": {"limit": 300}, "cohorts": [{"name": "hard10"}]}
    probe.write(tmp_path / "manifest.json", manifest)
    probe.write(tmp_path / "progress.json", {"cases": [{"task": task, "status": "pending", "attempts": []} for task in tasks]})
    return tasks


def test_native_projection_preserves_order_terminal_errors_and_source(tmp_path):
    tasks = prepared(tmp_path)
    manifest = hard.configure_parallel(tmp_path, tmp_path / "secondary")
    assert manifest["technical_trial"] == tasks[:4]
    assert manifest["configs"]["val_seen"] == manifest["config"]
    queue = ParallelRecoveryQueue(tmp_path / "progress.json", tasks, manifest["fingerprint"], terminal_provider_errors=True)
    task, attempt = queue.claim(tasks, "A")
    result = {"episode_id": task["episode_id"], "infrastructure_error": {"stage": "provider", "error": {"retryable": True}}, "metrics": {}}
    receipt = {"cleanup_verified": True, "valid_score": False, "result": result}
    assert queue.accept(task, attempt["number"], receipt) == "model_error"
    assert queue.terminal(queue.state["tasks"][queue.key(task)])
    assert queue.accept(task, attempt["number"], receipt) == "duplicate"
    assert queue.claim(tasks, "A")[0] == tasks[1]
    before = (tmp_path / "progress.json").read_bytes()
    projection = probe.read_experiment_progress(tmp_path)
    assert [row["task"] for row in projection["cases"]] == tasks
    assert projection["cases"][0]["status"] == "model_error"
    assert projection["cases"][0]["attempts"][0]["result"] == result
    assert (tmp_path / "progress.json").read_bytes() == before


def test_parallel_conversion_rejects_reordered_or_started_cases(tmp_path):
    tasks = prepared(tmp_path)
    manifest = probe.read(tmp_path / "manifest.json")
    manifest["tasks"] = tasks[::-1]
    probe.write(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="ordered"):
        hard.configure_parallel(tmp_path, tmp_path)
    prepared(tmp_path)
    state = probe.read(tmp_path / "progress.json")
    state["cases"][0]["status"] = "running"
    probe.write(tmp_path / "progress.json", state)
    with pytest.raises(ValueError, match="untouched"):
        hard.configure_parallel(tmp_path, tmp_path)


@pytest.mark.parametrize("status", ["paused", "seen_complete"])
def test_run_exit_distinguishes_complete_from_paused_and_requires_all_admissions(tmp_path, monkeypatch, status):
    monkeypatch.setattr(probe, 'guard', lambda _: {"plan_mode": "optional", "place_memory_admission_required": True})
    monkeypatch.setattr(probe, 'admission_passed', lambda *_: True)
    monkeypatch.setattr(hard, 'resource_check', lambda _: None)
    monkeypatch.setattr(probe, 'report', lambda _: None)
    launched = []
    monkeypatch.setattr(hard.subprocess, 'run', lambda *args, **kwargs: launched.append(args))
    probe.write(tmp_path / 'admission.json', {})
    for name in ('historical_request_replay.json', 'pixel_place_admission.json'):
        probe.write(tmp_path / name, {"passed": True})
    probe.write(tmp_path / 'place_memory_admission.json', {"status": "failed"})
    with pytest.raises(RuntimeError, match="Place memory admission"):
        hard.run(tmp_path)
    assert not launched
    probe.write(tmp_path / 'place_memory_admission.json', {"status": "passed"})
    probe.write(tmp_path / 'progress.json', {"status": status})
    if status == 'paused':
        with pytest.raises(RuntimeError, match="incomplete"):
            hard.run(tmp_path)
    else:
        hard.run(tmp_path)
    assert len(launched) == 1
