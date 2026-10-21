"""Closed-attempt archives preserve public evidence and exact diagnostic bytes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from evaluation.benchmarks.core.campaign import artifacts, retention
from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue
from evaluation.benchmarks.core.process_review import write_process_review


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def closed_attempt(root):
    run = root / "runs/test"
    put(root / "cleanup.json", {"verified": True})
    put(root / "process.json", {"status": "finished", "pid": 1})
    put(run / "debug/context_checkpoints/checkpoint_0001.json", {"observations": ["old", "current"] * 100})
    put(run / "contexts/place_memory/task/map_v1.json", {"places": ["place"] * 100})
    put(run / "contexts/place_memory/task/graph.json", {"places": ["place"] * 100})
    put(run / "debug/model_call_metrics/task/decision_0001.json", {
        "model_rounds": [{"canonical_turn": {"public_intent": "接近目标", "tool_calls": [
            {"name": "measure_pixel_depth", "arguments": {"observation_id": "old", "pixel_x": 1, "pixel_y": 2}}]},
            "request_media": [{"asset_ref": "artifacts/scene/rgb/frame.png", "sha256": "exact-wire-hash"}],
            "response": {"usage": {"total_tokens": 42}}}]})
    depth = run / "artifacts/scene/depth/frame.npy"
    depth.parent.mkdir(parents=True);depth.write_bytes(b"exact float32 depth bytes\0" * 400)
    rgb = run / "artifacts/scene/rgb/frame.png"
    rgb.parent.mkdir(parents=True);rgb.write_bytes(b"exact RGB bytes")
    records = [dict(kind="user_instruction", payload={"instruction": "Find the bicycle"}),
               dict(kind="assistant_turn", payload={"public_intent": "接近目标", "tool_calls": [{"name": "move"}]}),
               dict(kind="tool_result", payload={"ok": True, "source_observation_id": "old", "role": "HISTORY"})]
    memory = run / "memory";memory.mkdir()
    (memory / "agent_transcript.jsonl").write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in records), encoding="utf-8")
    (memory / "instructions.jsonl").write_text(json.dumps({"instruction": "Find the bicycle"}), encoding="utf-8")
    (memory / "frames.jsonl").write_text(json.dumps({"modality": "rgb", "observation_id": "old",
        "metadata": {"artifact_path": "artifacts/scene/rgb/frame.png"}}), encoding="utf-8")
    result = dict(run_dir=str(run), episode_id="test", dataset_split="test_seen", metrics={"success": False})
    case = root / "evaluation/cases/test";case.mkdir(parents=True)
    put(case / "audit.json", {"large_duplicate": records * 80})
    put(case / "case_result.json", result)
    write_process_review(case, result)
    return run, case, result


def test_archive_preserves_intents_calls_images_metrics_and_regenerated_html(tmp_path):
    run, case, result = closed_attempt(tmp_path)
    before = tree(tmp_path)
    record = artifacts.compact_attempt(tmp_path, profile=artifacts.PROFILE)
    assert record["archive_bytes"] < record["archived_original_bytes"]
    assert artifacts.validate_archive(tmp_path) == record
    archived = {v["path"] for v in record["members"]}
    assert "runs/test/artifacts/scene/depth/frame.npy" in archived
    assert "runs/test/contexts/place_memory/task/map_v1.json" in archived
    assert "runs/test/contexts/place_memory/task/graph.json" not in archived
    for name, data in before.items():
        if name not in archived:
            assert (tmp_path / name).read_bytes() == data
    write_process_review(case, result)
    assert (case / "review.html").read_bytes() == before["evaluation/cases/test/review.html"]
    assert "HISTORY" in (case / "review.html").read_text(encoding="utf-8")
    target = tmp_path.parent / (tmp_path.name + "-restore")
    artifacts.restore_archive(tmp_path, target)
    for name in archived:
        assert (target / name).read_bytes() == before[name]
    # A repeated cleanup does not create another archive or touch public data.
    snapshot = tree(tmp_path)
    assert artifacts.compact_attempt(tmp_path, profile=artifacts.PROFILE) == record
    assert tree(tmp_path) == snapshot


@pytest.mark.parametrize("fault", ["active", "cleanup", "full"])
def test_active_unverified_and_full_audit_attempts_cannot_be_archived(tmp_path, fault):
    closed_attempt(tmp_path)
    if fault == "active":put(tmp_path / "process.json", {"status": "running"})
    if fault == "cleanup":put(tmp_path / "cleanup.json", {"verified": False})
    before = tree(tmp_path)
    with pytest.raises(ValueError):
        artifacts.compact_attempt(tmp_path, profile="full" if fault == "full" else artifacts.PROFILE)
    assert tree(tmp_path) == before


def test_html_linked_snapshots_remain_directly_accessible(tmp_path):
    run, case, _ = closed_attempt(tmp_path)
    page = case / "review.html"
    page.write_text(page.read_text(encoding="utf-8") +
        '<a href="../../../runs/test/contexts/place_memory/task/map_v1.json">snapshot</a>', encoding="utf-8")
    record = artifacts.compact_attempt(tmp_path, profile=artifacts.PROFILE)
    assert (run / "contexts/place_memory/task/map_v1.json").is_file()
    assert not any(v["path"].endswith("map_v1.json") for v in record["members"])


def test_finished_owner_receipt_can_prove_an_older_stale_polling_heartbeat(tmp_path):
    closed_attempt(tmp_path)
    put(tmp_path / "process.json", {"status": "running"})
    put(tmp_path / "receipt.json", {"cleanup_verified": True, "process": {"status": "finished"}})
    assert artifacts.compact_attempt(tmp_path, profile=artifacts.PROFILE)["process_finished"]


def test_staging_failure_preserves_every_original(tmp_path, monkeypatch):
    closed_attempt(tmp_path)
    before = tree(tmp_path)
    monkeypatch.setattr(artifacts, "write_json", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        artifacts.compact_attempt(tmp_path, profile=artifacts.PROFILE)
    assert all((tmp_path / name).read_bytes() == data for name, data in before.items())


def test_corrupt_archive_is_rejected_without_overwriting_restoration(tmp_path):
    closed_attempt(tmp_path)
    artifacts.compact_attempt(tmp_path, profile=artifacts.PROFILE)
    (tmp_path / artifacts.ARCHIVE).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        artifacts.restore_archive(tmp_path, tmp_path.parent / (tmp_path.name + "-restore"))


def test_restoration_cannot_overwrite_changed_or_active_content(tmp_path):
    closed_attempt(tmp_path)
    record = artifacts.compact_attempt(tmp_path, profile=artifacts.PROFILE)
    with pytest.raises(ValueError, match="separate directory"):
        artifacts.restore_archive(tmp_path, tmp_path)
    target = tmp_path.parent / (tmp_path.name + "-restore")
    p = target / record["members"][0]["path"];p.parent.mkdir(parents=True);p.write_bytes(b"different")
    with pytest.raises(ValueError, match="overwrite"):
        artifacts.restore_archive(tmp_path, target)
    assert p.read_bytes() == b"different"


def test_new_profile_budget_uses_measured_seed_and_real_peak_without_old_profile_bias(tmp_path, monkeypatch):
    tasks = [dict(split="seen", episode_id=str(i), scene_id="map") for i in range(2)]
    q = ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    q.state["tasks"][q.key(tasks[0])]["status"] = "scored"
    q.state["tasks"][q.key(tasks[1])]["attempts"] = [dict(logging_profile="paper_process_metrics_v1", retained_bytes=500 * 1024**2)]
    policy = dict(full_task_keys=[q.key(tasks[0])], full_seed_bytes=1, compact_profile=artifacts.PROFILE,
                  compact_seed_bytes=6 * 1024**3, compact_peak_bytes=1024**3)
    retention.validate(policy, tasks)
    before = deepcopy(q.state)
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda _: SimpleNamespace(free=15 * 1024**3))
    result = retention.budget(tmp_path, tasks, q, policy, 3)
    assert result["required_bytes"] == 15 * 1024**3 and result["passed"]
    assert result["observed_cases"] == 0 and result["calibration_pending"]
    assert retention.profile(tasks[0], policy) == "full"
    assert retention.profile(tasks[1], policy) == artifacts.PROFILE
    assert q.state == before
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda _: SimpleNamespace(free=15 * 1024**3 - 1))
    assert not retention.budget(tmp_path, tasks, q, policy, 3)["passed"]
    for key in ("compact_seed_bytes", "compact_peak_bytes"):
        with pytest.raises(ValueError, match="measured"):
            retention.validate({**policy, key: 0}, tasks)


@pytest.mark.parametrize("mode", ["archive", "archive_failure", "full", "unverified"])
def test_controller_archiving_never_changes_a_valid_navigation_score(tmp_path, monkeypatch, mode):
    from evaluation.benchmarks.core.campaign import parallel
    task = dict(split="seen", episode_id="test", scene_id="map")
    directory = parallel._attempt_dir(tmp_path, task, 1)
    _, _, result = closed_attempt(directory)
    put(directory / "result.json", result)
    original_metrics = deepcopy(result["metrics"])
    proof = [{"remote_stop": {"cleanup_verified": mode != "unverified"},
              "endpoint": {"status": "unreachable"}}]
    monkeypatch.setattr(parallel, "cleanup_owned", lambda *a, **k: proof)
    monkeypatch.setattr(parallel, "cleanup_verified", lambda _: mode != "unverified")
    monkeypatch.setattr(parallel.compat, "attach_review", lambda *a: None)
    if mode == "archive_failure":
        monkeypatch.setattr(artifacts, "compact_attempt", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    if mode == "unverified":
        # Avoid the unrelated startup fallback in this focused storage test.
        monkeypatch.setattr(parallel, "_extension", lambda *a: SimpleNamespace())
    manifest = {"configs": {"seen": {"map_config_path": "unused", "report_profile": "paper_process"}}}
    attempt = dict(number=1, logging_profile="full" if mode == "full" else artifacts.PROFILE)
    done = parallel.finish_attempt(tmp_path, manifest, task, attempt, {"descendants": []})
    assert done["valid_score"] and done["result"]["metrics"] == original_metrics
    assert done["cleanup_verified"] is (mode != "unverified")
    if mode == "archive":
        assert done["artifact_archive"]["verified"] and (directory / artifacts.ARCHIVE).is_file()
    elif mode == "archive_failure":
        assert done["artifact_archive_error"]["error"] == "disk full"
    else:
        assert not (directory / artifacts.ARCHIVE).exists()


def test_polling_during_real_cleanup_cannot_overwrite_terminal_archive_proof(tmp_path, monkeypatch):
    """Reproduce the live race with a blocked archive and actual cleanup worker."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from evaluation.benchmarks.core.campaign import parallel
    task = dict(split="seen", episode_id="test", scene_id="map")
    directory = parallel._attempt_dir(tmp_path, task, 1)
    _, _, result = closed_attempt(directory)
    put(directory / "result.json", result)
    monkeypatch.setattr(parallel, "cleanup_owned", lambda *a, **k: [
        {"remote_stop": {"cleanup_verified": True}, "endpoint": {"status": "unreachable"}}])
    monkeypatch.setattr(parallel.compat, "attach_review", lambda *a: None)
    entered, release = Event(), Event()
    original = artifacts.compact_attempt

    def waiting_archive(*args, **kwargs):
        entered.set()
        assert release.wait(10), "Polling test failed to release the archive"
        return original(*args, **kwargs)

    monkeypatch.setattr(artifacts, "compact_attempt", waiting_archive)
    running = {"status": "running", "descendants": []}
    parallel.write_attempt_heartbeat(directory, running, cleanup_pending=False)
    assert parallel.read_json(directory / "process.json")["status"] == "running"
    manifest = {"configs": {"seen": {"map_config_path": "unused", "report_profile": "paper_process"}}}
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(parallel.finish_attempt, tmp_path, manifest, task,
                               dict(number=1, logging_profile=artifacts.PROFILE), running)
        try:
            assert entered.wait(10), "Cleanup worker did not reach archive"
            for _ in range(20):
                parallel.write_attempt_heartbeat(directory, running, cleanup_pending=True)
            assert parallel.read_json(directory / "process.json")["status"] == "finished"
        finally:
            release.set()
        done = future.result(timeout=10)
    assert done["valid_score"] and done["cleanup_verified"]
    assert done["artifact_archive_error"] is None and done["artifact_archive"]["verified"]
    artifacts.validate_archive(directory)
