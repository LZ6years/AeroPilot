"""Deterministic campaign recovery and lightweight evidence contract tests."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.aerial_vln.recovery import RecoveryQueue
from evaluation.benchmarks.core.executor import _error_payload
from evaluation.benchmarks.core.process_review import write_process_review
from evaluation.benchmarks.core.supervision import cleanup_verified
from src.llm import ProviderRequestError
from src.memory.stores.event_store import MemoryEventWriter
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.task.journal import RuntimeJournal


def task(identity):
    return {"split": "val_seen", "episode_id": identity}


def receipt(*, provider=False, retryable=True, cleanup=True, success=False, report_error=False):
    return {"cleanup_verified": cleanup, "valid_score": not provider,
            "result": {"episode_id": "case", "status": "ok", "metrics": {"success": success},
                       "report_error": {"error": "renderer failed"} if report_error else None,
                       "infrastructure_error": {"stage": "provider", "error": {
                           "retryable": retryable, "category": "tls", "attempts": 9,
                           "exhaustion_reason": "retry_count"}} if provider else None}}


def test_transient_provider_defers_episode_and_scores_once_after_other_task(tmp_path):
    tasks = [task("one"), task("two")]
    queue = RecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    calls = []

    def execute(current, attempt):
        calls.append((current["episode_id"], attempt["number"]))
        return receipt(provider=current["episode_id"] == "one" and attempt["number"] == 1)

    assert queue.run(tasks, execute, lambda: True)
    assert calls == [("one", 1), ("two", 1), ("one", 2)]
    assert len(queue.state["tasks"]["val_seen/one"]["attempts"]) == 2
    assert all(item["status"] == "scored" for item in queue.state["tasks"].values())
    assert queue.run(tasks, lambda *args: pytest.fail("scored flight repeated"), lambda: True)


def test_persistent_provider_outage_pauses_remaining_tasks(tmp_path):
    tasks = [task(str(index)) for index in range(5)]
    sleeps = []
    queue = RecoveryQueue(tmp_path / "progress.json", tasks, "frozen", sleep=sleeps.append)
    assert not queue.run(tasks, lambda *args: receipt(provider=True), lambda: False)
    assert sleeps == [30, 120, 300]
    assert queue.state["reason"] == "provider_unavailable"
    assert queue.state["tasks"]["val_seen/3"]["status"] == "pending"
    assert queue.state["tasks"]["val_seen/0"]["status"] == "retry_pending"
    assert all(item["result"] is None for item in queue.state["tasks"].values())


def test_service_check_recovers_then_continues_original_queue(tmp_path):
    tasks = [task(str(index)) for index in range(4)]
    queue = RecoveryQueue(tmp_path / "progress.json", tasks, "frozen", sleep=lambda _: None)
    probes = []
    def probe():
        probes.append(True)
        return True
    def execute(current, attempt):
        return receipt(provider=current["episode_id"] != "3" and attempt["number"] == 1)
    assert queue.run(tasks, execute, probe)
    assert len(probes) == 1


def test_episode_attempt_limit_keeps_no_score_for_tls_trajectory(tmp_path):
    queue = RecoveryQueue(tmp_path / "progress.json", [task("one")], "frozen", sleep=lambda _: None)
    assert not queue.run([task("one")], lambda *args: receipt(provider=True), lambda: True)
    item = queue.state["tasks"]["val_seen/one"]
    assert item["status"] == "retry_exhausted" and len(item["attempts"]) == 3
    assert item["result"] is None


@pytest.mark.parametrize("failure", ["cleanup", "auth", "interrupt"])
def test_pause_preserves_untouched_queue(tmp_path, failure):
    tasks = [task("one"), task("two")]
    queue = RecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    if failure == "interrupt":
        def execute(*args):
            raise KeyboardInterrupt()
    else:
        execute = lambda *args: receipt(cleanup=failure != "cleanup", provider=failure == "auth", retryable=False)
    assert not queue.run(tasks, execute, lambda: True)
    assert queue.state["status"] == "paused"
    assert queue.state["tasks"]["val_seen/two"]["status"] == "pending"
    restored = RecoveryQueue(queue.path, tasks, "frozen")
    assert restored.state == queue.state
    if failure == "interrupt":
        assert not restored.recover_interrupted(lambda *args: False)
        assert restored.recover_interrupted(lambda *args: True)
        assert restored.state["tasks"]["val_seen/one"]["status"] == "retry_pending"


def test_report_failure_preserves_score_without_repeat_flight(tmp_path):
    queue = RecoveryQueue(tmp_path / "progress.json", [task("one")], "frozen")
    assert queue.run([task("one")], lambda *args: receipt(report_error=True), lambda: True)
    assert queue.state["tasks"]["val_seen/one"]["result"]["report_error"]
    assert queue.run([task("one")], lambda *args: pytest.fail("report repair must not fly"), lambda: True)


def test_structured_provider_error_survives_executor_wrapper():
    error = ProviderRequestError("TLS failed", category="tls", attempts=9,
        retry_exhausted=True, retryable=True, status_code=None, exhaustion_reason="deadline", elapsed_seconds=600)
    wrapper = RuntimeError("wrapped")
    wrapper.original_error = error
    assert _error_payload(wrapper) == {"error_type": "ProviderRequestError", "error": "TLS failed", **error.to_dict()}


def test_cleanup_requires_group_and_endpoint_proof():
    assert not cleanup_verified([])
    assert not cleanup_verified([{"remote_stop": {"cleanup_verified": True}, "endpoint": {"status": "listening"}}])
    assert cleanup_verified([{"remote_stop": {"cleanup_verified": True}, "endpoint": {"status": "unreachable"}}])


def test_process_logging_keeps_trace_without_request_or_image_copies(tmp_path):
    logger = AsyncRunLogger(tmp_path, capture_provider_debug=False)
    journal = RuntimeJournal(logger)
    session = SimpleNamespace(transcript=SimpleNamespace(
        instruction_id="instruction", latest_observation_id=lambda: "observation_1"))
    journal.capture_provider_exchange(SimpleNamespace(last_exchange={
        "request": {"messages": ["must not persist"]},
        "transport": {"attempt_count": 3, "recovered": True},
        "response": {"usage": {"total_tokens": 10}}}), session, "decision_1")
    logger.debug_event("observation", {"frames": {"rgb": {"base64": "duplicate", "path": "original.png"}}})
    logger.close()
    assert not (tmp_path / "debug/provider_media").exists()
    assert not (tmp_path / "debug/tool_loop").exists()
    data = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "must not persist" not in data and '"attempt_count": 3' in data
    assert "duplicate" not in (tmp_path / "debug/debug_events.jsonl").read_text(encoding="utf-8")


def test_media_metadata_deduplicates_only_persisted_bytes_without_mutation(tmp_path):
    image = tmp_path / "rgb.png"
    image.write_bytes(b"original")
    frame = {"artifact_path": str(image), "base64": base64.b64encode(b"original").decode()}
    writer = MemoryEventWriter(tmp_path, persist_inline_media=False)
    writer.write_frame("obs", "rgb", frame)
    writer.write_frame("obs2", "rgb", {"artifact_path": "missing.png", "base64": "retain"})
    records = [json.loads(line) for line in (tmp_path / "memory/frames.jsonl").read_text().splitlines()]
    assert "base64" not in records[0]["metadata"] and "sha256" in records[0]["metadata"]
    assert records[1]["metadata"]["base64"] == "retain"
    assert frame["base64"]


def test_process_report_uses_all_original_images_and_transcript_without_debug(tmp_path):
    run = tmp_path / "runs/run_1"
    rgb = run / "artifacts/rgb"
    rgb.mkdir(parents=True)
    frames = []
    for index in range(3):
        path = rgb / f"{index}.png"
        path.write_bytes(b"original")
        frames.append({"modality": "rgb", "observation_id": f"obs{index}",
                       "metadata": {"artifact_path": str(path)}})
    (run / "memory").mkdir()
    (run / "memory/frames.jsonl").write_text("\n".join(json.dumps(row) for row in frames))
    (run / "memory/agent_transcript.jsonl").write_text(json.dumps({"kind": "assistant_turn", "payload": {
        "public_intent": "Move toward the building", "tool_calls": [{"tool_name": "uav.move", "args": {"vx": 1}}]}}))
    case = tmp_path / "evaluation/cases/one"
    case.mkdir(parents=True)
    result = {"episode_id": "one", "dataset_split": "val_unseen", "run_dir": str(run), "metrics": {"success": False}}
    report = write_process_review(case, result, "Find the building", profile="simple_process")
    page = Path(report["path"]).read_text()
    assert report["image_count"] == 3 and "Move toward the building" in page
    assert "data:image" not in page and not (case / "review_assets/frames").exists()
    assert len(list(run.rglob("*.png"))) == 3
