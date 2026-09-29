"""Test asynchronous run logging and artifact writes."""

from __future__ import annotations

import json
import threading

from src.runtime.logging.async_run_logger import AsyncRunLogger


def test_async_run_logger_writes_system_events_and_artifacts(tmp_path):
    # Verify queued log, event, JSON, text, and byte writes flush on close.
    logger = AsyncRunLogger(tmp_path)

    logger.info("hello", step=1)
    logger.event("action_completed", {"ok": True})
    json_path = logger.write_json("artifacts/observations/step_0001.json", {"pose": {"x": 1}})
    text_path = logger.save_text("notes/readme.txt", "debug note")
    bytes_path = logger.save_bytes("artifacts/rgb/frame_000001.png", b"fake-png")
    logger.close()

    assert "hello" in (tmp_path / "debug" / "system.log").read_text(encoding="utf-8")
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(events[0])["event"] == "action_completed"
    assert json.loads(json_path.read_text(encoding="utf-8"))["pose"]["x"] == 1
    assert text_path.read_text(encoding="utf-8") == "debug note"
    assert bytes_path.read_bytes() == b"fake-png"


def test_async_run_logger_records_worker_write_errors(tmp_path):
    # Verify failed queued writes surface as artifact_failed events.
    logger = AsyncRunLogger(tmp_path)
    logger.save_text(".", "this cannot overwrite a directory")
    logger.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "debug" / "debug_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert events[0]["event"] == "artifact_failed"
    assert "error_type" in events[0]["payload"]


def test_async_run_logger_flushes_prior_writes_without_closing(tmp_path):
    logger = AsyncRunLogger(tmp_path)

    logger.event("before_boundary", {"index": 1})
    logger.flush()

    first_snapshot = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["event"] for item in first_snapshot] == ["before_boundary"]

    logger.event("after_boundary", {"index": 2})
    logger.close()
    final_snapshot = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["event"] for item in final_snapshot] == [
        "before_boundary",
        "after_boundary",
    ]


def test_async_run_logger_close_serializes_with_enqueue(tmp_path):
    logger = AsyncRunLogger(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original_put = logger._queue.put

    def blocking_put(item):
        if item is not None and not entered.is_set():
            entered.set()
            assert release.wait(5)
        original_put(item)

    logger._queue.put = blocking_put
    write_error = []

    def write():
        try:
            logger.save_text("notes/accepted.txt", "accepted before close")
        except Exception as exc:  # pragma: no cover - assertion reports detail
            write_error.append(exc)

    producer = threading.Thread(target=write)
    producer.start()
    assert entered.wait(5)
    closer = threading.Thread(target=logger.close)
    closer.start()
    release.set()
    producer.join(5)
    closer.join(5)

    assert not write_error
    assert not producer.is_alive() and not closer.is_alive()
    assert (tmp_path / "notes" / "accepted.txt").read_text(encoding="utf-8") == (
        "accepted before close"
    )
