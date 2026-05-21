"""Test asynchronous run logging and artifact writes."""

from __future__ import annotations

import json

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
