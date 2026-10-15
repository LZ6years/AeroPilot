"""Cost ledger regressions for invalid output and abandoned background work."""

import json
import threading

import pytest

from src.agent.context.compression import (
    CompressionRequest,
    CompressionWindow,
    ModelHistoryCompactor,
)
from src.llm import ModelResponse
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.model_call_audit import ModelCallAuditWriter


def request():
    return CompressionRequest(
        "digest", "inspect", (CompressionWindow("w", 0, 1, ()),), (), {}, {}
    )


class Client:
    def __init__(self, content, finish_reason="stop", gate=None):
        self.content = content
        self.finish_reason = finish_reason
        self.gate = gate
        self.started = threading.Event()

    def request(self, value, **kwargs):
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(5)
        return ModelResponse({
            "usage": {"prompt_tokens": 123, "completion_tokens": 17},
            "choices": [{"finish_reason": self.finish_reason,
                         "message": {"content": self.content}}],
        })


@pytest.mark.parametrize("content,finish", [("invalid", "stop"), ("{}", "length")])
def test_invalid_response_still_records_billable_usage(content, finish):
    records = []
    compactor = ModelHistoryCompactor(
        Client(content, finish), summary_max_tokens=4096, audit_sink=records.append
    )
    with pytest.raises(ValueError):
        compactor.compact(request())
    assert [item["status"] for item in records] == ["started", "error"]
    assert records[0]["call_id"] == records[1]["call_id"]
    assert records[1]["usage"]["prompt_tokens"] == 123
    assert "content" not in records[1]


def test_transport_error_has_missing_usage_and_preserves_primary_error():
    class FailingClient:
        def request(self, value, **kwargs):
            raise TimeoutError("transport timeout")

    records = []
    compactor = ModelHistoryCompactor(
        FailingClient(), summary_max_tokens=4096, audit_sink=records.append
    )
    with pytest.raises(TimeoutError, match="transport timeout"):
        compactor.compact(request())
    assert records[-1]["usage"] is None
    assert records[-1]["error_type"] == "TimeoutError"


def test_background_ledger_finishes_after_async_logger_closes(tmp_path):
    gate = threading.Event()
    client = Client(json.dumps({
        "mission_progress": [], "decisions_and_outcomes": [], "failures_and_recovery": []
    }), gate=gate)
    logger = AsyncRunLogger(tmp_path)
    compactor = ModelHistoryCompactor(
        client, summary_max_tokens=4096, audit_sink=ModelCallAuditWriter(tmp_path)
    )
    failures = []

    def work():
        try:
            compactor.compact(request())
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    assert client.started.wait(5)
    logger.close()
    gate.set()
    worker.join(5)
    assert not worker.is_alive() and not failures
    directory = tmp_path / "debug/history_compaction_calls"
    finished = list(directory.glob("*.finished.json"))
    assert len(finished) == 1
    assert json.loads(finished[0].read_text(encoding="utf-8"))["usage"]["completion_tokens"] == 17
