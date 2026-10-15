"""Offline regression for manual diagnostic tooling."""
import json
from evaluation.benchmarks.core.campaign import parallel


def test_handoff_keeps_elapsed_and_idle_time_in_deadlines(tmp_path, monkeypatch):
    from tests.manual import image_budget_order_handoff as handoff
    task = {"dataset": "aerial_vln", "image_budget": 10, "split": "seen", "episode_id": "one"}
    directory = parallel._attempt_dir(tmp_path, task, 1)
    directory.mkdir(parents=True)
    (directory / "process.json").write_text(json.dumps({"elapsed_seconds": 1800}), encoding="utf-8")
    monkeypatch.setattr(handoff.time, "monotonic", lambda: 10000)
    monkeypatch.setattr(handoff.time, "time", lambda: 20000)
    monkeypatch.setattr(handoff.control, "activity_time", lambda _: (20000 - 120) * 10**9)
    request = {"id": "one", "task": task, "attempt": {"number": 1}}
    limit = handoff.drain_deadlines(tmp_path, [(tmp_path, request)])["one"]
    assert limit["started"] == 8200 and limit["progress_at"] == 9880
    assert parallel.watchdog_reason(limit, 10002, 1801, 900) == "episode_deadline"
