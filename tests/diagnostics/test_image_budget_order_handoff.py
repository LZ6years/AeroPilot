"""Operator handoff checks protect exact live leases and terminal receipts."""
from pathlib import Path

import pytest

from tests.manual.image_budget_order_handoff import completed_requests, lease_requests
from evaluation.benchmarks.aerial_vln.parallel_campaign import write_json


def live_lease(tmp_path):
    session = tmp_path / "parallel/sessions/test"
    task = {"dataset": "aerial_vln", "image_budget": 30, "episode_id": "same"}
    attempt = {"number": 1, "lane": "B", "session": str(session), "status": "running"}
    request = {"id": "original-request", "task": task, "attempt": attempt, "lane": "B"}
    write_json(session / "commands/B.json", request)
    state = {"tasks": {"case": {"status": "running", "task": task, "attempts": [attempt]}}}
    return session, request, state


def test_handoff_preserves_running_lease_until_its_exact_terminal_receipt(tmp_path):
    session, request, state = live_lease(tmp_path)
    requests = lease_requests(tmp_path.resolve(), state)
    assert requests == [(session.resolve(), request)]
    assert completed_requests(requests) is False
    write_json(session / "results/original-request.json", {**request, "thread_finished": True})
    assert completed_requests(requests) is True


def test_handoff_rejects_a_command_or_receipt_for_another_budget(tmp_path):
    session, request, state = live_lease(tmp_path)
    changed = {**request, "task": {**request["task"], "image_budget": 20}}
    write_json(session / "commands/B.json", changed)
    with pytest.raises(ValueError, match="recorded lease"):
        lease_requests(tmp_path.resolve(), state)
    write_json(session / "results/original-request.json", {**changed, "thread_finished": True})
    with pytest.raises(ValueError, match="receipt"):
        completed_requests([(session, request)])


def test_handoff_rejects_an_unrelated_session_path(tmp_path):
    session, request, state = live_lease(tmp_path)
    state["tasks"]["case"]["attempts"][-1]["session"] = str(Path(tmp_path).parent / "other-campaign")
    with pytest.raises(ValueError, match="unrelated"):
        lease_requests(tmp_path.resolve(), state)
