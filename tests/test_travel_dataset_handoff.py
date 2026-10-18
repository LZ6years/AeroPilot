"""The deployment handoff must not preempt Aerial or override operator stops."""
import pytest

from tests.manual import travel_dataset_handoff as handoff


def state(count=300, status="scored"):
    return {"tasks": {str(i): {"task": {"dataset": "aerial_vln"}, "status": status}
                      for i in range(count)}}


@pytest.mark.parametrize("count,status,ready", [(300, "scored", True), (299, "scored", False),
                                               (300, "retry_pending", False), (300, "running", False)])
def test_boundary(count, status, ready):
    assert handoff.aerial_finished(state(count, status)) is ready


@pytest.mark.parametrize("pause", [None, "User requested stop", "Waiting for another dataset"])
def test_preserves_manual_pause(tmp_path, monkeypatch, pause):
    directory = tmp_path / "handoff"
    directory.mkdir()
    handoff.write(tmp_path / "progress.json", state())
    monkeypatch.setattr(handoff, "controller", lambda overlay: {"matching_process": lambda owner: None})
    if pause is not None:
        (tmp_path / "PAUSE").write_text(pause, encoding="utf-8")
    assert handoff.tick(tmp_path, directory) == "preserving_other_pause"
    assert not (directory / "handoff_status.json").exists()
    if pause is not None:
        assert (tmp_path / "PAUSE").read_text(encoding="utf-8") == pause


def test_waits_for_last_aerial_and_owner(tmp_path, monkeypatch):
    directory = tmp_path / "handoff"
    directory.mkdir()
    handoff.write(tmp_path / "progress.json", state(299))
    monkeypatch.setattr(handoff, "controller", lambda overlay: {
        "matching_process": lambda owner: True, "utc_now": lambda: "now"})
    assert handoff.tick(tmp_path, directory) == "waiting_for_aerial"
    handoff.write(tmp_path / "progress.json", state())
    handoff.write(tmp_path / "dispatcher.json", {"pid": 123})
    assert handoff.tick(tmp_path, directory) == "waiting_for_owner_exit"
