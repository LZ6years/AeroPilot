"""OS handoff guard must not duplicate helpers or overwrite deliberate pauses."""
import json
from tests.manual import recovery_revision_handoff as handoff


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_live_helper_is_left_alone(tmp_path, monkeypatch):
    candidate = tmp_path/"revision/candidate_overlay.json"
    write(candidate.parent/"handoff_status.json", {"status": "draining", "process": {"pid": 123}})
    monkeypatch.setattr(handoff.control, "matching_process", lambda _: object())
    monkeypatch.setattr(handoff.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("duplicate helper")))
    assert handoff.ensure(tmp_path, candidate) == "helper_alive"


def test_manual_pause_and_failed_handoff_require_attention(tmp_path):
    candidate = tmp_path/"revision/candidate_overlay.json"
    write(candidate.parent/"handoff_status.json", {"status": "failed"})
    assert handoff.ensure(tmp_path, candidate) == "needs_attention"
    write(candidate.parent/"handoff_status.json", {"status": "draining"})
    (tmp_path/"PAUSE").touch()
    assert handoff.ensure(tmp_path, candidate) == "needs_attention"


def test_dead_owners_delegate_abandoned_leases_to_controller(tmp_path, monkeypatch):
    candidate = tmp_path/"revision/candidate_overlay.json"
    write(candidate, {"recovery_revision": "new"})
    write(candidate.parent/"handoff_status.json", {"status": "draining"})
    write(tmp_path/"manifest.json", {"fingerprint": "original"})
    calls = []
    monkeypatch.setattr(handoff, "activate", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert handoff.ensure(tmp_path, candidate) == "installed_after_owner_exit"
    assert calls[0][1] == {"allow_abandoned": True}
