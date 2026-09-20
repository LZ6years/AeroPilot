"""Test remote launcher safety checks independently of real processes."""

from __future__ import annotations

import json

import pytest

import src.sim_server.remote.worker as launcher


def test_stop_refuses_reused_pid_identity(tmp_path, monkeypatch):
    registry = tmp_path / "scene-registry.json"
    registry.write_text(
        json.dumps(
            {
                "pid": 123,
                "pgid": 123,
                "process_start_ticks": 456,
                "executable": "/data/AirVLN.sh",
                "status": "running",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "_process_start_ticks", lambda pid: 999)
    monkeypatch.setattr(launcher, "_identity_matches", lambda value: False)

    with pytest.raises(RuntimeError, match="Refusing to stop"):
        launcher.stop_scene(str(registry))


def test_stop_is_idempotent_when_registry_is_missing(tmp_path):
    result = launcher.stop_scene(str(tmp_path / "missing.json"))

    assert result == {"ok": True, "status": "not_found"}


def test_orphan_child_requires_scene_identity_and_is_waited_for(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"pid": 123, "pgid": 123,
        "process_start_ticks": 456, "executable": "/app/start.sh",
        "settings_path": "/scene_unique/settings.json", "status": "running"}))
    monkeypatch.setattr(launcher, "_process_start_ticks", lambda pid: 457 if pid == 124 else None)
    alive = [124]
    monkeypatch.setattr(launcher, "_live_group_members", lambda pgid: list(alive))
    monkeypatch.setattr(launcher, "_cmdline", lambda pid: "UE --settings /scene_unique/settings.json")
    signals = []
    def stop(pgid, sig):
        signals.append((pgid, sig))
        alive.clear()
    monkeypatch.setattr(launcher.os, "killpg", stop, raising=False)
    result = launcher.stop_scene(str(registry))
    assert result["cleanup_verified"]
    assert len(signals) == 1


def test_unknown_orphan_is_never_signalled(monkeypatch):
    monkeypatch.setattr(launcher, "_process_start_ticks", lambda pid: None)
    monkeypatch.setattr(launcher, "_live_group_members", lambda pgid: [124])
    monkeypatch.setattr(launcher, "_cmdline", lambda pid: "unrelated UE")
    with pytest.raises(RuntimeError, match="ownership"):
        launcher._verified_group({"pid": 123, "pgid": 123,
            "process_start_ticks": 1, "settings_path": "/owned/settings.json"})


def test_exit_cmdline_disappearing_does_not_invalidate_known_child(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"pid": 123, "pgid": 123,
        "process_start_ticks": 456, "executable": "/app/start.sh",
        "settings_path": "/scene/settings.json", "status": "running"}))
    alive = [123, 124]
    terminating = False
    monkeypatch.setattr(launcher, "_live_group_members", lambda pgid: list(alive))
    monkeypatch.setattr(launcher, "_process_start_ticks", lambda pid: {123: 456, 124: 457}.get(pid) if pid in alive else None)
    monkeypatch.setattr(launcher, "_cmdline", lambda pid: "" if terminating else "/app/start.sh --settings /scene/settings.json")
    def signal_group(pgid, sig):
        nonlocal terminating
        terminating = True
        alive.remove(123)
    monkeypatch.setattr(launcher.os, "killpg", signal_group, raising=False)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: alive.clear())
    assert launcher.stop_scene(str(registry))["cleanup_verified"] is True
