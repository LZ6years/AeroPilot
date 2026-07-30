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
