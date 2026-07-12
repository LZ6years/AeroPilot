"""Shared pytest fixtures for repository tests."""

from __future__ import annotations

import pytest


RUNTIME_CONTROL_ENV_VARS = (
    "FLYAGENT_RUNTIME_CONTROL_ENABLED",
    "FLYAGENT_RUNTIME_CONTROL_ASK_USER_ENABLED",
    "FLYAGENT_RUNTIME_CONTROL_ASK_USER_EXPOSED",
    "FLYAGENT_RUNTIME_CONTROL_SEMANTIC_PLAN_UPDATE_ENABLED",
    "FLYAGENT_RUNTIME_CONTROL_SEMANTIC_PLAN_UPDATE_EXPOSED",
)


@pytest.fixture(autouse=True)
def clear_runtime_control_env(monkeypatch):
    """Keep runtime control tests independent from the developer shell."""

    for name in RUNTIME_CONTROL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
