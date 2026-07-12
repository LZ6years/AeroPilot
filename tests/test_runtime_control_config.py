"""Tests for runtime control availability configuration."""

from __future__ import annotations

import json

from src.runtime.configs.runtime_control_config import (
    RuntimeControlActionConfig,
    RuntimeControlConfig,
    load_runtime_control_config,
)


def test_runtime_control_config_defaults_expose_nothing():
    config = RuntimeControlConfig()

    assert config.enabled is False
    assert config.is_enabled("ask_user") is False
    assert config.is_exposed("ask_user") is False
    assert config.exposed_actions(["ask_user", "semantic_plan_update"]) == []


def test_runtime_control_config_loads_enabled_and_exposed_actions(tmp_path):
    path = tmp_path / "runtime_control_config.json"
    path.write_text(
        json.dumps(
            {
                "enabled": True,
                "actions": {
                    "ask_user": {"enabled": True, "exposed": False},
                    "semantic_plan_update": {"enabled": True, "exposed": True},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_runtime_control_config(path)

    assert config.is_enabled("ask_user") is True
    assert config.is_exposed("ask_user") is False
    assert config.is_exposed("semantic_plan_update") is True
    assert config.exposed_actions(["ask_user", "semantic_plan_update"]) == ["semantic_plan_update"]


def test_runtime_control_config_env_overrides(monkeypatch):
    monkeypatch.setenv("FLYAGENT_RUNTIME_CONTROL_ENABLED", "true")
    monkeypatch.setenv("FLYAGENT_RUNTIME_CONTROL_ASK_USER_ENABLED", "1")
    monkeypatch.setenv("FLYAGENT_RUNTIME_CONTROL_ASK_USER_EXPOSED", "yes")

    config = load_runtime_control_config("missing-runtime-control-config.json")

    assert config.enabled is True
    assert config.action_config("ask_user") == RuntimeControlActionConfig(enabled=True, exposed=True)
    assert config.is_exposed("ask_user") is True
    assert config.is_exposed("semantic_plan_update") is False
