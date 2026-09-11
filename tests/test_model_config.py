"""Test model provider configuration loading."""

from __future__ import annotations

import pytest

from src.runtime.configs.model_config import ModelConfig
from src.runtime.configs.model_config import load_model_config


def test_model_config_defaults_to_model_mode_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_MODE", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_NAME", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_TIMEOUT_RETRIES", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_CONNECTION_RETRIES", raising=False)
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_STRUCTURE_PROMPT_PATH", raising=False)
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_CRITERIA_PROMPT_PATH", raising=False)
    monkeypatch.delenv("FLYAGENT_AGENT_MAX_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv(
        "FLYAGENT_AGENT_MAX_CONSECUTIVE_PROTOCOL_REJECTIONS", raising=False
    )

    config = load_model_config(tmp_path / ".env")

    assert config.task_parser_mode == "model"
    assert config.provider == "openai_compatible"
    assert config.api_key == ""
    assert config.provider_timeout_retries == 1
    assert config.provider_connection_retries == 2
    assert config.max_tool_rounds == 10
    assert config.max_consecutive_protocol_rejections == 6


def test_model_config_loads_dotenv_and_redacts_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_MODE", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_NAME", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_STRUCTURE_PROMPT_PATH", raising=False)
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_CRITERIA_PROMPT_PATH", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "FLYAGENT_TASK_PARSER_MODE=rule_based",
                "FLYAGENT_MODEL_PROVIDER=openai_compatible",
                "FLYAGENT_MODEL_NAME=test-model",
                "FLYAGENT_MODEL_API_KEY=secret-key",
                "FLYAGENT_MODEL_BASE_URL=https://example.test/v1",
                "FLYAGENT_MODEL_TIMEOUT_RETRIES=3",
                "FLYAGENT_MODEL_CONNECTION_RETRIES=5",
                "FLYAGENT_TASK_PARSER_STRUCTURE_PROMPT_PATH=prompts/structure.txt",
                "FLYAGENT_TASK_PARSER_CRITERIA_PROMPT_PATH=prompts/criteria.txt",
                "FLYAGENT_ACTION_PLANNER_PROMPT_PATH=prompts/action.txt",
                "FLYAGENT_AGENT_MAX_TOOL_ROUNDS=12",
                "FLYAGENT_AGENT_MAX_CONSECUTIVE_PROTOCOL_REJECTIONS=4",
            ]
        ),
        encoding="utf-8",
    )

    config = load_model_config(env_path)
    summary = config.redacted_summary()

    assert config.task_parser_mode == "rule_based"
    assert config.model_name == "test-model"
    assert config.api_key == "secret-key"
    assert config.provider_timeout_retries == 3
    assert config.provider_connection_retries == 5
    assert config.task_parser_structure_prompt_path == "prompts/structure.txt"
    assert config.task_parser_criteria_prompt_path == "prompts/criteria.txt"
    assert config.action_planner_prompt_path == "prompts/action.txt"
    assert config.max_tool_rounds == 12
    assert config.max_consecutive_protocol_rejections == 4
    assert summary["max_consecutive_protocol_rejections"] == 4
    assert summary["provider_timeout_retries"] == 3
    assert summary["provider_connection_retries"] == 5
    assert summary["has_api_key"] is True
    assert "secret-key" not in str(summary)


def test_model_config_process_env_overrides_dotenv(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("FLYAGENT_TASK_PARSER_MODE=rule_based\n", encoding="utf-8")
    monkeypatch.setenv("FLYAGENT_TASK_PARSER_MODE", "model")
    monkeypatch.setenv("FLYAGENT_MODEL_NAME", "env-model")

    config = load_model_config(env_path)

    assert config.task_parser_mode == "model"
    assert config.model_name == "env-model"


@pytest.mark.parametrize(
    "field",
    ["provider_timeout_retries", "provider_connection_retries"],
)
def test_model_config_rejects_negative_provider_retry_counts(field):
    with pytest.raises(ValueError, match=f"{field} must be non-negative"):
        ModelConfig(**{field: -1})
