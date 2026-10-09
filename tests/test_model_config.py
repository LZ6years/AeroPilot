"""Test model provider configuration loading."""

from __future__ import annotations

import pytest

from src.agent.config import ModelConfig
from src.agent.config import load_model_config


def test_model_config_defaults_to_model_mode_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_MODE", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_NAME", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_API_PROTOCOL", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_STREAM", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_THINKING_WIRE_FORMAT", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_STRUCTURED_OUTPUT_FORMAT", raising=False)
    monkeypatch.delenv("FLYAGENT_AGENT_OUTPUT_LANGUAGE", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_TIMEOUT_RETRIES", raising=False)
    monkeypatch.delenv("FLYAGENT_MODEL_CONNECTION_RETRIES", raising=False)
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_STRUCTURE_PROMPT_PATH", raising=False)
    monkeypatch.delenv("FLYAGENT_TASK_PARSER_CRITERIA_PROMPT_PATH", raising=False)
    monkeypatch.delenv("FLYAGENT_AGENT_MAX_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv(
        "FLYAGENT_AGENT_MAX_CONSECUTIVE_PROTOCOL_REJECTIONS", raising=False
    )

    config = load_model_config(tmp_path / ".env")

    assert config.provider == "openai_compatible"
    assert config.api_key == ""
    assert config.api_protocol == "chat_completions"
    assert config.reasoning_effort == ""
    assert config.provider_timeout_retries == 1
    assert config.provider_connection_retries == 2
    assert config.max_tool_rounds == 10
    assert config.max_consecutive_protocol_rejections == 6
    assert config.output_language == "en"


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
                "FLYAGENT_MODEL_PROVIDER=openai_compatible",
                "FLYAGENT_MODEL_NAME=test-model",
                "FLYAGENT_MODEL_API_KEY=secret-key",
                "FLYAGENT_MODEL_BASE_URL=https://example.test/v1",
                "FLYAGENT_MODEL_API_PROTOCOL=responses",
                "FLYAGENT_MODEL_REASONING_EFFORT=low",
                "FLYAGENT_MODEL_STREAM=false",
                "FLYAGENT_MODEL_TIMEOUT_RETRIES=3",
                "FLYAGENT_MODEL_CONNECTION_RETRIES=5",
                "FLYAGENT_ACTION_PLANNER_PROMPT_PATH=prompts/action.txt",
                "FLYAGENT_AGENT_MAX_TOOL_ROUNDS=12",
                "FLYAGENT_AGENT_MAX_CONSECUTIVE_PROTOCOL_REJECTIONS=4",
                "FLYAGENT_AGENT_OUTPUT_LANGUAGE=zh",
            ]
        ),
        encoding="utf-8",
    )

    config = load_model_config(env_path)
    summary = config.redacted_summary()

    assert config.model_name == "test-model"
    assert config.api_key == "secret-key"
    assert config.provider_timeout_retries == 3
    assert config.provider_connection_retries == 5
    assert config.api_protocol == "responses"
    assert config.reasoning_effort == "low"
    assert config.stream is False
    assert config.to_provider_settings().api_protocol == "responses"
    assert config.action_planner_prompt_path == "prompts/action.txt"
    assert config.max_tool_rounds == 12
    assert config.max_consecutive_protocol_rejections == 4
    assert config.output_language == "zh"
    assert config.planner_config().output_language == "zh"
    assert summary["max_consecutive_protocol_rejections"] == 4
    assert summary["output_language"] == "zh"
    assert summary["provider_timeout_retries"] == 3
    assert summary["provider_connection_retries"] == 5
    assert summary["api_protocol"] == "responses"
    assert summary["reasoning_effort"] == "low"
    assert summary["has_api_key"] is True
    assert "secret-key" not in str(summary)


def test_model_config_process_env_overrides_dotenv(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("FLYAGENT_MODEL_NAME=file-model\n", encoding="utf-8")
    monkeypatch.setenv("FLYAGENT_MODEL_NAME", "env-model")

    config = load_model_config(env_path)

    assert config.model_name == "env-model"


@pytest.mark.parametrize(
    "field",
    ["provider_timeout_retries", "provider_connection_retries"],
)
def test_model_config_rejects_negative_provider_retry_counts(field):
    with pytest.raises(ValueError, match=f"{field} must be non-negative"):
        ModelConfig(**{field: -1})


def test_removed_codex_sdk_provider_fails_during_configuration():
    with pytest.raises(ValueError, match="unsupported model provider"):
        ModelConfig(provider="codex_sdk", api_protocol="responses")


def test_unknown_model_provider_fails_during_configuration():
    with pytest.raises(ValueError, match="unsupported model provider"):
        ModelConfig(provider="unknown")



def test_removed_parser_configuration_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYAGENT_TASK_PARSER_MODE", "model")
    with pytest.raises(ValueError, match="removed"):
        load_model_config(tmp_path / ".env")


def test_removed_control_environment_fails_explicitly():
    with pytest.raises(ValueError, match="Legacy runtime control configuration was removed"):
        ModelConfig.from_env({"FLYAGENT_RUNTIME_CONTROL_CONFIG": "obsolete.json"})


def test_thinking_mode_is_optional_validated_and_not_transport():
    import pytest
    from dataclasses import asdict
    for mode in ('', 'enabled', 'disabled'):
        config = ModelConfig.from_env({'FLYAGENT_MODEL_THINKING_MODE': mode})
        assert config.thinking_mode == mode
        assert config.redacted_summary()['thinking_mode'] == mode
        assert 'thinking_mode' not in asdict(config.to_provider_settings())
    with pytest.raises(ValueError, match='thinking_mode'):
        ModelConfig(thinking_mode='automatic')


def test_thinking_wire_format_builds_explicit_provider_parameters():
    legacy = ModelConfig(thinking_mode="disabled")
    dashscope = ModelConfig(
        thinking_mode="disabled",
        thinking_wire_format="enable_thinking",
    )

    assert legacy.thinking_parameters() == {"thinking": {"type": "disabled"}}
    assert dashscope.thinking_parameters() == {"enable_thinking": False}
    assert dashscope.thinking_parameters(mode="enabled") == {
        "enable_thinking": True
    }
    assert ModelConfig().thinking_parameters() == {}
    assert dashscope.redacted_summary()["thinking_wire_format"] == "enable_thinking"

    with pytest.raises(ValueError, match="thinking_wire_format"):
        ModelConfig(thinking_wire_format="guess_from_url")


def test_thinking_wire_format_loads_from_environment():
    config = ModelConfig.from_env(
        {
            "FLYAGENT_MODEL_THINKING_MODE": "disabled",
            "FLYAGENT_MODEL_THINKING_WIRE_FORMAT": "enable_thinking",
        }
    )

    assert config.thinking_parameters() == {"enable_thinking": False}


def test_reasoning_effort_builds_explicit_generation_parameters():
    config = ModelConfig.from_env(
        {
            "FLYAGENT_MODEL_API_PROTOCOL": "responses",
            "FLYAGENT_MODEL_REASONING_EFFORT": "low",
        }
    )

    assert config.generation_parameters() == {"reasoning_effort": "low"}
    assert config.compaction_parameters() == {"reasoning_effort": "low"}
    assert config.provider_config().reasoning_effort == "low"

    with pytest.raises(ValueError, match="reasoning_effort"):
        ModelConfig(reasoning_effort="automatic")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelConfig(thinking_mode="disabled", reasoning_effort="low")


def test_responses_protocol_rejects_streaming_until_supported():
    with pytest.raises(ValueError, match="streaming Responses"):
        ModelConfig(api_protocol="responses", stream=True)
    with pytest.raises(ValueError, match="api_protocol"):
        ModelConfig(api_protocol="automatic")


def test_structured_output_format_is_explicit_and_validated():
    config = ModelConfig.from_env(
        {"FLYAGENT_MODEL_STRUCTURED_OUTPUT_FORMAT": "json_object"}
    )

    assert config.structured_output_format == "json_object"
    assert config.provider_config().structured_output_format == "json_object"
    assert config.redacted_summary()["structured_output_format"] == "json_object"

    with pytest.raises(ValueError, match="structured_output_format"):
        ModelConfig(structured_output_format="automatic")


@pytest.mark.parametrize("value", ["true", "1", "yes", "on"])
def test_model_stream_true_values_are_explicit(value):
    config = ModelConfig.from_env({"FLYAGENT_MODEL_STREAM": value})

    assert config.stream is True
    assert config.provider_config().stream is True
    assert config.redacted_summary()["stream"] is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
def test_model_stream_false_values_are_explicit(value):
    assert ModelConfig.from_env({"FLYAGENT_MODEL_STREAM": value}).stream is False


def test_model_stream_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="FLYAGENT_MODEL_STREAM"):
        ModelConfig.from_env({"FLYAGENT_MODEL_STREAM": "automatic"})


@pytest.mark.parametrize("value", ["", "auto", "english", "ZH", "en-US"])
def test_output_language_rejects_noncanonical_values(value):
    with pytest.raises(ValueError, match="FLYAGENT_AGENT_OUTPUT_LANGUAGE"):
        ModelConfig.from_env({"FLYAGENT_AGENT_OUTPUT_LANGUAGE": value})


@pytest.mark.parametrize("value", ["en", "zh"])
def test_output_language_accepts_explicit_values(value):
    config = ModelConfig.from_env({"FLYAGENT_AGENT_OUTPUT_LANGUAGE": value})

    assert config.output_language == value
    assert config.planner_config().output_language == value
