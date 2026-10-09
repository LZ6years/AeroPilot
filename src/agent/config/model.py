"""Load Agent configuration without introducing a Runtime dependency.

``ModelConfig`` remains the environment/file aggregate used by composition
roots.  The planner itself receives the three narrower contracts below, so
provider transport, prompt policy, and tool-loop limits have distinct owners.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.llm.contracts import ProviderSettings


OutputLanguage = Literal["en", "zh"]
OUTPUT_LANGUAGES = frozenset({"en", "zh"})


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "openai_compatible"
    model_name: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    api_protocol: str = "chat_completions"
    stream: bool = False
    thinking_mode: str = ""
    thinking_wire_format: str = "thinking_object"
    reasoning_effort: str = ""
    structured_output_format: str = "json_schema"
    timeout_sec: float = 60.0
    timeout_retries: int = 1
    connection_retries: int = 2
    retry_deadline_sec: float = 600.0

    def __post_init__(self) -> None:
        if self.provider != "openai_compatible":
            raise ValueError(f"unsupported model provider: {self.provider}")
        if self.api_protocol not in ("chat_completions", "responses"):
            raise ValueError(
                "api_protocol must be chat_completions or responses"
            )
        if self.api_protocol == "responses" and self.stream:
            raise ValueError("streaming Responses API is not supported")
        if self.thinking_mode not in ("", "enabled", "disabled"):
            raise ValueError("thinking_mode must be enabled, disabled, or unset")
        if self.reasoning_effort not in (
            "",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ):
            raise ValueError("unsupported reasoning_effort")
        if self.thinking_mode and self.reasoning_effort:
            raise ValueError(
                "thinking_mode and reasoning_effort are mutually exclusive"
            )
        if self.thinking_wire_format not in (
            "thinking_object",
            "enable_thinking",
        ):
            raise ValueError(
                "thinking_wire_format must be thinking_object or enable_thinking"
            )
        if self.structured_output_format not in ("json_schema", "json_object"):
            raise ValueError(
                "structured_output_format must be json_schema or json_object"
            )
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        # Reuse the transport contract's numeric validation.
        self.to_provider_settings()
        for value in (self.timeout_retries, self.connection_retries):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("provider retry counts must be non-negative integers")

    def to_provider_settings(self) -> ProviderSettings:
        return ProviderSettings(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            api_protocol=self.api_protocol,
            stream=self.stream,
            timeout_sec=self.timeout_sec,
            provider_timeout_retries=self.timeout_retries,
            provider_connection_retries=self.connection_retries,
            retry_deadline_sec=self.retry_deadline_sec,
        )

    def thinking_parameters(self, *, mode: str | None = None) -> dict[str, object]:
        """Build caller-owned generation parameters for one provider dialect.

        OpenAI-compatible endpoints do not agree on the extension used to
        control hybrid reasoning. Keeping this choice explicit avoids guessing
        from a base URL while allowing every model caller to use one contract.
        """

        resolved = self.thinking_mode if mode is None else mode
        if resolved not in ("", "enabled", "disabled"):
            raise ValueError("thinking mode must be enabled, disabled, or unset")
        if not resolved:
            return {}
        if self.thinking_wire_format == "enable_thinking":
            return {"enable_thinking": resolved == "enabled"}
        return {"thinking": {"type": resolved}}

    def generation_parameters(self) -> dict[str, object]:
        """Return explicit caller-owned reasoning controls for normal turns."""

        parameters = self.thinking_parameters()
        if self.reasoning_effort:
            parameters["reasoning_effort"] = self.reasoning_effort
        return parameters

    def compaction_parameters(self) -> dict[str, object]:
        """Use the lowest explicit reasoning mode supported by this provider."""

        if self.reasoning_effort:
            return {"reasoning_effort": self.reasoning_effort}
        return self.thinking_parameters(mode="disabled")


@dataclass(frozen=True)
class PlannerConfig:
    prompt_path: str = ""
    task_progress_reminder_interval: int = 5
    output_language: OutputLanguage = "en"

    def __post_init__(self) -> None:
        value = self.task_progress_reminder_interval
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "task_progress_reminder_interval must be a non-negative integer"
            )
        if self.output_language not in OUTPUT_LANGUAGES:
            raise ValueError("output_language must be en or zh")


@dataclass(frozen=True)
class ToolLoopConfig:
    max_rounds: int = 10
    max_consecutive_protocol_rejections: int = 6

    def __post_init__(self) -> None:
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.max_consecutive_protocol_rejections <= 0:
            raise ValueError(
                "max_consecutive_protocol_rejections must be positive"
            )


@dataclass(frozen=True)
class ModelConfig:
    """Flat serialization/environment boundary, not a planner dependency."""

    provider: str = "openai_compatible"
    model_name: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    api_protocol: str = "chat_completions"
    stream: bool = False
    thinking_mode: str = ""
    thinking_wire_format: str = "thinking_object"
    reasoning_effort: str = ""
    structured_output_format: str = "json_schema"
    timeout_sec: float = 60.0
    provider_timeout_retries: int = 1
    provider_connection_retries: int = 2
    retry_deadline_sec: float = 600.0
    max_tool_rounds: int = 10
    max_consecutive_protocol_rejections: int = 6
    action_planner_prompt_path: str = ""
    task_progress_reminder_interval: int = 5
    output_language: OutputLanguage = "en"

    def __post_init__(self) -> None:
        if self.provider_timeout_retries < 0:
            raise ValueError("provider_timeout_retries must be non-negative")
        if self.provider_connection_retries < 0:
            raise ValueError("provider_connection_retries must be non-negative")
        self.provider_config()
        self.planner_config()
        self.tool_loop_config()

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "ModelConfig":
        if any(key.startswith("FLYAGENT_TASK_PARSER_") for key in env):
            raise ValueError("Parser configuration was removed; use the unified tool loop")
        if any(key.startswith("FLYAGENT_RUNTIME_CONTROL_") for key in env):
            raise ValueError(
                "Legacy runtime control configuration was removed; use built-in tools"
            )
        return cls(
            provider=env.get("FLYAGENT_MODEL_PROVIDER", "openai_compatible").strip()
            or "openai_compatible",
            model_name=env.get("FLYAGENT_MODEL_NAME", "").strip(),
            api_key=env.get("FLYAGENT_MODEL_API_KEY", "").strip(),
            base_url=env.get("FLYAGENT_MODEL_BASE_URL", "").strip(),
            api_protocol=env.get(
                "FLYAGENT_MODEL_API_PROTOCOL", "chat_completions"
            ).strip()
            or "chat_completions",
            stream=_parse_env_bool(env.get("FLYAGENT_MODEL_STREAM", "false")),
            thinking_mode=env.get("FLYAGENT_MODEL_THINKING_MODE", "").strip(),
            thinking_wire_format=env.get(
                "FLYAGENT_MODEL_THINKING_WIRE_FORMAT", "thinking_object"
            ).strip()
            or "thinking_object",
            reasoning_effort=env.get(
                "FLYAGENT_MODEL_REASONING_EFFORT", ""
            ).strip(),
            structured_output_format=env.get(
                "FLYAGENT_MODEL_STRUCTURED_OUTPUT_FORMAT", "json_schema"
            ).strip()
            or "json_schema",
            timeout_sec=float(env.get("FLYAGENT_MODEL_TIMEOUT_SEC", "60")),
            provider_timeout_retries=int(
                env.get("FLYAGENT_MODEL_TIMEOUT_RETRIES", "1")
            ),
            provider_connection_retries=int(
                env.get("FLYAGENT_MODEL_CONNECTION_RETRIES", "2")
            ),
            retry_deadline_sec=float(
                env.get("FLYAGENT_MODEL_RETRY_DEADLINE_SEC", "600")
            ),
            max_tool_rounds=int(env.get("FLYAGENT_AGENT_MAX_TOOL_ROUNDS", "10")),
            max_consecutive_protocol_rejections=int(
                env.get("FLYAGENT_AGENT_MAX_CONSECUTIVE_PROTOCOL_REJECTIONS", "6")
            ),
            action_planner_prompt_path=env.get(
                "FLYAGENT_ACTION_PLANNER_PROMPT_PATH", ""
            ).strip(),
            task_progress_reminder_interval=int(
                env.get("FLYAGENT_AGENT_TASK_PROGRESS_REMINDER_INTERVAL", "5")
            ),
            output_language=_parse_output_language(
                env.get("FLYAGENT_AGENT_OUTPUT_LANGUAGE", "en")
            ),
        )

    def provider_config(self) -> ProviderConfig:
        return ProviderConfig(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            api_protocol=self.api_protocol,
            stream=self.stream,
            thinking_mode=self.thinking_mode,
            thinking_wire_format=self.thinking_wire_format,
            reasoning_effort=self.reasoning_effort,
            structured_output_format=self.structured_output_format,
            timeout_sec=self.timeout_sec,
            timeout_retries=self.provider_timeout_retries,
            connection_retries=self.provider_connection_retries,
            retry_deadline_sec=self.retry_deadline_sec,
        )

    def planner_config(self) -> PlannerConfig:
        return PlannerConfig(
            prompt_path=self.action_planner_prompt_path,
            task_progress_reminder_interval=self.task_progress_reminder_interval,
            output_language=self.output_language,
        )

    def tool_loop_config(self) -> ToolLoopConfig:
        return ToolLoopConfig(
            max_rounds=self.max_tool_rounds,
            max_consecutive_protocol_rejections=(
                self.max_consecutive_protocol_rejections
            ),
        )

    def redacted_summary(self) -> dict[str, str | float | int | bool]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_protocol": self.api_protocol,
            "stream": self.stream,
            "thinking_mode": self.thinking_mode,
            "thinking_wire_format": self.thinking_wire_format,
            "reasoning_effort": self.reasoning_effort,
            "structured_output_format": self.structured_output_format,
            "has_api_key": bool(self.api_key),
            "timeout_sec": self.timeout_sec,
            "provider_timeout_retries": self.provider_timeout_retries,
            "provider_connection_retries": self.provider_connection_retries,
            "retry_deadline_sec": self.retry_deadline_sec,
            "max_tool_rounds": self.max_tool_rounds,
            "task_progress_reminder_interval": self.task_progress_reminder_interval,
            "output_language": self.output_language,
            "max_consecutive_protocol_rejections": (
                self.max_consecutive_protocol_rejections
            ),
            "action_planner_prompt_path": self.action_planner_prompt_path,
        }

    def to_provider_settings(self) -> ProviderSettings:
        return self.provider_config().to_provider_settings()

    def thinking_parameters(self, *, mode: str | None = None) -> dict[str, object]:
        """Delegate provider-specific generation options to the narrow config."""

        return self.provider_config().thinking_parameters(mode=mode)

    def generation_parameters(self) -> dict[str, object]:
        return self.provider_config().generation_parameters()

    def compaction_parameters(self) -> dict[str, object]:
        return self.provider_config().compaction_parameters()

def load_model_config(env_path: str | Path = ".env") -> ModelConfig:
    merged = read_env_file(env_path)
    merged.update(
        {key: value for key, value in os.environ.items() if key.startswith("FLYAGENT_")}
    )
    return ModelConfig.from_env(merged)


def read_env_file(path: str | Path = ".env") -> dict[str, str]:
    """Read a dotenv-style file without mutating the process environment."""

    path = Path(path)
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _strip_env_value(value.strip())
    return values


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("FLYAGENT_MODEL_STREAM must be a boolean")


def _parse_output_language(value: str) -> OutputLanguage:
    text = value.strip()
    if text == "en":
        return "en"
    if text == "zh":
        return "zh"
    raise ValueError("FLYAGENT_AGENT_OUTPUT_LANGUAGE must be en or zh")
