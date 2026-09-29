"""Load Agent configuration without introducing a Runtime dependency.

``ModelConfig`` remains the environment/file aggregate used by composition
roots.  The planner itself receives the three narrower contracts below, so
provider transport, prompt policy, and tool-loop limits have distinct owners.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.llm.contracts import ProviderSettings


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "openai_compatible"
    model_name: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    thinking_mode: str = ""
    timeout_sec: float = 60.0
    timeout_retries: int = 1
    connection_retries: int = 2

    def __post_init__(self) -> None:
        if self.thinking_mode not in ("", "enabled", "disabled"):
            raise ValueError("thinking_mode must be enabled, disabled, or unset")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        for value in (self.timeout_retries, self.connection_retries):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("provider retry counts must be non-negative integers")

    def to_provider_settings(self) -> ProviderSettings:
        return ProviderSettings(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_sec=self.timeout_sec,
            provider_timeout_retries=self.timeout_retries,
            provider_connection_retries=self.connection_retries,
        )


@dataclass(frozen=True)
class PlannerConfig:
    prompt_path: str = ""
    task_progress_reminder_interval: int = 5

    def __post_init__(self) -> None:
        value = self.task_progress_reminder_interval
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "task_progress_reminder_interval must be a non-negative integer"
            )


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
    thinking_mode: str = ""
    timeout_sec: float = 60.0
    provider_timeout_retries: int = 1
    provider_connection_retries: int = 2
    max_tool_rounds: int = 10
    max_consecutive_protocol_rejections: int = 6
    action_planner_prompt_path: str = ""
    task_progress_reminder_interval: int = 5

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
            thinking_mode=env.get("FLYAGENT_MODEL_THINKING_MODE", "").strip(),
            timeout_sec=float(env.get("FLYAGENT_MODEL_TIMEOUT_SEC", "60")),
            provider_timeout_retries=int(
                env.get("FLYAGENT_MODEL_TIMEOUT_RETRIES", "1")
            ),
            provider_connection_retries=int(
                env.get("FLYAGENT_MODEL_CONNECTION_RETRIES", "2")
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
        )

    def provider_config(self) -> ProviderConfig:
        return ProviderConfig(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            thinking_mode=self.thinking_mode,
            timeout_sec=self.timeout_sec,
            timeout_retries=self.provider_timeout_retries,
            connection_retries=self.provider_connection_retries,
        )

    def planner_config(self) -> PlannerConfig:
        return PlannerConfig(
            prompt_path=self.action_planner_prompt_path,
            task_progress_reminder_interval=self.task_progress_reminder_interval,
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
            "thinking_mode": self.thinking_mode,
            "has_api_key": bool(self.api_key),
            "timeout_sec": self.timeout_sec,
            "provider_timeout_retries": self.provider_timeout_retries,
            "provider_connection_retries": self.provider_connection_retries,
            "max_tool_rounds": self.max_tool_rounds,
            "task_progress_reminder_interval": self.task_progress_reminder_interval,
            "max_consecutive_protocol_rejections": (
                self.max_consecutive_protocol_rejections
            ),
            "action_planner_prompt_path": self.action_planner_prompt_path,
        }

    def to_provider_settings(self) -> ProviderSettings:
        return self.provider_config().to_provider_settings()

    def chat_completions_url(self) -> str:
        from src.llm.providers.openai_compatible import chat_completions_url

        return chat_completions_url(self.base_url)


def load_model_config(env_path: str | Path = ".env") -> ModelConfig:
    merged = _read_dotenv(Path(env_path))
    merged.update(
        {key: value for key, value in os.environ.items() if key.startswith("FLYAGENT_")}
    )
    return ModelConfig.from_env(merged)


def _read_dotenv(path: Path) -> dict[str, str]:
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
