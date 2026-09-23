"""Load provider settings and caller-owned unified tool-loop configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.contracts import ProviderSettings


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "openai_compatible"
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout_sec: float = 60.0
    provider_timeout_retries: int = 1
    provider_connection_retries: int = 2
    max_tool_rounds: int = 10
    max_consecutive_protocol_rejections: int = 6
    action_planner_prompt_path: str = ""
    # Completed decisions between reminders; zero disables request-only reminders.
    task_progress_reminder_interval: int = 5

    def __post_init__(self) -> None:
        if type(self.task_progress_reminder_interval) is not int or self.task_progress_reminder_interval < 0:
            raise ValueError("task_progress_reminder_interval must be a non-negative integer")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if self.provider_timeout_retries < 0:
            raise ValueError("provider_timeout_retries must be non-negative")
        if self.provider_connection_retries < 0:
            raise ValueError("provider_connection_retries must be non-negative")
        if self.max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")
        if self.max_consecutive_protocol_rejections <= 0:
            raise ValueError("max_consecutive_protocol_rejections must be positive")

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "ModelConfig":
        if any(key.startswith("FLYAGENT_TASK_PARSER_") for key in env):
            raise ValueError("Parser configuration was removed; use the unified tool loop")
        if any(key.startswith("FLYAGENT_RUNTIME_CONTROL_") for key in env):
            raise ValueError("Legacy runtime control configuration was removed; use built-in tools")
        return cls(
            provider=env.get("FLYAGENT_MODEL_PROVIDER", "openai_compatible").strip()
            or "openai_compatible",
            model_name=env.get("FLYAGENT_MODEL_NAME", "").strip(),
            api_key=env.get("FLYAGENT_MODEL_API_KEY", "").strip(),
            base_url=env.get("FLYAGENT_MODEL_BASE_URL", "").strip(),
            timeout_sec=float(env.get("FLYAGENT_MODEL_TIMEOUT_SEC", "60")),
            provider_timeout_retries=int(
                env.get("FLYAGENT_MODEL_TIMEOUT_RETRIES", "1")
            ),
            provider_connection_retries=int(
                env.get("FLYAGENT_MODEL_CONNECTION_RETRIES", "2")
            ),
            max_tool_rounds=int(env.get("FLYAGENT_AGENT_MAX_TOOL_ROUNDS", "10")),
            task_progress_reminder_interval=int(env.get("FLYAGENT_AGENT_TASK_PROGRESS_REMINDER_INTERVAL", "5")),
            max_consecutive_protocol_rejections=int(
                env.get("FLYAGENT_AGENT_MAX_CONSECUTIVE_PROTOCOL_REJECTIONS", "6")
            ),


            action_planner_prompt_path=env.get("FLYAGENT_ACTION_PLANNER_PROMPT_PATH", "").strip(),
        )

    def redacted_summary(self) -> dict[str, str | float | int | bool]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "timeout_sec": self.timeout_sec,
            "provider_timeout_retries": self.provider_timeout_retries,
            "provider_connection_retries": self.provider_connection_retries,
            "max_tool_rounds": self.max_tool_rounds,
            "task_progress_reminder_interval": self.task_progress_reminder_interval,
            "max_consecutive_protocol_rejections": self.max_consecutive_protocol_rejections,
            "action_planner_prompt_path": self.action_planner_prompt_path,
        }

    def to_provider_settings(self) -> ProviderSettings:
        """Pass only communication settings to the business-independent client."""
        from src.llm.contracts import ProviderSettings

        return ProviderSettings(
            provider=self.provider, model_name=self.model_name, api_key=self.api_key,
            base_url=self.base_url, timeout_sec=self.timeout_sec,
            provider_timeout_retries=self.provider_timeout_retries,
            provider_connection_retries=self.provider_connection_retries,
        )

    def chat_completions_url(self) -> str:
        """Compatibility convenience; the provider adapter owns URL policy."""
        from src.llm.providers.openai_compatible import chat_completions_url

        return chat_completions_url(self.base_url)


def load_model_config(env_path: str | Path = ".env") -> ModelConfig:
    # Merge .env first, then let process env override local file values.
    merged = _read_dotenv(Path(env_path))
    merged.update({key: value for key, value in os.environ.items() if key.startswith("FLYAGENT_")})
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
