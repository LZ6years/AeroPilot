"""Load model provider configuration for task parsing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    task_parser_mode: str = "model"
    provider: str = "openai_compatible"
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout_sec: float = 60.0
    max_tool_rounds: int = 6
    task_parser_prompt_path: str = ""
    task_parser_structure_prompt_path: str = ""
    task_parser_criteria_prompt_path: str = ""
    action_planner_prompt_path: str = ""

    def __post_init__(self) -> None:
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if self.max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "ModelConfig":
        return cls(
            task_parser_mode=env.get("FLYAGENT_TASK_PARSER_MODE", "model").strip() or "model",
            provider=env.get("FLYAGENT_MODEL_PROVIDER", "openai_compatible").strip()
            or "openai_compatible",
            model_name=env.get("FLYAGENT_MODEL_NAME", "").strip(),
            api_key=env.get("FLYAGENT_MODEL_API_KEY", "").strip(),
            base_url=env.get("FLYAGENT_MODEL_BASE_URL", "").strip(),
            timeout_sec=float(env.get("FLYAGENT_MODEL_TIMEOUT_SEC", "60")),
            max_tool_rounds=int(env.get("FLYAGENT_AGENT_MAX_TOOL_ROUNDS", "6")),
            task_parser_prompt_path=env.get("FLYAGENT_TASK_PARSER_PROMPT_PATH", "").strip(),
            task_parser_structure_prompt_path=env.get(
                "FLYAGENT_TASK_PARSER_STRUCTURE_PROMPT_PATH", ""
            ).strip(),
            task_parser_criteria_prompt_path=env.get(
                "FLYAGENT_TASK_PARSER_CRITERIA_PROMPT_PATH", ""
            ).strip(),
            action_planner_prompt_path=env.get("FLYAGENT_ACTION_PLANNER_PROMPT_PATH", "").strip(),
        )

    def redacted_summary(self) -> dict[str, str | float | bool]:
        return {
            "task_parser_mode": self.task_parser_mode,
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "timeout_sec": self.timeout_sec,
            "max_tool_rounds": self.max_tool_rounds,
            "task_parser_prompt_path": self.task_parser_prompt_path,
            "task_parser_structure_prompt_path": self.task_parser_structure_prompt_path,
            "task_parser_criteria_prompt_path": self.task_parser_criteria_prompt_path,
            "action_planner_prompt_path": self.action_planner_prompt_path,
        }

    def chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if not base_url:
            return "https://api.openai.com/v1/chat/completions"
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"


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
