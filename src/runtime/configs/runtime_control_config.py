"""Load runtime control action availability settings.

Runtime control actions are orchestration requests, not UAV tools. This config
only decides which registered actions may be exposed to the planner in the
current runtime; execution still goes through the runtime control executor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuntimeControlActionConfig:
    """Enablement and model exposure for one runtime control action."""

    enabled: bool = False
    exposed: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeControlActionConfig":
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            exposed=bool(data.get("exposed", cls.exposed)),
        )

    def to_dict(self) -> dict[str, bool]:
        return {"enabled": self.enabled, "exposed": self.exposed}


@dataclass(frozen=True)
class RuntimeControlConfig:
    """Top-level runtime control configuration.

    Defaults intentionally expose no action, so adding the control plane cannot
    change existing flight behavior until a caller opts in explicitly.
    """

    enabled: bool = False
    actions: dict[str, RuntimeControlActionConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeControlConfig":
        actions_data = data.get("actions", {})
        actions: dict[str, RuntimeControlActionConfig] = {}
        if isinstance(actions_data, dict):
            actions = {
                str(name).strip(): RuntimeControlActionConfig.from_dict(config if isinstance(config, dict) else {})
                for name, config in actions_data.items()
                if str(name).strip()
            }
        return cls(enabled=bool(data.get("enabled", cls.enabled)), actions=actions)

    def action_config(self, name: str) -> RuntimeControlActionConfig:
        return self.actions.get(name, RuntimeControlActionConfig())

    def is_enabled(self, name: str) -> bool:
        action = self.action_config(name)
        return self.enabled and action.enabled

    def is_exposed(self, name: str) -> bool:
        action = self.action_config(name)
        return self.enabled and action.enabled and action.exposed

    def exposed_actions(self, registered_names: list[str]) -> list[str]:
        return [name for name in registered_names if self.is_exposed(name)]

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "actions": {name: config.to_dict() for name, config in sorted(self.actions.items())},
        }


def load_runtime_control_config(
    path: str | Path = "configs/runtime_control_config.json",
) -> RuntimeControlConfig:
    """Load runtime control config with coarse environment overrides."""

    config_path = Path(path)
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
        config = RuntimeControlConfig.from_dict(data)
    else:
        config = RuntimeControlConfig()
    return _apply_env_overrides(config)


def _apply_env_overrides(config: RuntimeControlConfig) -> RuntimeControlConfig:
    enabled = _env_bool("FLYAGENT_RUNTIME_CONTROL_ENABLED", config.enabled)
    actions = dict(config.actions)
    for name in ("ask_user", "semantic_plan_update"):
        current = actions.get(name, RuntimeControlActionConfig())
        action_enabled = _env_bool(_action_env_name(name, "ENABLED"), current.enabled)
        action_exposed = _env_bool(_action_env_name(name, "EXPOSED"), current.exposed)
        actions[name] = RuntimeControlActionConfig(enabled=action_enabled, exposed=action_exposed)
    return RuntimeControlConfig(enabled=enabled, actions=actions)


def _action_env_name(action_name: str, suffix: str) -> str:
    return f"FLYAGENT_RUNTIME_CONTROL_{action_name.upper()}_{suffix}"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
