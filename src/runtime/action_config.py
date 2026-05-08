"""Load unified UAV primitive action scale settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UavActionConfig:
    linear_velocity_mps: float = 2.0
    vertical_velocity_mps: float = 1.0
    move_duration_sec: float = 2.0
    vertical_duration_sec: float = 2.0
    vertical_step_m: float = 2.0
    yaw_step_deg: float = 30.0

    @classmethod
    def from_dict(cls, data: dict) -> "UavActionConfig":
        # Normalize JSON values into one fixed action scale profile.
        return cls(
            linear_velocity_mps=float(data.get("linear_velocity_mps", cls.linear_velocity_mps)),
            vertical_velocity_mps=float(data.get("vertical_velocity_mps", cls.vertical_velocity_mps)),
            move_duration_sec=float(data.get("move_duration_sec", cls.move_duration_sec)),
            vertical_duration_sec=float(data.get("vertical_duration_sec", cls.vertical_duration_sec)),
            vertical_step_m=float(data.get("vertical_step_m", cls.vertical_step_m)),
            yaw_step_deg=float(data.get("yaw_step_deg", cls.yaw_step_deg)),
        )

    def to_dict(self) -> dict[str, float]:
        # Return a JSON-safe view for logging and demo output.
        return {
            "linear_velocity_mps": self.linear_velocity_mps,
            "vertical_velocity_mps": self.vertical_velocity_mps,
            "move_duration_sec": self.move_duration_sec,
            "vertical_duration_sec": self.vertical_duration_sec,
            "vertical_step_m": self.vertical_step_m,
            "yaw_step_deg": self.yaw_step_deg,
        }


def load_uav_action_config(path: str | Path = "configs/uav_action_config.json") -> UavActionConfig:
    # Load the shared primitive action scale config, falling back to defaults if absent.
    config_path = Path(path)
    if not config_path.exists():
        return UavActionConfig()
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    return UavActionConfig.from_dict(data)
