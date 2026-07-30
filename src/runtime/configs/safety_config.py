"""Load UAV primitive safety limits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UavSafetyConfig:
    max_yaw_step_deg: float = 30.0
    max_linear_velocity_mps: float = 6.0
    max_vertical_velocity_mps: float = 3.0
    max_move_duration_sec: float = 8.0
    max_yaw_scan_extent_deg: float = 480.0

    @classmethod
    def from_dict(cls, data: dict) -> "UavSafetyConfig":
        return cls(
            max_yaw_step_deg=float(data.get("max_yaw_step_deg", cls.max_yaw_step_deg)),
            max_linear_velocity_mps=float(data.get("max_linear_velocity_mps", cls.max_linear_velocity_mps)),
            max_vertical_velocity_mps=float(data.get("max_vertical_velocity_mps", cls.max_vertical_velocity_mps)),
            max_move_duration_sec=float(data.get("max_move_duration_sec", cls.max_move_duration_sec)),
            max_yaw_scan_extent_deg=float(
                data.get(
                    "max_yaw_scan_extent_deg",
                    data.get("max_yaw_scan_accum_deg", cls.max_yaw_scan_extent_deg),
                )
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "max_yaw_step_deg": self.max_yaw_step_deg,
            "max_linear_velocity_mps": self.max_linear_velocity_mps,
            "max_vertical_velocity_mps": self.max_vertical_velocity_mps,
            "max_move_duration_sec": self.max_move_duration_sec,
            "max_yaw_scan_extent_deg": self.max_yaw_scan_extent_deg,
        }


def load_uav_safety_config(path: str | Path = "configs/uav_safety_config.json") -> UavSafetyConfig:
    config_path = Path(path)
    if not config_path.exists():
        return UavSafetyConfig()
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    return UavSafetyConfig.from_dict(data)
