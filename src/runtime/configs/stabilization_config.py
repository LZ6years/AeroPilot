"""Load UAV runtime stabilization settings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class UavStabilizationConfig:
    """Controls when and how runtime waits for motion to settle."""

    enabled: bool = True
    velocity_threshold_mps: float = 0.15
    timeout_sec: float = 8.0
    poll_interval_sec: float = 0.25
    initialization_min_settle_sec: float = 0.0
    zero_velocity_brake_duration_sec: float = 2.0
    hover_after_brake: bool = False
    motion_tools: set[str] = field(default_factory=lambda: {"uav.move_body", "uav.turn_relative"})

    def __post_init__(self) -> None:
        value = float(self.initialization_min_settle_sec)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "initialization_min_settle_sec must be a finite "
                "non-negative number"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "UavStabilizationConfig":
        hover_after_brake = data.get("hover_after_brake", data.get("hover_before_wait", cls.hover_after_brake))
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            velocity_threshold_mps=float(data.get("velocity_threshold_mps", cls.velocity_threshold_mps)),
            timeout_sec=float(data.get("timeout_sec", cls.timeout_sec)),
            poll_interval_sec=float(data.get("poll_interval_sec", cls.poll_interval_sec)),
            initialization_min_settle_sec=float(
                data.get(
                    "initialization_min_settle_sec",
                    cls.initialization_min_settle_sec,
                )
            ),
            zero_velocity_brake_duration_sec=float(
                data.get("zero_velocity_brake_duration_sec", cls.zero_velocity_brake_duration_sec)
            ),
            hover_after_brake=bool(hover_after_brake),
            motion_tools=set(data.get("motion_tools", cls().motion_tools)),
        )

    @property
    def hover_before_wait(self) -> bool:
        """Backward-compatible alias for older callers/config snapshots."""

        return self.hover_after_brake

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "velocity_threshold_mps": self.velocity_threshold_mps,
            "timeout_sec": self.timeout_sec,
            "poll_interval_sec": self.poll_interval_sec,
            "initialization_min_settle_sec": (
                self.initialization_min_settle_sec
            ),
            "zero_velocity_brake_duration_sec": self.zero_velocity_brake_duration_sec,
            "hover_after_brake": self.hover_after_brake,
            "motion_tools": sorted(self.motion_tools),
        }


def load_uav_stabilization_config(
    path: str | Path = "configs/uav_stabilization_config.json",
) -> UavStabilizationConfig:
    config_path = Path(path)
    if not config_path.exists():
        return UavStabilizationConfig()
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    return UavStabilizationConfig.from_dict(data)
