"""Define per-task execution inputs that stay outside semantic planning.

Benchmarks and other orchestrators may select an episode start pose and the
canonical observations available to the Agent.  Runtime applies those choices
before the first planner window; simulator-specific coordinates and evaluation
truth remain outside this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.env_client import ObservationProfile


@dataclass(frozen=True)
class InitialPose:
    """Describe one absolute pose in FlyAgent's canonical coordinate frame.

    Positions use meters with positive Z pointing up. Euler angles use degrees.
    Dataset adapters must convert their raw position and quaternion conventions
    before constructing this value.
    """

    x: float
    y: float
    z: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    ignore_collision: bool = True

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"InitialPose {name} must be a finite number")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"InitialPose {name} must be a finite number"
                ) from exc
            if not math.isfinite(normalized):
                raise ValueError(f"InitialPose {name} must be a finite number")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.ignore_collision, bool):
            raise ValueError("InitialPose ignore_collision must be a boolean")

    def to_action_args(self) -> dict[str, float | bool]:
        """Return the canonical ``uav.set_pose`` argument shape."""

        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "ignore_collision": self.ignore_collision,
        }

    def to_dict(self) -> dict[str, float | bool]:
        """Return a stable JSON-serializable representation."""

        return self.to_action_args()


@dataclass(frozen=True)
class TaskExecutionOptions:
    """Configure episode initialization and Agent-visible observations.

    ``pose`` remains mandatory in v1 because MissionState motion measurements
    and the current flight-memory contract depend on it. Runtime-internal pose
    polling for stabilization is separate from this Agent-visible selection.
    """

    observation_modalities: tuple[str, ...] = field(
        default_factory=lambda: ObservationProfile().modalities
    )
    initial_pose: InitialPose | None = None

    def __post_init__(self) -> None:
        raw_modalities = self.observation_modalities
        if isinstance(raw_modalities, str) or not isinstance(
            raw_modalities, (list, tuple)
        ):
            raise ValueError(
                "TaskExecutionOptions observation_modalities must be a non-empty sequence"
            )
        modalities = tuple(str(value).strip().lower() for value in raw_modalities)
        if not modalities or any(not value for value in modalities):
            raise ValueError(
                "TaskExecutionOptions observation_modalities must be non-empty"
            )
        if len(set(modalities)) != len(modalities):
            raise ValueError(
                "TaskExecutionOptions observation_modalities must be unique"
            )
        if "pose" not in modalities:
            raise ValueError(
                "TaskExecutionOptions observation_modalities must include pose"
            )
        if self.initial_pose is not None and not isinstance(
            self.initial_pose, InitialPose
        ):
            raise ValueError(
                "TaskExecutionOptions initial_pose must be an InitialPose or None"
            )
        object.__setattr__(self, "observation_modalities", modalities)

    @classmethod
    def from_observation_profile(
        cls,
        profile: ObservationProfile,
        *,
        initial_pose: InitialPose | None = None,
    ) -> "TaskExecutionOptions":
        """Use one profile as the source for Runtime modality selection."""

        if not isinstance(profile, ObservationProfile):
            raise ValueError("profile must be an ObservationProfile")
        return cls(
            observation_modalities=profile.modalities,
            initial_pose=initial_pose,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return resolved options for human/debug audit artifacts."""

        return {
            "observation_modalities": list(self.observation_modalities),
            "initial_pose": (
                self.initial_pose.to_dict()
                if self.initial_pose is not None
                else None
            ),
        }
