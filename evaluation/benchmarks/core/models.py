"""Immutable benchmark episode input passed to the shared executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from src.env_client import ObservationProfile
from src.runtime.configs import TaskExecutionOptions


ScenarioPayloadT = TypeVar("ScenarioPayloadT")
EvaluationPayloadT = TypeVar("EvaluationPayloadT")


@dataclass(frozen=True)
class PreparedBenchmarkEpisode(Generic[ScenarioPayloadT, EvaluationPayloadT]):
    """Describe one episode without exposing evaluation truth to Runtime."""

    benchmark_name: str
    case_id: str
    protocol_variant: str
    instruction: str
    map_id: str
    scene_id: str | None
    agent_id: str
    observation_profile: ObservationProfile
    execution_options: TaskExecutionOptions
    planner_safe_metadata: dict[str, Any] = field(default_factory=dict)
    scenario_payload: ScenarioPayloadT | None = None
    evaluation_payload: EvaluationPayloadT | None = None
    protocol_context: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "benchmark_name",
            "case_id",
            "protocol_variant",
            "instruction",
            "map_id",
            "agent_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"PreparedBenchmarkEpisode {name} cannot be empty")
        if not isinstance(self.observation_profile, ObservationProfile):
            raise ValueError("observation_profile must be an ObservationProfile")
        if not isinstance(self.execution_options, TaskExecutionOptions):
            raise ValueError("execution_options must be TaskExecutionOptions")
        if (
            tuple(self.observation_profile.modalities)
            != self.execution_options.observation_modalities
        ):
            raise ValueError(
                "ObservationProfile and TaskExecutionOptions modalities must match"
            )
        object.__setattr__(
            self, "planner_safe_metadata", dict(self.planner_safe_metadata)
        )

    def planner_manifest(self) -> dict[str, Any]:
        """Return the only benchmark data allowed in a Runtime run directory."""

        return {
            "benchmark_name": self.benchmark_name,
            "case_id": self.case_id,
            "protocol_variant": self.protocol_variant,
            "map_id": self.map_id,
            "scene_id": self.scene_id,
            "agent_id": self.agent_id,
            "observation_profile": self.observation_profile.to_dict(),
            "execution_options": self.execution_options.to_dict(),
            "metadata": dict(self.planner_safe_metadata),
        }
