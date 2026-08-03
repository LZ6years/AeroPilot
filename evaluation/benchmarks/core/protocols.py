"""Structural interfaces implemented by individual benchmark adapters."""

from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar

from evaluation.benchmarks.core.models import PreparedBenchmarkEpisode
from evaluation.benchmarks.core.trace import EpisodeTrace
from src.env_client import SceneInstance


RawCaseT = TypeVar("RawCaseT")
ScenarioPayloadT = TypeVar("ScenarioPayloadT")
EvaluationPayloadT = TypeVar("EvaluationPayloadT")


class BenchmarkCaseAdapter(
    Protocol, Generic[RawCaseT, ScenarioPayloadT, EvaluationPayloadT]
):
    """Convert one dataset record into the shared episode contract."""

    def prepare_case(
        self, case: RawCaseT
    ) -> PreparedBenchmarkEpisode[ScenarioPayloadT, EvaluationPayloadT]:
        ...


class BenchmarkScenarioController(Protocol, Generic[ScenarioPayloadT]):
    """Manage benchmark-only scene state without controlling the UAV."""

    def prepare_episode(
        self, scene: SceneInstance, payload: ScenarioPayloadT | None
    ) -> None:
        ...

    def cleanup_episode(
        self, scene: SceneInstance, payload: ScenarioPayloadT | None
    ) -> None:
        ...


class BenchmarkMetricSuite(Protocol, Generic[EvaluationPayloadT]):
    """Evaluate one frozen Runtime trace against benchmark-owned truth."""

    def evaluate(
        self,
        payload: EvaluationPayloadT | None,
        trace: EpisodeTrace,
        outcome: Any,
    ) -> dict[str, Any]:
        ...


class NoopScenarioController:
    """Scenario controller for datasets whose scene is already complete."""

    def prepare_episode(self, scene: SceneInstance, payload: Any) -> None:
        return

    def cleanup_episode(self, scene: SceneInstance, payload: Any) -> None:
        return
