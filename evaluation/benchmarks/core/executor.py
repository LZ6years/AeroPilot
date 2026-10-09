"""Execute benchmark episodes around the existing TaskRuntime lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.benchmarks.core.models import PreparedBenchmarkEpisode
from evaluation.benchmarks.core.audit import EpisodeAuditCursor
from evaluation.benchmarks.core.protocols import (
    BenchmarkMetricSuite,
    BenchmarkScenarioController,
)
from evaluation.benchmarks.core.trace import (
    EpisodeTrace,
    EpisodeTraceReader,
    TraceCursor,
)


@dataclass(frozen=True)
class BenchmarkExecutionOutcome:
    """Runtime completion facts separated from metric interpretation."""

    completed: bool = False
    stopped: bool = False
    stopped_reason: str | None = None
    stop_source: str | None = None
    error: dict[str, str] | None = None
    cleanup_errors: tuple[dict[str, str], ...] = ()

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        if self.stopped:
            return "stopped"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "completed": self.completed,
            "stopped": self.stopped,
            "stopped_reason": self.stopped_reason,
            "stop_source": self.stop_source,
            "error": dict(self.error) if self.error else None,
            "cleanup_errors": [dict(item) for item in self.cleanup_errors],
        }


@dataclass(frozen=True)
class BenchmarkEpisodeResult:
    """Complete shared-executor result for one benchmark case."""

    outcome: BenchmarkExecutionOutcome
    cursor: TraceCursor
    audit_cursor: EpisodeAuditCursor
    trace: EpisodeTrace
    metrics: dict[str, Any]
    scene_capabilities: dict[str, Any] | None = None
    metric_error: dict[str, str] | None = None


class BenchmarkEpisodeExecutor:
    """Enforce scene ownership, trace freezing, cleanup, then evaluation."""

    def __init__(self, trace_reader: EpisodeTraceReader | None = None) -> None:
        self.trace_reader = trace_reader or EpisodeTraceReader()

    def execute(
        self,
        episode: PreparedBenchmarkEpisode,
        *,
        runtime: Any,
        scenario_controller: BenchmarkScenarioController[Any],
        metric_suite: BenchmarkMetricSuite[Any],
        run_dir: str | Path,
    ) -> BenchmarkEpisodeResult:
        run_path = Path(run_dir)
        _write_planner_manifest(run_path, episode)
        scene = None
        state = None
        error: dict[str, str] | None = None
        cleanup_errors: list[dict[str, str]] = []
        prepare_started = False
        cursor = TraceCursor.freeze(run_path)
        audit_cursor = EpisodeAuditCursor.freeze(run_path)
        try:
            scene = runtime.ensure_reused_scene(episode.map_id, episode.scene_id)
            prepare_started = True
            scenario_controller.prepare_episode(
                scene, episode.scenario_payload
            )
            try:
                state = runtime.run_task(
                    episode.map_id,
                    episode.instruction,
                    agent_id=episode.agent_id,
                    scene_policy="reuse_running",
                    stop_scene_on_finish=False,
                    scene_id=episode.scene_id,
                    execution_options=episode.execution_options,
                    protocol_context=episode.protocol_context,
                )
            except Exception as exc:
                error = _error_payload(exc)
                cleanup_errors.extend(getattr(exc, "cleanup_errors", ()))
        except Exception as exc:
            error = _error_payload(exc)
            cleanup_errors.extend(getattr(exc, "cleanup_errors", ()))
        finally:
            # Both cursors are frozen before Runtime shutdown so reports and
            # benchmark trajectories exclude cleanup hover/land observations.
            cursor = TraceCursor.freeze(run_path)
            audit_cursor = EpisodeAuditCursor.freeze(run_path)
            if scene is not None:
                try:
                    runtime.finish_reused_scene(
                        episode.map_id,
                        state,
                        agent_id=episode.agent_id,
                        scene_id=episode.scene_id,
                    )
                except Exception as exc:
                    cleanup_errors.extend(getattr(exc, "cleanup_errors", ()))
                    cleanup_errors.append(
                        {"stage": "runtime_shutdown", **_error_payload(exc)}
                    )
                if prepare_started:
                    try:
                        scenario_controller.cleanup_episode(
                            scene, episode.scenario_payload
                        )
                    except Exception as exc:
                        cleanup_errors.append(
                            {"stage": "scenario_cleanup", **_error_payload(exc)}
                        )

        outcome = _outcome_from_state(state, error, cleanup_errors)
        trace = self.trace_reader.read(cursor)
        metrics: dict[str, Any] = {}
        metric_error = None
        try:
            metrics = metric_suite.evaluate(
                episode.evaluation_payload, trace, outcome
            )
        except Exception as exc:
            metric_error = _error_payload(exc)
        capabilities = (
            scene.capabilities.__dict__.copy()
            if scene is not None and scene.capabilities is not None
            else None
        )
        return BenchmarkEpisodeResult(
            outcome=outcome,
            cursor=cursor,
            audit_cursor=audit_cursor,
            trace=trace,
            metrics=metrics,
            scene_capabilities=capabilities,
            metric_error=metric_error,
        )


def _outcome_from_state(
    state: Any,
    error: dict[str, str] | None,
    cleanup_errors: list[dict[str, str]],
) -> BenchmarkExecutionOutcome:
    return BenchmarkExecutionOutcome(
        completed=bool(getattr(state, "completed", False)),
        stopped=bool(getattr(state, "stopped", False)),
        stopped_reason=getattr(state, "stopped_reason", None),
        stop_source=getattr(state, "stop_source", None),
        error=error,
        cleanup_errors=tuple(cleanup_errors),
    )


def _error_payload(exc: Exception) -> dict[str, Any]:
    exc = getattr(exc, "original_error", exc)
    from src.llm import ProviderRequestError
    payload = {"error_type": type(exc).__name__, "error": str(exc)}
    if isinstance(exc, ProviderRequestError):
        payload.update(exc.to_dict())
    return payload


def _write_planner_manifest(
    run_dir: Path, episode: PreparedBenchmarkEpisode
) -> None:
    path = run_dir / "benchmark" / "episode_planner_safe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(episode.planner_manifest(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
