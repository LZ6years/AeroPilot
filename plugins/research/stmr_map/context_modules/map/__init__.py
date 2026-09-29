"""Expose the online STMR snapshot through the existing ``map`` context slot.

The module deliberately keeps perception and voxel diagnostics out of the
planner payload.  It preserves the default recalled-map clues, adds one compact
STMR summary, and supplies one labelled PNG through ``model_inputs`` so the
shared planner adapter places it before the trailing JSON context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.modules.map import DefaultMapContextModule
from src.memory.context.contracts import (
    ContextFragment,
    ContextModuleResult,
    FragmentAudit,
    MemorySourceBundle,
    TaskContextInput,
)
from src.memory.context.media_refs import image_media_ref

from ...online import StmrOnlineMapBuilder, StmrOnlineMapResult
from ...perception import DEFAULT_UAV_PRIOR_NAME, MAX_LANDMARK_QUERIES, StmrPerceptionSession


STMR_MAP_INPUT_ID = "map_1_stmr_semantic_history"
STMR_MAP_USAGE = (
    "HISTORICAL STMR SEMANTIC MAP. "
    "Orange trace runs START/ENTRY to NOW; arrowheads point toward newer actual poses; "
    "green triangle is current UAV heading and image top is current forward. "
    "Use for localization and accumulated semantic relations only; "
    "not current visibility or verified traversability."
)


class PluginContextModule(ContextModule):
    """Stateful STMR provider installed in the standard ``map`` slot."""

    name = "map"
    priority = 78
    max_items = 1

    def __init__(
        self,
        *,
        mode: str = "planner",
        session: StmrPerceptionSession | None = None,
        prior: str = DEFAULT_UAV_PRIOR_NAME,
        max_queries: int = MAX_LANDMARK_QUERIES,
        device: str = "auto",
        online_builder: StmrOnlineMapBuilder | None = None,
    ) -> None:
        if mode not in {"shadow", "planner"}:
            raise ValueError(f"Unsupported STMR context mode: {mode}")
        self.mode = mode
        self.default_map = DefaultMapContextModule()
        self.online_builder = online_builder or StmrOnlineMapBuilder(
            session=session,
            prior=prior,
            max_queries=max_queries,
            device=device,
        )

    def build(self, sources: MemorySourceBundle, stage_context: TaskContextInput) -> ContextModuleResult:
        baseline = self.default_map.build(sources, stage_context)
        try:
            result = self.online_builder.build(sources.flight_memory, stage_context)
        except Exception as exc:
            # Context construction is on the critical flight path.  A research
            # map failure must never prevent the default context from reaching
            # the planner.
            if self.mode == "shadow":
                return baseline
            return _fallback_fragment(baseline, exc)

        if self.mode == "shadow":
            return baseline
        observation_id = "stmr"
        if sources.flight_memory.recent_observation_refs:
            observation_id = str(
                sources.flight_memory.recent_observation_refs[-1].get("observation_id")
                or observation_id
            )
        return _planner_fragment(
            baseline,
            result,
            observation_id,
            run_dir=getattr(sources.flight_memory.memory_reader, "run_dir", None),
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Default recalled map clues plus one compact online STMR semantic-history snapshot.",
            "semantic_evidence": [
                "STMR semantic history: accumulated semantics and UAV trajectory; not verified free space or exact target pose."
            ],
            "model_fields": {
                "recalled_map_memories": "Existing map-memory clues preserved from the default map provider.",
                "recall_filter": "Existing map/scene recall filter.",
                "stmr": "Compact current snapshot status, center pose, grid configuration, evidence counts, and semantic cell counts.",
                "stmr.trajectory_summary": "Compact numeric actual-motion summary; the full ordered trace remains debug-only.",
                "stmr.representation": "Always semantic_history_not_traversability; blank cells are not verified free space.",
            },
            "model_inputs": "At most one labelled STMR PNG, ordered after existing RGB inputs and before context JSON.",
            "audit_fields": {
                "stmr_map_unavailable": "The online STMR update failed; only default recalled-map clues remain.",
                "stmr_map_stale": "At least one newer observation failed, so the latest successful map may be stale.",
            },
        }

    def debug_summary(self) -> dict[str, Any]:
        """Return human-facing diagnostics without rebuilding context."""

        return self.online_builder.debug_summary()


def _planner_fragment(
    baseline: ContextModuleResult,
    result: StmrOnlineMapResult,
    observation_id: str,
    *,
    run_dir: str | Path | None,
) -> ContextModuleResult:
    model_input = _stmr_model_input(
        result.png_path,
        observation_id,
        run_dir=run_dir,
    )
    warnings = list(baseline.audit.warnings)
    if result.stale:
        warnings.append("stmr_map_stale")
    artifact_refs = [*baseline.audit.artifact_refs]
    for path in (result.png_path, result.json_path):
        if path is not None:
            artifact_refs.append(str(path))
    return ContextModuleResult(
        ContextFragment(
            name="map",
            model_content={**baseline.fragment.model_content, "stmr": result.context_content()},
            model_inputs=(model_input,) if model_input is not None else (),
        ),
        FragmentAudit(
            name="map",
            source=PluginContextModule.__name__,
            confidence="medium" if result.status == "ready" and not result.stale else "low",
            artifact_refs=tuple(artifact_refs),
            warnings=tuple(sorted({*warnings, *result.warnings})),
            metadata={
            **baseline.audit.metadata,
            "stmr_mode": "planner",
            "stmr_png_attached": model_input is not None,
            },
        ),
    )


def _fallback_fragment(baseline: ContextModuleResult, exc: Exception) -> ContextModuleResult:
    return ContextModuleResult(
        ContextFragment("map", dict(baseline.fragment.model_content)),
        FragmentAudit(
            name="map",
            source=PluginContextModule.__name__,
            confidence="low",
            artifact_refs=baseline.audit.artifact_refs,
            warnings=(*baseline.audit.warnings, f"stmr_map_failed:{type(exc).__name__}"),
            metadata={
                **baseline.audit.metadata,
                "stmr_mode": "planner",
                "stmr_png_attached": False,
                "error_type": type(exc).__name__,
            },
        ),
    )


def _stmr_model_input(
    path: Path | None,
    observation_id: str,
    *,
    run_dir: str | Path | None,
) -> dict[str, Any] | None:
    if path is None or run_dir is None:
        return None
    try:
        return image_media_ref(
            path,
            run_dir=run_dir,
            observation_id=observation_id,
            frame_id="stmr_map",
            role="stmr_semantic_map",
            usage=STMR_MAP_USAGE,
        ).to_model_input()
    except (OSError, ValueError):
        return None


__all__ = ["PluginContextModule", "STMR_MAP_INPUT_ID", "STMR_MAP_USAGE"]
