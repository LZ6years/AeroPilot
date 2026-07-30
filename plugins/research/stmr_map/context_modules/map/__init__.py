"""Expose the online STMR snapshot through the existing ``map`` context slot.

The module deliberately keeps perception and voxel diagnostics out of the
planner payload.  It preserves the default recalled-map clues, adds one compact
STMR summary, and supplies one labelled PNG through ``model_inputs`` so the
shared planner adapter places it before the trailing JSON context.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.modules.map import DefaultMapContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput

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

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
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
        return _planner_fragment(baseline, result)

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Default recalled map clues plus one compact online STMR semantic-history snapshot.",
            "semantic_evidence": [
                "STMR semantic history: accumulated semantics and UAV trajectory; not verified free space or exact target pose."
            ],
            "content_fields": {
                "recalled_map_memories": "Existing map-memory clues preserved from the default map provider.",
                "recall_filter": "Existing map/scene recall filter.",
                "stmr": "Compact current snapshot status, center pose, grid configuration, evidence counts, and semantic cell counts.",
                "stmr.trajectory_summary": "Compact numeric actual-motion summary; the full ordered trace remains debug-only.",
                "stmr.representation": "Always semantic_history_not_traversability; blank cells are not verified free space.",
            },
            "model_inputs": "At most one labelled STMR PNG, ordered after existing RGB inputs and before context JSON.",
            "risk_fields": {
                "stmr_map_unavailable": "The online STMR update failed; only default recalled-map clues remain.",
                "stmr_map_stale": "At least one newer observation failed, so the latest successful map may be stale.",
            },
        }

    def debug_summary(self) -> dict[str, Any]:
        """Return human-facing diagnostics without rebuilding context."""

        return self.online_builder.debug_summary()


def _planner_fragment(baseline: ContextFragment, result: StmrOnlineMapResult) -> ContextFragment:
    model_input = _stmr_model_input(result.png_path)
    risks = list(baseline.risks)
    if result.stale:
        risks.append({"type": "stmr_map_stale", "source": PluginContextModule.__name__})
    artifact_refs = [*baseline.artifact_refs]
    for path in (result.png_path, result.json_path):
        if path is not None:
            artifact_refs.append(str(path))
    return ContextFragment(
        name="map",
        enabled=True,
        source=PluginContextModule.__name__,
        content={**baseline.content, "stmr": result.context_content()},
        model_inputs=[model_input] if model_input is not None else [],
        facts=list(baseline.facts),
        inferences=list(baseline.inferences),
        goals=list(baseline.goals),
        risks=risks,
        confidence="medium" if result.status == "ready" and not result.stale else "low",
        artifact_refs=artifact_refs,
        budget=BudgetSpec(priority=PluginContextModule.priority, tokens_estimate=340, max_items=1),
        warnings=sorted({*baseline.warnings, *result.warnings}),
        metadata={
            **baseline.metadata,
            "stmr_mode": "planner",
            "stmr_png_attached": model_input is not None,
        },
    )


def _fallback_fragment(baseline: ContextFragment, exc: Exception) -> ContextFragment:
    return ContextFragment(
        name="map",
        enabled=True,
        source=PluginContextModule.__name__,
        content=dict(baseline.content),
        model_inputs=[],
        facts=list(baseline.facts),
        inferences=list(baseline.inferences),
        goals=list(baseline.goals),
        risks=[
            *baseline.risks,
            {
                "type": "stmr_map_unavailable",
                "value": {"error_type": type(exc).__name__},
                "source": PluginContextModule.__name__,
            },
        ],
        confidence="low",
        artifact_refs=list(baseline.artifact_refs),
        budget=BudgetSpec(priority=PluginContextModule.priority, tokens_estimate=240, max_items=1),
        warnings=[*baseline.warnings, f"stmr_map_failed:{type(exc).__name__}"],
        metadata={**baseline.metadata, "stmr_mode": "planner", "stmr_png_attached": False},
    )


def _stmr_model_input(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return {
        "type": "image",
        "input_id": STMR_MAP_INPUT_ID,
        "role": "stmr_semantic_map",
        "usage": STMR_MAP_USAGE,
        "mime_type": "image/png",
        "encoding": "data_url",
        "data": f"data:image/png;base64,{encoded}",
    }


__all__ = ["PluginContextModule", "STMR_MAP_INPUT_ID", "STMR_MAP_USAGE"]
