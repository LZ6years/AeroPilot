"""Plugin context module that builds a model-facing custom UAV map.

This module composes run memory into compact spatial evidence. It does not own
flight execution, safety decisions, or raw action/pose history; it only renders
and summarizes evidence for the planner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput

from .grid_builder import DEFAULT_GRID_SIZE, DEFAULT_METERS_PER_CELL, build_grid_map
from .html_report import render_custom_map_html
from .mask_semantics import build_semantic_layer
from .model_inputs import custom_map_model_input
from .projection import build_projected_semantic_layer
from .records import read_custom_map_records
from .renderer import render_custom_map
from .trajectory import build_trajectory


class PluginContextModule(ContextModule):
    """Build a custom_map fragment from run-local UAV memory."""

    name = "custom_map"
    priority = 82
    max_items = 1

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        memory = sources.flight_memory
        records = read_custom_map_records(memory)
        trajectory = build_trajectory(records)
        center_pose = trajectory.path[-1] if trajectory.path else None
        semantic_layer = build_projected_semantic_layer(
            records,
            center_pose,
            grid_size=DEFAULT_GRID_SIZE,
            meters_per_cell=DEFAULT_METERS_PER_CELL,
        ) or build_semantic_layer(records, grid_size=DEFAULT_GRID_SIZE)
        grid_map = build_grid_map(
            trajectory,
            semantic_layer,
            grid_size=DEFAULT_GRID_SIZE,
            meters_per_cell=DEFAULT_METERS_PER_CELL,
        )
        output_dir = _output_dir(records.run_dir)
        rendered_path = render_custom_map(grid_map, output_dir)
        html_path = render_custom_map_html(grid_map, rendered_path, output_dir)
        model_input = custom_map_model_input(rendered_path)
        warnings = [*records.warnings, *grid_map.warnings]
        risks = [] if trajectory.path else [{"type": "custom_map_pose_missing", "source": self.__class__.__name__}]

        content = {
            "semantic_matrix": grid_map.semantic_matrix,
            "semantic_legend": _string_keyed_legend(grid_map.semantic_legend),
            "path_summary": grid_map.path_summary,
            "center_pose": grid_map.center_pose,
            "map_config": grid_map.map_config,
            "rendered_map": {
                "role": "custom_map",
                "artifact_path": str(rendered_path),
                "model_input_attached": model_input is not None,
            },
            "rendered_html": {
                "artifact_path": str(html_path),
                "role": "custom_map_debug_viewer",
            },
            "projection_debug": grid_map.projection_debug,
            "path_cells": grid_map.path_cells,
            "failed_cells": grid_map.failed_cells,
            "current_cell": grid_map.current_cell,
        }
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content=content,
            model_inputs=[model_input] if model_input else [],
            risks=risks,
            confidence="medium" if trajectory.path else "low",
            artifact_refs=[str(rendered_path), str(html_path)],
            budget=BudgetSpec(priority=self.priority, tokens_estimate=420, max_items=1),
            warnings=sorted(set(warnings)),
            metadata={
                "record_counts": {
                    "actions": len(records.actions),
                    "poses": len(records.poses),
                    "frames": len(records.frames),
                },
                "output_dir": str(output_dir),
                "semantic_status": semantic_layer.status,
                "html_report": str(html_path),
            },
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Rendered UAV-centered spatial map plus compact semantic matrix and trajectory summary.",
            "semantic_evidence": [
                "Local semantic map: UAV-centered semantic cells and trajectory; not current visual truth or verified free space."
            ],
            "content_fields": {
                "semantic_matrix": "Local grid centered on the UAV. -1 is visited path, 0 is free, -2 is unknown, positive values refer to semantic_legend ids.",
                "semantic_legend": "Semantic id to label mapping. Unknown mask colors collapse to -2 unknown.",
                "path_summary": "Compressed run history summary derived from action and pose memory.",
                "center_pose": "Latest UAV pose used as the local map center.",
                "map_config": "Grid size, meters per cell, orientation, and semantic projection status.",
                "rendered_map": "Artifact path and planner input status for the generated map image.",
                "rendered_html": "Human-facing standalone debug viewer artifact path.",
            },
            "model_inputs": "One rendered PNG image with role custom_map when artifact generation succeeds.",
            "risk_fields": {
                "custom_map_pose_missing": "No pose history was available, so the map cannot localize the UAV.",
            },
        }


def _output_dir(run_dir: Path | None) -> Path:
    root = run_dir if run_dir is not None else Path.cwd()
    return root / "contexts" / "custom_map"


def _string_keyed_legend(legend: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(key): dict(value) for key, value in sorted(legend.items())}
