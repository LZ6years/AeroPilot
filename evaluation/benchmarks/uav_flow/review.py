"""Map UAV-Flow evaluation results into Benchmark Core review sections.

This module owns UAV-Flow-specific metric and STMR presentation. Runtime audit
reading, planner diagnostics, frame resolution, HTML, and CSS remain in
``evaluation.benchmarks.core``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evaluation.benchmarks.core import ReviewSection


def build_uav_flow_review_sections(
    *,
    case_dir: str | Path,
    ndtw_result: dict[str, Any],
    sr_result: dict[str, Any],
    visual_diagnostics: dict[str, Any],
    stmr_summary: dict[str, Any],
    plot_result: dict[str, Any],
    turn_diagnostics: dict[str, Any] | None = None,
    backend_action_tracking: dict[str, Any] | None = None,
    scenario_diagnostics: dict[str, Any] | None = None,
) -> tuple[ReviewSection, ...]:
    """Build typed UAV-Flow panels without injecting benchmark-owned HTML."""

    sections: list[ReviewSection] = []
    interpretation = ndtw_result.get("metric_interpretation")
    interpretation = (
        interpretation if isinstance(interpretation, dict) else {}
    )
    if interpretation.get("summary"):
        sections.append(
            ReviewSection(
                "Metric Interpretation",
                "text",
                interpretation["summary"],
            )
        )
    if scenario_diagnostics:
        sections.append(
            ReviewSection(
                "Scenario Contract Diagnostics",
                "key_value",
                scenario_diagnostics,
            )
        )
    if turn_diagnostics:
        trajectory_diagnostics = dict(turn_diagnostics)
        turn_tracking = trajectory_diagnostics.pop(
            "backend_turn_tracking", None
        )
        sections.append(
            ReviewSection(
                "Turn Trajectory Diagnostics",
                "key_value",
                trajectory_diagnostics,
            )
        )
        if isinstance(turn_tracking, dict):
            tracking_summary = {
                key: value
                for key, value in turn_tracking.items()
                if key != "samples"
            }
            sections.append(
                ReviewSection(
                    "Backend Turn Tracking",
                    "key_value",
                    tracking_summary,
                )
            )
            sections.append(
                ReviewSection(
                    "Backend Turn Tracking Samples",
                    "table",
                    [
                        dict(item)
                        for item in turn_tracking.get("samples", [])
                        if isinstance(item, dict)
                    ],
                )
            )

    if backend_action_tracking:
        tracking_summary = {
            key: value
            for key, value in backend_action_tracking.items()
            if key != "windows"
        }
        sections.append(
            ReviewSection(
                "Physical Window Tracking",
                "key_value",
                tracking_summary,
            )
        )
        sections.append(
            ReviewSection(
                "Physical Window Tracking Details",
                "table",
                [
                    dict(item)
                    for item in backend_action_tracking.get("windows", [])
                    if isinstance(item, dict)
                ],
            )
        )

    sections.extend(
        (
            ReviewSection(
                "Visual Diagnostics",
                "key_value",
                visual_diagnostics,
            ),
            ReviewSection("SR Judge", "key_value", sr_result),
        )
    )

    if stmr_summary.get("mode") not in (None, "off"):
        stmr_assets, asset_diagnostics = _materialize_stmr_assets(
            case_dir,
            stmr_summary,
        )
        sections.append(
            ReviewSection(
                "STMR",
                "key_value",
                _compact_stmr_summary(
                    stmr_summary,
                    asset_diagnostics=asset_diagnostics,
                ),
            )
        )
        if stmr_assets:
            sections.append(
                ReviewSection(
                    "STMR Render History",
                    "image_group",
                    stmr_assets,
                )
            )
        profiling_rows = _stmr_profiling_rows(stmr_summary)
        if profiling_rows:
            sections.append(
                ReviewSection(
                    "STMR Profiling",
                    "table",
                    profiling_rows,
                )
            )

    plot_assets = [
        {
            "path": value,
            "label": (
                "Executed vs reference trajectory (2D)"
                if key == "trajectory_2d"
                else "Executed vs reference trajectory (3D)"
            ),
        }
        for key, value in plot_result.items()
        if key in {"trajectory_2d", "trajectory_3d"} and value
    ]
    sections.append(
        ReviewSection("Trajectory Plots", "image_group", plot_assets, primary=True)
    )
    return tuple(sections)


def _materialize_stmr_assets(
    case_dir: str | Path,
    stmr: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Copy report-only STMR images beside the portable case review."""

    candidates = _stmr_image_candidates(stmr)
    destination = Path(case_dir) / "review_assets" / "stmr"
    assets: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    copied_sources: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        source = Path(str(candidate.get("path") or ""))
        source_key = str(source.resolve()) if source.exists() else str(source)
        if source_key in copied_sources:
            continue
        copied_sources.add(source_key)
        if not source.is_file():
            diagnostics.append(
                {
                    "stage": "stmr_asset_copy",
                    "path": str(source),
                    "reason": "file_missing",
                }
            )
            continue
        try:
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / (
                f"{index:04d}_{_safe_name(source.stem)}"
                f"{source.suffix.lower()}"
            )
            shutil.copy2(source, target)
            assets.append(
                {
                    "path": str(target),
                    "label": candidate.get("label") or source.name,
                    "description": candidate.get("description"),
                }
            )
        except OSError as exc:
            diagnostics.append(
                {
                    "stage": "stmr_asset_copy",
                    "path": str(source),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return assets, diagnostics


def _stmr_image_candidates(stmr: dict[str, Any]) -> list[dict[str, Any]]:
    history = stmr.get("render_history")
    history = history if isinstance(history, dict) else {}
    manifest_path = _existing_path(history.get("manifest_path"))
    if manifest_path is None:
        json_path = _existing_path(stmr.get("json_path"))
        candidate = (
            json_path.parent / "render_history.json"
            if json_path is not None
            else None
        )
        if candidate is not None and candidate.is_file():
            manifest_path = candidate

    snapshots: list[dict[str, Any]] = []
    manifest = _read_json(manifest_path)
    values = manifest.get("snapshots")
    values = values if isinstance(values, list) else []
    for item in values:
        if not isinstance(item, dict) or not item.get("png_path"):
            continue
        snapshots.append(
            {
                "path": item["png_path"],
                "label": f"STMR step {item.get('sequence')}",
                "description": (
                    f"observation={item.get('observation_id')}; "
                    f"processed={item.get('processed_observation_count')}; "
                    f"points={item.get('point_count')}; "
                    f"voxels={item.get('voxel_count')}"
                ),
            }
        )
    latest = stmr.get("png_path")
    if latest:
        snapshots.append(
            {
                "path": latest,
                "label": "Final STMR map",
                "description": (
                    "Final accumulated semantic map for human review."
                ),
            }
        )
    return snapshots


def _compact_stmr_summary(
    stmr: dict[str, Any],
    *,
    asset_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": stmr.get("mode"),
        "status": stmr.get("status"),
        "processed_observation_count": stmr.get(
            "processed_observation_count"
        ),
        "failed_observation_count": stmr.get("failed_observation_count"),
        "point_count": stmr.get("point_count"),
        "voxel_count": stmr.get("voxel_count"),
        "semantic_cell_counts": stmr.get("semantic_cell_counts"),
        "batch_incremental_parity": (
            stmr.get("batch_incremental_parity") or {}
        ).get("equivalent"),
        "render_snapshot_count": (
            stmr.get("render_history") or {}
        ).get("snapshot_count"),
        "asset_diagnostics": asset_diagnostics,
    }


def _stmr_profiling_rows(stmr: dict[str, Any]) -> list[dict[str, Any]]:
    profiling = stmr.get("profiling")
    profiling = profiling if isinstance(profiling, dict) else {}
    stages = profiling.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    rows = []
    for name, value in stages.items():
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "stage": name,
                "sample_count": value.get("sample_count"),
                "p50_ms": value.get("p50_ms"),
                "p95_ms": value.get("p95_ms"),
            }
        )
    return rows


def _existing_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_name(value: str) -> str:
    text = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in value
    ).strip("_")
    return text or "stmr"
