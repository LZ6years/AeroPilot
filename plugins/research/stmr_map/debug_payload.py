"""Build lightweight STMR debug payloads for browser-friendly reports.

The STMR research path can produce hundreds of thousands of projected points.
Those are valuable for offline diagnosis, but embedding them directly in HTML
makes the browser parse and draw giant JSON blobs. This module keeps the full
debug data available on demand while producing stratified previews for the
interactive viewer.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any


DEFAULT_HTML_STMR_POINT_BUDGET = 8_000
DEFAULT_HTML_CUSTOM_POINT_BUDGET = 0
DEFAULT_HTML_VOXEL_BUDGET = 4_000

FULL_POINT_KEYS = {"sampled_points", "voxel_world_centers"}


def projection_debug_summary(projection_debug: dict[str, Any]) -> dict[str, Any]:
    """Return projection debug without heavy point arrays."""

    summary = {}
    for key, value in projection_debug.items():
        if key in FULL_POINT_KEYS:
            continue
        summary[key] = value
    summary["sampled_points_total"] = int(
        projection_debug.get("sampled_points_total", len(projection_debug.get("sampled_points") or []))
    )
    summary["voxel_world_centers_total"] = int(
        projection_debug.get("voxel_world_centers_total", len(projection_debug.get("voxel_world_centers") or []))
    )
    return summary


def build_points_preview(
    *,
    stmr_projection_debug: dict[str, Any],
    custom_map: dict[str, Any] | None,
    stmr_point_budget: int = DEFAULT_HTML_STMR_POINT_BUDGET,
    custom_point_budget: int = DEFAULT_HTML_CUSTOM_POINT_BUDGET,
    voxel_budget: int = DEFAULT_HTML_VOXEL_BUDGET,
) -> dict[str, Any]:
    """Build stratified point subsets for the HTML viewer."""

    stmr_points = stmr_projection_debug.get("sampled_points") or []
    stmr_voxels = stmr_projection_debug.get("voxel_world_centers") or []
    custom_points = (((custom_map or {}).get("projection_debug") or {}).get("sampled_points") or [])
    stmr_points_preview = stratified_sample_points(stmr_points, stmr_point_budget)
    stmr_voxels_preview = stratified_sample_points(stmr_voxels, voxel_budget)
    custom_points_preview = stratified_sample_points(custom_points, custom_point_budget)
    return {
        "sampling_strategy": "stratified_by_semantic_id_and_observation_id_v1",
        "stmr_points": stmr_points_preview,
        "stmr_voxels": stmr_voxels_preview,
        "custom_points": custom_points_preview,
        "counts": {
            "stmr_points_total": int(stmr_projection_debug.get("sampled_points_total", len(stmr_points))),
            "stmr_points_displayed": len(stmr_points_preview),
            "stmr_voxels_total": int(stmr_projection_debug.get("voxel_world_centers_total", len(stmr_voxels))),
            "stmr_voxels_displayed": len(stmr_voxels_preview),
            "custom_points_total": len(custom_points),
            "custom_points_displayed": len(custom_points_preview),
        },
        "budgets": {
            "stmr_point_budget": int(stmr_point_budget),
            "stmr_voxel_budget": int(voxel_budget),
            "custom_point_budget": int(custom_point_budget),
        },
    }


def stratified_sample_points(points: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Sample points by semantic and observation strata with deterministic spacing."""

    budget = max(0, int(budget))
    if budget <= 0 or not points:
        return []
    if len(points) <= budget:
        return list(points)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        semantic_id = str(point.get("semantic_id", "unknown"))
        observation_id = str(point.get("source_observation_id") or _first_observation_id(point) or "unknown")
        groups[(semantic_id, observation_id)].append(point)

    base_quota = max(1, budget // max(1, len(groups)))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for group_points in groups.values():
        take = min(base_quota, len(group_points))
        selected.extend(_evenly_spaced(group_points, take))
        if len(group_points) > take:
            leftovers.extend(group_points[take:])

    remaining = budget - len(selected)
    if remaining > 0 and leftovers:
        selected.extend(_evenly_spaced(leftovers, remaining))
    return selected[:budget]


def slim_custom_map_for_html(custom_map: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only custom_map fields needed by the STMR compare viewer."""

    if custom_map is None:
        return None
    projection_debug = custom_map.get("projection_debug") or {}
    return {
        "center_pose": custom_map.get("center_pose"),
        "semantic_legend": custom_map.get("semantic_legend"),
        "map_config": custom_map.get("map_config"),
        "projection_debug": projection_debug_summary(projection_debug),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / float(count - 1)
    return [items[int(round(index * step))] for index in range(count)]


def _first_observation_id(point: dict[str, Any]) -> str | None:
    observation_ids = point.get("observation_ids")
    if isinstance(observation_ids, list) and observation_ids:
        return str(observation_ids[0])
    return None
