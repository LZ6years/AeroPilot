"""Run standalone STMR perception/projection on existing UAV run outputs.

This script is intentionally manual. It can download and run GroundingDINO/SAM
models, writes large debug artifacts, and does not participate in pytest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.research.custom_map.context_modules.custom_map.records import read_custom_map_records
from plugins.research.custom_map.context_modules.custom_map.schemas import CustomMapRecords, PosePoint
from plugins.research.custom_map.perception.semantic_palette import semantic_class_for_ade_label
from plugins.research.stmr_map.core import IncrementalStmrMapState, StmrMap, build_stmr_map
from plugins.research.stmr_map.debug_payload import (
    DEFAULT_HTML_CUSTOM_POINT_BUDGET,
    DEFAULT_HTML_STMR_POINT_BUDGET,
    DEFAULT_HTML_VOXEL_BUDGET,
    build_points_preview,
    projection_debug_summary,
    slim_custom_map_for_html,
    write_json,
)
from plugins.research.stmr_map.html_report import render_stmr_html
from plugins.research.stmr_map.perception import (
    DynamicTargetFilterConfig,
    ModelLandmarkExtractor,
    ModelLandmarkExtractorError,
    StmrPerceptionSession,
    build_dynamic_candidate,
    extract_landmarks,
    filter_detections_by_tfidf,
)
from plugins.research.stmr_map.perception.grounding_dino_backend import DEFAULT_GROUNDING_DINO_MODEL, DetectionResult
from plugins.research.stmr_map.perception.sam_backend import DEFAULT_SAM_MODEL, SamMaskResult
from plugins.research.stmr_map.perception.tfidf_filter import TfidfMatch
from plugins.research.stmr_map.profiling import StageProfiler, artifact_size_bytes
from plugins.research.stmr_map.renderer import render_stmr_map
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryRecordReader


REPRESENTATIVE_CLASSES = ("Move", "Shift", "Approach", "Pass", "Retreat")


def main() -> None:
    args = _parser().parse_args()
    runs = _select_runs(args)
    if not runs:
        raise SystemExit("No runs selected for STMR probe")
    session = _create_perception_session(args) if args.perception_source == "grounding_dino_sam" else None
    summaries = []
    for run in runs:
        summaries.append(run_stmr_probe(run, args, session=session))
    summary = {"run_count": len(summaries), "runs": summaries}
    root = _summary_root(args, runs)
    json_path = root / "stmr_compare_summary.json"
    html_path = root / "stmr_compare_summary.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_summary_html(summary, html_path.parent), encoding="utf-8")
    print(f"stmr summary json: {json_path}")
    print(f"stmr summary html: {html_path}")
    for item in summaries:
        print(f"{item['case_id']}: {item['stmr_html']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build standalone STMR artifacts for existing UAV run outputs")
    parser.add_argument("--run-dir", nargs="*", help="Explicit run directories")
    parser.add_argument("--sweep-dir", default="runs/experiments/custom_map_semantic_sweep")
    parser.add_argument("--preset", choices=["representative_5"], default="representative_5")
    parser.add_argument("--perception-source", choices=["grounding_dino_sam", "fake_boxes"], default="grounding_dino_sam")
    parser.add_argument("--grounding-model", default=DEFAULT_GROUNDING_DINO_MODEL)
    parser.add_argument("--sam-model", default=DEFAULT_SAM_MODEL)
    parser.add_argument("--seg-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--tfidf-threshold", type=float, default=0.8)
    parser.add_argument("--landmarks", nargs="*", default=[])
    parser.add_argument("--landmark-source", choices=["rule", "model", "manual"], default="rule")
    parser.add_argument("--max-landmarks", type=int, default=12)
    parser.add_argument("--dump-landmark-prompt", action="store_true")
    parser.add_argument("--allow-rule-fallback", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for smoke runs")
    parser.add_argument("--dynamic-target-filter", choices=["on", "off"], default="on")
    parser.add_argument("--dynamic-min-score", type=float, default=0.45)
    parser.add_argument("--dynamic-min-mask-pixels", type=int, default=100)
    parser.add_argument("--dynamic-max-depth-p95", type=float, default=35.0)
    parser.add_argument("--dynamic-track-radius-m", type=float, default=8.0)
    parser.add_argument("--html-stmr-point-budget", type=int, default=DEFAULT_HTML_STMR_POINT_BUDGET)
    parser.add_argument("--html-voxel-budget", type=int, default=DEFAULT_HTML_VOXEL_BUDGET)
    parser.add_argument("--html-custom-point-budget", type=int, default=DEFAULT_HTML_CUSTOM_POINT_BUDGET)
    parser.add_argument("--write-full-debug", action="store_true")
    return parser


def run_stmr_probe(
    run: dict[str, Any],
    args: argparse.Namespace,
    *,
    session: StmrPerceptionSession | None = None,
    profiler: StageProfiler | None = None,
    output_dir_override: str | Path | None = None,
    prior_name: str | None = None,
) -> dict[str, Any]:
    """Build one STMR case with optional shared models and stage profiling."""

    if profiler is None:
        return _run_stmr_probe_impl(
            run,
            args,
            session=session,
            profiler=None,
            output_dir_override=output_dir_override,
            prior_name=prior_name,
        )
    with profiler.stage("complete_pipeline_total") as stage:
        summary = _run_stmr_probe_impl(
            run,
            args,
            session=session,
            profiler=profiler,
            output_dir_override=output_dir_override,
            prior_name=prior_name,
        )
        stage.update(
            projected_point_count=summary.get("point_count", 0),
            voxel_count=summary.get("voxel_count", 0),
        )
        return summary


def _run_stmr_probe_impl(
    run: dict[str, Any],
    args: argparse.Namespace,
    *,
    session: StmrPerceptionSession | None,
    profiler: StageProfiler | None,
    output_dir_override: str | Path | None,
    prior_name: str | None,
) -> dict[str, Any]:
    run_dir = Path(str(run["run_dir"]))
    output_dir = Path(output_dir_override) if output_dir_override is not None else run_dir / "contexts" / "stmr_map"
    variant_dir = output_dir / "_variant_run"
    with _optional_stage(profiler, "variant_memory_prepare") as stage:
        _prepare_variant_memory(run_dir, variant_dir)
        stage.update(source_run_dir=str(run_dir), variant_dir=str(variant_dir))
    instruction = str(run.get("instruction") or _read_instruction(run_dir) or "")
    landmark_result = _extract_probe_landmarks(instruction, run, args, include_defaults=prior_name is None)
    landmarks = landmark_result["landmarks"]
    if prior_name is not None:
        active_session = session or _create_perception_session(args, profiler=profiler)
        landmarks = active_session.assemble_queries(
            landmarks,
            prior=prior_name,
            max_queries=max(1, int(args.max_landmarks)),
            profiler=profiler,
        )
        landmark_result = {**landmark_result, "landmarks": landmarks, "prior_name": prior_name}
        session = active_session
    perception_results = _run_open_vocab_perception(
        variant_dir,
        landmarks,
        args,
        session=session,
        profiler=profiler,
    )
    memory = FlightMemory(
        run_id=variant_dir.name,
        map_id="stmr",
        scene_instance_id="stmr",
        agent_id="uav_flow_drone",
        active_instruction=instruction,
        memory_reader=MemoryRecordReader(variant_dir),
    )
    with _optional_stage(profiler, "memory_record_load") as stage:
        records = read_custom_map_records(memory)
        stage.update(
            pose_count=len(records.poses),
            frame_count=len(records.frames),
        )
    subgoal_ids = _subgoal_semantic_ids([item.label for item in landmarks])
    stmr_map, incremental_summary = _build_incremental_stmr(
        records,
        perception_results,
        subgoal_semantic_ids=subgoal_ids,
        debug_point_limit=None if args.write_full_debug else args.html_stmr_point_budget,
        output_dir=output_dir,
        profiler=profiler,
    )
    with _optional_stage(profiler, "png_render") as stage:
        rendered_path = render_stmr_map(stmr_map, output_dir)
        stage.update(artifact_bytes=artifact_size_bytes(rendered_path))
    custom_map_json = run_dir / "contexts" / "custom_map" / "latest_custom_map.json"
    custom_map = _load_custom_map_content(custom_map_json)
    points_preview = build_points_preview(
        stmr_projection_debug=stmr_map.projection_debug,
        custom_map=custom_map,
        stmr_point_budget=args.html_stmr_point_budget,
        custom_point_budget=args.html_custom_point_budget,
        voxel_budget=args.html_voxel_budget,
    )
    points_preview_path = output_dir / "latest_stmr_map_points_preview.json"
    write_json(points_preview_path, {"content": points_preview})
    with _optional_stage(profiler, "html_render") as stage:
        html_path = render_stmr_html(
            stmr_map,
            rendered_path,
            output_dir,
            landmarks=[item.__dict__ for item in landmarks],
            landmark_debug={
                "landmark_source": landmark_result["source"],
                "prior_name": prior_name,
                "llm_landmark_model": landmark_result.get("model_summary"),
                "llm_landmark_raw_response": landmark_result.get("raw_response"),
                "llm_landmark_prompt": landmark_result.get("prompt") if args.dump_landmark_prompt else None,
            },
            perception_results=perception_results,
            custom_map_json=custom_map_json,
            points_preview=points_preview,
            pipeline_timing=profiler.summary() if profiler is not None else None,
        )
        stage.update(artifact_bytes=artifact_size_bytes(html_path))
    full_debug_path = output_dir / "latest_stmr_map_full_debug.json"
    if args.write_full_debug:
        write_json(
            full_debug_path,
            {
                "content": {
                    "projection_debug": stmr_map.projection_debug,
                    "custom_map_projection_debug": (custom_map or {}).get("projection_debug"),
                }
            },
        )
    elif full_debug_path.exists():
        full_debug_path.unlink()
    content = {
        "semantic_matrix": stmr_map.semantic_matrix,
        "semantic_legend": {str(key): dict(value) for key, value in sorted(stmr_map.semantic_legend.items())},
        "path_cells": stmr_map.path_cells,
        "trajectory_trace": stmr_map.trajectory_trace,
        "trajectory_summary": stmr_map.trajectory_summary,
        "current_cell": stmr_map.current_cell,
        "center_pose": stmr_map.center_pose,
        "map_config": stmr_map.map_config,
        "coordinate_frame": stmr_map.coordinate_frame,
        "projection_debug": projection_debug_summary(stmr_map.projection_debug),
        "source_frames": stmr_map.source_frames,
        "landmarks": [item.__dict__ for item in landmarks],
        "landmark_source": landmark_result["source"],
        "prior_name": prior_name,
        "llm_landmark_model": landmark_result.get("model_summary"),
        "llm_landmark_raw_response": landmark_result.get("raw_response"),
        "llm_landmark_prompt": landmark_result.get("prompt") if args.dump_landmark_prompt else None,
        "perception_results": perception_results,
        "rendered_map": {"artifact_path": str(rendered_path), "role": "stmr_map"},
        "rendered_html": {"artifact_path": str(html_path), "role": "stmr_debug_viewer"},
        "points_preview": {"artifact_path": str(points_preview_path), "role": "stmr_points_preview", "counts": points_preview.get("counts", {})},
        "full_debug": {"artifact_path": str(full_debug_path), "role": "stmr_full_debug", "written": bool(args.write_full_debug)},
        "custom_map": slim_custom_map_for_html(custom_map),
        "warnings": stmr_map.warnings,
        "incremental_update": incremental_summary,
    }
    json_path = output_dir / "latest_stmr_map.json"
    with _optional_stage(profiler, "json_render") as stage:
        json_path.write_text(json.dumps({"content": content}, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.update(artifact_bytes=artifact_size_bytes(json_path))
    return {
        "run_dir": str(run_dir),
        "case_id": str(run.get("case_id") or run_dir.name),
        "class_name": str(run.get("class_name") or run_dir.name.split("_", 1)[0]),
        "stmr_json": str(json_path),
        "stmr_png": str(rendered_path),
        "stmr_html": str(html_path),
        "landmarks": [item.label for item in landmarks],
        "landmark_source": landmark_result["source"],
        "prior_name": prior_name,
        "query_count": len(landmarks),
        "point_count": stmr_map.projection_debug.get("point_count", 0),
        "voxel_count": stmr_map.projection_debug.get("voxel_count", 0),
        "subgoal_priority_hit_count": len(stmr_map.projection_debug.get("subgoal_priority_hits", [])),
        "perception_frame_count": len(perception_results),
        "incremental_update": incremental_summary,
        "artifact_sizes": {
            "json": artifact_size_bytes(json_path),
            "png": artifact_size_bytes(rendered_path),
            "html": artifact_size_bytes(html_path),
        },
    }


def _build_incremental_stmr(
    records: CustomMapRecords,
    perception_results: list[dict[str, Any]],
    *,
    subgoal_semantic_ids: set[int],
    debug_point_limit: int | None,
    output_dir: Path,
    profiler: StageProfiler | None,
) -> tuple[StmrMap, dict[str, Any]]:
    """Replay complete observations through the process-local incremental state."""

    state = IncrementalStmrMapState()
    observation_ids = [
        str(item.get("observation_id"))
        for item in perception_results
        if item.get("observation_id")
    ]
    if not observation_ids:
        observation_ids = _complete_stmr_observation_ids(records)

    updates: list[dict[str, Any]] = []
    latest_records = records
    latest_map: StmrMap | None = None
    for index, observation_id in enumerate(observation_ids, start=1):
        latest_records = _records_through_observation(records, observation_ids[:index])
        if profiler is None:
            update = state.update(latest_records, [observation_id])
        else:
            frame_profiler = profiler.child(observation_id=observation_id, frame_index=index)
            with frame_profiler.stage("incremental_frame_total") as frame_stage:
                update = state.update(
                    latest_records,
                    [observation_id],
                    profiler=frame_profiler,
                )
                with frame_profiler.stage("stmr_incremental_snapshot") as snapshot_stage:
                    latest_map = state.snapshot(
                        subgoal_semantic_ids=subgoal_semantic_ids,
                        debug_point_limit=0,
                    )
                    snapshot_stage.update(
                        total_voxel_count=latest_map.projection_debug.get("voxel_count", 0),
                        total_projected_point_count=latest_map.projection_debug.get("point_count", 0),
                    )
                frame_dir = output_dir / "incremental_frames" / f"{index:03d}_{observation_id}"
                with frame_profiler.stage("incremental_png_render") as png_stage:
                    frame_png = render_stmr_map(latest_map, frame_dir)
                    png_stage.update(artifact_bytes=artifact_size_bytes(frame_png))
                with frame_profiler.stage("incremental_compact_json_render") as json_stage:
                    frame_json = frame_dir / "latest_stmr_map.json"
                    _write_incremental_compact_json(frame_json, latest_map, update.to_dict())
                    json_stage.update(artifact_bytes=artifact_size_bytes(frame_json))
                frame_stage.update(
                    processed_observation_count=len(state.processed_observation_ids),
                    new_projected_point_count=update.new_projected_point_count,
                    total_projected_point_count=state.point_count,
                    total_voxel_count=state.voxel_count,
                )
        updates.append(update.to_dict())

    if latest_map is None:
        with _optional_stage(profiler, "stmr_incremental_snapshot") as stage:
            latest_map = state.snapshot(
                subgoal_semantic_ids=subgoal_semantic_ids,
                debug_point_limit=debug_point_limit,
            )
            stage.update(
                total_voxel_count=latest_map.projection_debug.get("voxel_count", 0),
                total_projected_point_count=latest_map.projection_debug.get("point_count", 0),
            )
    else:
        # The measured per-frame snapshots omit debug points. Rebuild only the
        # final lightweight/full debug payload without touching depth or masks.
        latest_map = state.snapshot(
            subgoal_semantic_ids=subgoal_semantic_ids,
            debug_point_limit=debug_point_limit,
        )

    parity: dict[str, Any] | None = None
    if profiler is not None:
        with profiler.stage("batch_rebuild_reference") as stage:
            batch_map = build_stmr_map(latest_records, subgoal_semantic_ids=subgoal_semantic_ids)
            parity = _stmr_parity(latest_map, batch_map)
            stage.update(
                equivalent=parity["equivalent"],
                mismatch_fields=parity["mismatch_fields"],
                projected_point_count=batch_map.projection_debug.get("point_count", 0),
                voxel_count=batch_map.projection_debug.get("voxel_count", 0),
            )
        if not parity["equivalent"]:
            raise RuntimeError(f"Incremental STMR differs from batch reference: {parity['mismatch_fields']}")

    return latest_map, {
        "mode": "incremental_all_history",
        "processed_observation_ids": list(state.processed_observation_ids),
        "processed_observation_count": len(state.processed_observation_ids),
        "point_count": state.point_count,
        "voxel_count": state.voxel_count,
        "updates": updates,
        "batch_reference_parity": parity,
    }


def _records_through_observation(
    records: CustomMapRecords,
    observation_ids: list[str],
) -> CustomMapRecords:
    """Return the cumulative online-visible record prefix for one snapshot."""

    allowed = set(observation_ids)
    pose_cutoff = -1
    for index, pose in enumerate(records.poses):
        if pose.observation_id in allowed:
            pose_cutoff = index
    poses = list(records.poses[: pose_cutoff + 1]) if pose_cutoff >= 0 else []
    frames = [frame for frame in records.frames if frame.observation_id in allowed]
    return CustomMapRecords(
        run_dir=records.run_dir,
        actions=list(records.actions),
        poses=poses,
        frames=frames,
        poses_by_observation_id={
            observation_id: pose
            for observation_id, pose in records.poses_by_observation_id.items()
            if observation_id in allowed
        },
        warnings=list(records.warnings),
    )


def _complete_stmr_observation_ids(records: CustomMapRecords) -> list[str]:
    modalities: dict[str, set[str]] = {}
    for frame in records.frames:
        if not frame.observation_id:
            continue
        modalities.setdefault(frame.observation_id, set()).add(frame.modality.lower())
    return [
        observation_id
        for observation_id, available in modalities.items()
        if available.intersection({"depth", "depth_perspective"})
        and "stmr_open_vocab_mask" in available
        and observation_id in records.poses_by_observation_id
    ]


def _write_incremental_compact_json(
    path: Path,
    stmr_map: StmrMap,
    update: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "content": {
                "semantic_matrix": stmr_map.semantic_matrix,
                "semantic_legend": {
                    str(key): dict(value)
                    for key, value in sorted(stmr_map.semantic_legend.items())
                },
                "path_cells": stmr_map.path_cells,
                "current_cell": stmr_map.current_cell,
                "center_pose": stmr_map.center_pose,
                "map_config": stmr_map.map_config,
                "coordinate_frame": stmr_map.coordinate_frame,
                "projection_debug": projection_debug_summary(stmr_map.projection_debug),
                "incremental_update": update,
                "warnings": stmr_map.warnings,
            }
        },
    )


def _stmr_parity(incremental: StmrMap, batch: StmrMap) -> dict[str, Any]:
    checks = {
        "semantic_matrix": incremental.semantic_matrix == batch.semantic_matrix,
        "voxel_count": incremental.projection_debug.get("voxel_count")
        == batch.projection_debug.get("voxel_count"),
        "semantic_cell_counts": incremental.projection_debug.get("semantic_cell_counts")
        == batch.projection_debug.get("semantic_cell_counts"),
        "source_observation_ids": incremental.projection_debug.get("source_observation_ids")
        == batch.projection_debug.get("source_observation_ids"),
        "trajectory_trace": incremental.trajectory_trace == batch.trajectory_trace,
        "trajectory_summary": incremental.trajectory_summary == batch.trajectory_summary,
        "center_pose": incremental.center_pose == batch.center_pose,
    }
    mismatches = [name for name, matches in checks.items() if not matches]
    return {"equivalent": not mismatches, "mismatch_fields": mismatches, "checks": checks}


def _run_stmr_probe(run: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Backward-compatible private entry point used by older manual scripts."""

    return run_stmr_probe(run, args)


def _extract_probe_landmarks(
    instruction: str,
    run: dict[str, Any],
    args: argparse.Namespace,
    *,
    include_defaults: bool = True,
) -> dict[str, Any]:
    manual = list(args.landmarks or [])
    max_landmarks = max(1, int(args.max_landmarks))
    if args.landmark_source == "manual":
        if not manual:
            raise SystemExit("--landmark-source manual requires --landmarks")
        landmarks = extract_landmarks("", manual_landmarks=manual, include_defaults=False)[:max_landmarks]
        return {"source": "manual", "landmarks": landmarks}
    if args.landmark_source == "rule":
        landmarks = extract_landmarks(
            instruction,
            manual_landmarks=manual,
            include_defaults=include_defaults,
        )[:max_landmarks]
        return {"source": "rule", "landmarks": landmarks}

    extractor = ModelLandmarkExtractor()
    try:
        result = extractor.extract(
            instruction,
            task_class=str(run.get("class_name") or ""),
            max_landmarks=max_landmarks,
        )
        landmarks = _merge_manual_landmarks(result.landmarks, manual, max_landmarks)
        return {
            "source": "model",
            "landmarks": landmarks,
            "raw_response": result.raw_response,
            "prompt": result.prompt,
            "model_summary": result.model_summary,
        }
    except ModelLandmarkExtractorError:
        if not args.allow_rule_fallback:
            raise
        landmarks = extract_landmarks(
            instruction,
            manual_landmarks=manual,
            include_defaults=include_defaults,
        )[:max_landmarks]
        return {"source": "model_fallback_rule", "landmarks": landmarks}


def _merge_manual_landmarks(model_landmarks: list[Any], manual: list[str], max_landmarks: int) -> list[Any]:
    if not manual:
        return model_landmarks[:max_landmarks]
    manual_landmarks = extract_landmarks("", manual_landmarks=manual, include_defaults=False)
    merged = []
    seen = set()
    for item in [*manual_landmarks, *model_landmarks]:
        if item.label in seen:
            continue
        seen.add(item.label)
        merged.append(item)
        if len(merged) >= max_landmarks:
            break
    return merged


def _run_open_vocab_perception(
    variant_dir: Path,
    landmarks: list[Any],
    args: argparse.Namespace,
    *,
    session: StmrPerceptionSession | None = None,
    profiler: StageProfiler | None = None,
) -> list[dict[str, Any]]:
    rgb_frames = _rgb_frames(variant_dir)
    if args.max_frames > 0:
        rgb_frames = rgb_frames[: args.max_frames]
    frames_by_observation = _frames_by_observation_id(variant_dir)
    poses_by_observation = _poses_by_observation_id(variant_dir)
    center_pose = _center_pose(poses_by_observation)
    dynamic_config = DynamicTargetFilterConfig(
        enabled=args.dynamic_target_filter == "on",
        min_detection_score=args.dynamic_min_score,
        min_mask_pixels=args.dynamic_min_mask_pixels,
        max_depth_p95_m=args.dynamic_max_depth_p95,
        primary_track_radius_m=args.dynamic_track_radius_m,
    )
    if args.perception_source == "grounding_dino_sam" and session is None:
        session = _create_perception_session(args, profiler=profiler)
    frame_items: list[dict[str, Any]] = []
    prompts = [item.label for item in landmarks]
    for frame in rgb_frames:
        if args.perception_source == "fake_boxes":
            detections = _fake_detections(frame["path"], prompts)
            masks = _fake_masks(frame["path"], detections)
            matches = filter_detections_by_tfidf(detections, landmarks, threshold=args.tfidf_threshold)
        else:
            if session is None:
                raise RuntimeError("GroundingDINO/SAM perception requires a StmrPerceptionSession")
            perception = session.infer_frame(
                frame["path"],
                landmarks,
                observation_id=str(frame["observation_id"]),
                profiler=profiler,
            )
            detections = perception.detections
            matches = perception.matches
            masks = perception.masks
        candidate_builder = lambda: _dynamic_candidates_for_frame(
            variant_dir,
            frame,
            masks,
            matches,
            frames_by_observation=frames_by_observation,
            poses_by_observation=poses_by_observation,
            center_pose=center_pose,
        )
        candidates = (
            session.build_dynamic_candidates(
                candidate_builder,
                observation_id=str(frame["observation_id"]),
                profiler=profiler,
            )
            if session is not None
            else candidate_builder()
        )
        frame_items.append(
            {
                "frame": frame,
                "masks": masks,
                "matches": matches,
                "dynamic_target_candidates": candidates,
            }
        )

    if session is not None:
        policy_summary = session.apply_dynamic_policy(frame_items, config=dynamic_config, profiler=profiler)
    else:
        from plugins.research.stmr_map.perception.dynamic_targets import apply_dynamic_target_policy

        policy_summary = apply_dynamic_target_policy(frame_items, config=dynamic_config)
    results = []
    for item in frame_items:
        frame = item["frame"]
        candidates = item.get("dynamic_target_candidates", [])
        active_session = session or _create_perception_session(args, profiler=profiler)
        result = active_session.write_frame_artifacts(
            variant_dir,
            frame,
            item["masks"],
            item["matches"],
            candidates=candidates,
            dynamic_filter_enabled=dynamic_config.enabled,
            rejected_masks=item.get("rejected_masks", []),
            kept_non_primary_masks=item.get("kept_non_primary_masks", []),
            primary_dynamic_track=item.get("primary_dynamic_track"),
            profiler=profiler,
        )
        results.append(
            {
                "observation_id": result.observation_id,
                "rgb_path": str(result.rgb_path),
                "mask_path": str(result.mask_path),
                "overlay_path": str(result.overlay_path),
                "boxes_path": str(result.boxes_path),
                "tfidf_matches": result.metadata.get("tfidf_matches", []),
                "kept_masks": result.metadata.get("kept_masks", []),
                "dynamic_target_candidates": result.metadata.get("dynamic_target_candidates", []),
                "rejected_masks": result.metadata.get("rejected_masks", []),
                "kept_non_primary_masks": result.metadata.get("kept_non_primary_masks", []),
                "primary_dynamic_track": result.metadata.get("primary_dynamic_track"),
                "dynamic_filter_summary": policy_summary,
            }
        )
    return results


def _create_perception_session(
    args: argparse.Namespace,
    *,
    profiler: StageProfiler | None = None,
) -> StmrPerceptionSession:
    return StmrPerceptionSession(
        grounding_model=args.grounding_model,
        sam_model=args.sam_model,
        device=args.seg_device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        tfidf_threshold=args.tfidf_threshold,
        profiler=profiler,
    )


def _optional_stage(
    profiler: StageProfiler | None,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Any:
    if profiler is not None:
        return profiler.stage(name, metadata=metadata)
    from contextlib import nullcontext

    from plugins.research.stmr_map.profiling import StageContext

    return nullcontext(StageContext(dict(metadata or {})))


def _prepare_variant_memory(run_dir: Path, variant_dir: Path) -> None:
    source_memory = run_dir / "memory"
    frames_source = source_memory / "frames.jsonl"
    poses_source = source_memory / "poses.jsonl"
    if not source_memory.exists() or not frames_source.exists() or not poses_source.exists():
        raise SystemExit(f"Run directory does not contain source memory required for STMR probe: {run_dir}")
    if not frames_source.read_text(encoding="utf-8").strip() or not poses_source.read_text(encoding="utf-8").strip():
        raise SystemExit(f"Run source memory is empty; refusing to overwrite STMR artifacts: {run_dir}")
    memory_dir = variant_dir / "memory"
    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    for name in ("actions.jsonl", "observations.jsonl", "poses.jsonl"):
        source = source_memory / name
        if source.exists():
            shutil.copyfile(source, memory_dir / name)
    kept_frames = []
    for frame in _read_jsonl(source_memory / "frames.jsonl"):
        modality = str(frame.get("modality") or "").lower()
        if modality in {"segmentation", "mask", "object_mask", "stmr_open_vocab_mask"}:
            continue
        raw_path = frame.get("path")
        if isinstance(raw_path, str) and raw_path:
            frame = dict(frame)
            frame["path"] = str(_resolve_path(raw_path, run_dir))
        kept_frames.append(frame)
    with (memory_dir / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for frame in kept_frames:
            handle.write(json.dumps(frame, ensure_ascii=False) + "\n")


def _select_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.run_dir:
        return [_run_item_from_dir(Path(path)) for path in args.run_dir]
    sweep_dir = Path(args.sweep_dir)
    summary_path = sweep_dir / "custom_map_semantic_sweep_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs = [item for item in summary.get("runs", []) if isinstance(item, dict)]
    else:
        runs = [_run_item_from_dir(path.parent.parent.parent) for path in sorted(sweep_dir.rglob("contexts/custom_map/latest_custom_map.json"))]
    selected = []
    for class_name in REPRESENTATIVE_CLASSES:
        candidates = [item for item in runs if item.get("class_name") == class_name]
        if candidates:
            selected.append(candidates[0])
    return selected


def _run_item_from_dir(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "collection_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data.setdefault("run_dir", str(run_dir))
        return data
    return {"run_dir": str(run_dir), "case_id": run_dir.name, "class_name": run_dir.name.split("_", 1)[0]}


def _rgb_frames(run_dir: Path) -> list[dict[str, Any]]:
    frames = []
    for item in _read_jsonl(run_dir / "memory" / "frames.jsonl"):
        if str(item.get("modality", "")).lower() != "rgb":
            continue
        path = _resolve_path(str(item.get("path") or ""), run_dir)
        if path.exists() and item.get("observation_id"):
            frames.append({"observation_id": str(item["observation_id"]), "path": path})
    return frames


def _dynamic_candidates_for_frame(
    variant_dir: Path,
    frame: dict[str, Any],
    masks: list[SamMaskResult],
    matches: list[TfidfMatch],
    *,
    frames_by_observation: dict[str, dict[str, dict[str, Any]]],
    poses_by_observation: dict[str, PosePoint],
    center_pose: PosePoint | None,
) -> list[dict[str, Any]]:
    observation_id = str(frame["observation_id"])
    image = Image.open(frame["path"])
    image_size = image.size
    bundle = frames_by_observation.get(observation_id, {})
    depth_frame = bundle.get("depth")
    depth = _load_depth_frame(depth_frame, variant_dir)
    pose = poses_by_observation.get(observation_id)
    fov_degrees = 90.0
    if isinstance(depth_frame, dict):
        metadata = depth_frame.get("metadata") or {}
        if isinstance(metadata, dict):
            try:
                fov_degrees = float(metadata.get("fov_degrees", fov_degrees))
            except (TypeError, ValueError):
                fov_degrees = 90.0
    candidates = []
    for index, (mask_result, match) in enumerate(zip(masks, matches)):
        candidate = build_dynamic_candidate(
            observation_id=observation_id,
            detection_index=index,
            mask_result=mask_result,
            match=match,
            image_size=image_size,
            depth=depth,
            pose=pose,
            center_pose=center_pose,
            fov_degrees=fov_degrees,
        )
        candidate["tfidf_keep"] = bool(match.keep)
        candidates.append(candidate)
    return candidates


def _frames_by_observation_id(run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for item in _read_jsonl(run_dir / "memory" / "frames.jsonl"):
        observation_id = item.get("observation_id")
        modality = str(item.get("modality") or "").lower()
        if not observation_id or not modality:
            continue
        frame = dict(item)
        raw_path = frame.get("path")
        if isinstance(raw_path, str) and raw_path:
            frame["path"] = str(_resolve_path(raw_path, run_dir))
        grouped.setdefault(str(observation_id), {})[modality] = frame
    return grouped


def _poses_by_observation_id(run_dir: Path) -> dict[str, PosePoint]:
    poses = {}
    for item in _read_jsonl(run_dir / "memory" / "poses.jsonl"):
        observation_id = item.get("observation_id")
        raw_pose = item.get("pose")
        if not observation_id or not isinstance(raw_pose, dict):
            continue
        try:
            poses[str(observation_id)] = PosePoint(
                x=float(raw_pose.get("x", 0.0)),
                y=float(raw_pose.get("y", 0.0)),
                z=float(raw_pose.get("z", 0.0)),
                yaw_deg=float(raw_pose.get("yaw_deg", raw_pose.get("yaw", 0.0))),
            )
        except (TypeError, ValueError):
            continue
    return poses


def _center_pose(poses_by_observation: dict[str, PosePoint]) -> PosePoint | None:
    if not poses_by_observation:
        return None
    return list(poses_by_observation.values())[-1]


def _load_depth_frame(frame: dict[str, Any] | None, run_dir: Path) -> np.ndarray | None:
    if not isinstance(frame, dict) or not frame.get("path"):
        return None
    path = _resolve_path(str(frame["path"]), run_dir)
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npy":
            return np.load(path)
        image = Image.open(path)
        return np.asarray(image, dtype=np.float32)
    except (OSError, ValueError):
        return None


def _read_instruction(run_dir: Path) -> str:
    for item in _read_jsonl(run_dir / "memory" / "observations.jsonl"):
        instruction = item.get("instruction") or item.get("active_instruction")
        if instruction:
            return str(instruction)
    return ""


def _load_custom_map_content(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(raw, dict):
        content = raw.get("content", raw)
        return content if isinstance(content, dict) else None
    return None


def _subgoal_semantic_ids(labels: list[str]) -> set[int]:
    ids = set()
    for label in labels:
        semantic = semantic_class_for_ade_label(label)
        if semantic.semantic_id > 0:
            ids.add(semantic.semantic_id)
    return ids


def _fake_detections(rgb_path: Path, prompts: list[str]) -> list[DetectionResult]:
    from PIL import Image

    width, height = Image.open(rgb_path).size
    detections = []
    for index, prompt in enumerate(prompts[:3]):
        pad = 12 + index * 8
        detections.append(
            DetectionResult(
                box_xyxy=(pad, pad, width - pad, height - pad),
                phrase=prompt,
                score=0.9,
                label=prompt,
            )
        )
    return detections


def _fake_masks(rgb_path: Path, detections: list[DetectionResult]) -> list[SamMaskResult]:
    import numpy as np
    from PIL import Image

    width, height = Image.open(rgb_path).size
    masks = []
    for detection in detections:
        x0, y0, x1, y1 = [int(round(value)) for value in detection.box_xyxy]
        mask = np.zeros((height, width), dtype=bool)
        mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
        masks.append(SamMaskResult(detection=detection, mask=mask, iou_score=1.0))
    return masks


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _resolve_path(raw_path: str, run_dir: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    run_path = run_dir / path
    if run_path.exists():
        return run_path
    return Path.cwd() / path


def _summary_root(args: argparse.Namespace, runs: list[dict[str, Any]]) -> Path:
    if args.run_dir and len(runs) == 1:
        return Path(str(runs[0]["run_dir"])) / "contexts" / "stmr_map"
    return Path(args.sweep_dir)


def _summary_html(summary: dict[str, Any], output_dir: Path) -> str:
    cards = []
    for item in summary["runs"]:
        cards.append(
            f"<article><h2>{_escape(item['class_name'])} / {_escape(item['case_id'])}</h2>"
            f"<a href='{_rel(item['stmr_html'], output_dir)}'>open STMR viewer</a>"
            f"<dl><dt>points</dt><dd>{item['point_count']}</dd>"
            f"<dt>voxels</dt><dd>{item['voxel_count']}</dd>"
            f"<dt>subgoal hits</dt><dd>{item['subgoal_priority_hit_count']}</dd></dl>"
            f"<img src='{_rel(item['stmr_png'], output_dir)}'></article>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>STMR Compare Summary</title>
<style>body{{font:14px/1.45 system-ui;margin:24px;background:#f8fafc;color:#0f172a}}article{{background:white;border:1px solid #cbd5e1;border-radius:8px;padding:14px;margin:16px 0}}img{{max-width:720px;width:100%;border:1px solid #e2e8f0}}</style>
</head><body><h1>STMR Compare Summary</h1>{''.join(cards)}</body></html>"""


def _rel(path: str, output_dir: Path) -> str:
    try:
        return os.path.relpath(Path(path), output_dir).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


if __name__ == "__main__":
    main()
