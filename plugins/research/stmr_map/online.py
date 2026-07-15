"""Stateful online adapter for the standalone STMR implementation.

The STMR core deliberately knows nothing about planner context or benchmark
execution.  This adapter consumes run-local observation memory, runs the
existing open-vocabulary perception stack once per complete observation, and
publishes the latest compact snapshot for a context module to expose.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image

from plugins.research.custom_map.context_modules.custom_map.records import read_custom_map_records
from plugins.research.custom_map.context_modules.custom_map.schemas import (
    CustomMapRecords,
    FrameRecord,
    PosePoint,
)

from .core import IncrementalStmrMapState, StmrMap, build_stmr_map
from .perception import (
    DEFAULT_UAV_PRIOR_NAME,
    MAX_LANDMARK_QUERIES,
    DynamicTargetFilterConfig,
    Landmark,
    StmrPerceptionSession,
    assemble_landmark_queries,
    build_dynamic_candidate,
    extract_landmarks,
)
from .profiling import StageProfiler, artifact_size_bytes
from .renderer import render_stmr_map


@dataclass(frozen=True)
class StmrOnlineMapResult:
    """Latest online result split into model-facing summary and debug paths."""

    status: str
    stale: bool
    latest_observation_id: str | None
    processed_observation_count: int
    point_count: int
    voxel_count: int
    center_pose: dict[str, Any]
    map_config: dict[str, Any]
    semantic_cell_counts: dict[str, int]
    trajectory_summary: dict[str, Any]
    png_path: Path | None
    json_path: Path | None
    warnings: tuple[str, ...] = ()

    def context_content(self) -> dict[str, Any]:
        """Return the intentionally small planner-facing representation."""

        return {
            "status": self.status,
            "stale": self.stale,
            "latest_observation_id": self.latest_observation_id,
            "center_pose": self.center_pose,
            "map_config": _compact_map_config(self.map_config),
            "processed_observation_count": self.processed_observation_count,
            "point_count": self.point_count,
            "voxel_count": self.voxel_count,
            "semantic_cell_counts": self.semantic_cell_counts,
            "trajectory_summary": self.trajectory_summary,
            "representation": "semantic_history_not_traversability",
        }


class StmrOnlineMapBuilder:
    """Incrementally update one case-local STMR map from run memory.

    A builder belongs to exactly one context-module instance and therefore one
    task runtime/case.  The perception session may be shared across builders so
    model weights are loaded once, while map evidence never crosses case
    boundaries.
    """

    def __init__(
        self,
        *,
        session: StmrPerceptionSession | None = None,
        prior: str = DEFAULT_UAV_PRIOR_NAME,
        max_queries: int = MAX_LANDMARK_QUERIES,
        device: str = "auto",
        profiler: StageProfiler | None = None,
        dynamic_filter_config: DynamicTargetFilterConfig | None = None,
    ) -> None:
        self.session = session
        self.prior = str(prior)
        self.max_queries = max(1, int(max_queries))
        self.device = str(device)
        self.profiler = profiler or StageProfiler(enabled=True, device=self.device, context={"pipeline": "stmr_online"})
        self.dynamic_filter_config = dynamic_filter_config or DynamicTargetFilterConfig()
        self.state = IncrementalStmrMapState()
        self._attempted_observation_ids: set[str] = set()
        self._failed_observation_ids: set[str] = set()
        self._failures: list[dict[str, str]] = []
        self._query_signature: tuple[str, str] | None = None
        self._queries: list[Landmark] = []
        self._last_processed_observation_id: str | None = None
        self._last_records: CustomMapRecords | None = None
        self._last_snapshot: StmrMap | None = None
        self._last_png_path: Path | None = None
        self._last_rendered_observation_id: str | None = None
        self._render_history: list[dict[str, Any]] = []
        self._artifact_warnings: set[str] = set()
        self._parity: dict[str, Any] | None = None
        self._last_result = StmrOnlineMapResult(
            status="waiting_for_complete_observation",
            stale=False,
            latest_observation_id=None,
            processed_observation_count=0,
            point_count=0,
            voxel_count=0,
            center_pose={},
            map_config={},
            semantic_cell_counts={},
            trajectory_summary={},
            png_path=None,
            json_path=None,
        )

    @property
    def last_result(self) -> StmrOnlineMapResult:
        return self._last_result

    def build(self, memory: Any, stage_context: Any) -> StmrOnlineMapResult:
        """Process unseen complete observations and publish the latest snapshot."""

        records = read_custom_map_records(memory)
        if records.run_dir is None:
            self._last_result = replace(
                self._last_result,
                status="run_memory_unavailable",
                stale=self._last_result.png_path is not None,
                warnings=tuple(sorted({*self._last_result.warnings, *records.warnings})),
            )
            return self._last_result

        queries = self._queries_for(stage_context)
        current_records = records
        work = [
            (observation_id, bundle)
            for observation_id, bundle in _complete_observations(current_records)
            if observation_id not in self.state.processed_observation_ids
            and observation_id not in self._attempted_observation_ids
        ]
        if not work:
            return self._last_result

        with self.profiler.stage("stmr_online_perception_map_total") as total_stage:
            current_records = self._process_work(current_records, work, queries)
            self._last_records = current_records
            self._last_result = self._publish(current_records)
            total_stage.update(
                attempted_observation_count=len(work),
                processed_observation_count=len(self.state.processed_observation_ids),
                point_count=self.state.point_count,
                voxel_count=self.state.voxel_count,
            )
        return self._last_result

    def _process_work(
        self,
        records: CustomMapRecords,
        work: list[tuple[str, dict[str, FrameRecord]]],
        queries: list[Landmark],
    ) -> CustomMapRecords:
        """Process one context build's unseen observations transactionally per frame."""

        current_records = records
        for observation_id, bundle in work:
            frame_profiler = self.profiler.child(observation_id=observation_id)
            try:
                with frame_profiler.stage("stmr_online_frame_total") as stage:
                    if bundle.get("mask") is None:
                        mask_frame = self._infer_mask(
                            current_records,
                            observation_id,
                            bundle,
                            queries,
                            frame_profiler,
                        )
                        current_records = replace(current_records, frames=[*current_records.frames, mask_frame])
                    update = self.state.update(current_records, [observation_id], profiler=frame_profiler)
                    stage.update(**update.to_dict())
                self._attempted_observation_ids.add(observation_id)
                if observation_id in update.pending_observation_ids:
                    self._failed_observation_ids.add(observation_id)
                elif observation_id in update.processed_observation_ids:
                    self._last_processed_observation_id = observation_id
                    self._parity = None
            except Exception as exc:
                # A failed frame must not be retried on every context build; it
                # remains visible in debug output while later observations can
                # still extend the last good map.
                self._attempted_observation_ids.add(observation_id)
                self._failed_observation_ids.add(observation_id)
                self._failures.append(
                    {
                        "observation_id": observation_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue

            if observation_id in update.processed_observation_ids:
                # Capture at the observation boundary while the state still
                # represents this exact incremental step. Catch-up builds may
                # contain several observations, so publishing only once after
                # the loop would lose the intermediate rendering history.
                try:
                    self._render_observation_snapshot(current_records, observation_id)
                except Exception as exc:
                    # The canonical evidence update already succeeded. Keep it
                    # processed and let the final publish retry the latest map;
                    # rendering diagnostics must never roll back map evidence.
                    self._artifact_warnings.add(
                        f"stmr_render_failed:{observation_id}:{type(exc).__name__}:{exc}"
                    )
        return current_records

    def debug_summary(self) -> dict[str, Any]:
        """Return case-level diagnostics for UAV-Flow reports."""

        result = self._last_result
        parity = self._batch_incremental_parity()
        return {
            **result.context_content(),
            "png_path": str(result.png_path) if result.png_path else None,
            "json_path": str(result.json_path) if result.json_path else None,
            "attempted_observation_count": len(self._attempted_observation_ids),
            "failed_observation_count": len(self._failed_observation_ids),
            "failed_observation_ids": sorted(self._failed_observation_ids),
            "failures": list(self._failures),
            "query_count": len(self._queries),
            "queries": [item.label for item in self._queries],
            "profiling": self.profiler.summary(),
            "batch_incremental_parity": parity,
            "render_history": self._render_history_summary(),
            "warnings": sorted({*result.warnings, *self._artifact_warnings}),
        }

    def _session(self) -> StmrPerceptionSession:
        if self.session is None:
            self.session = StmrPerceptionSession(device=self.device, profiler=self.profiler)
        return self.session

    def _queries_for(self, stage_context: Any) -> list[Landmark]:
        instruction = str(getattr(stage_context, "active_instruction", "") or "")
        semantic_goal = getattr(stage_context, "semantic_goal", {})
        target = str(semantic_goal.get("target") or "") if isinstance(semantic_goal, dict) else ""
        signature = (instruction, target)
        if signature == self._query_signature:
            return self._queries
        task_landmarks: list[Landmark] = []
        if target.strip():
            task_landmarks.append(
                Landmark(
                    label=target.strip(),
                    source="semantic_goal",
                    matched_text=target.strip(),
                    reason="Current task target; detector query only, not scene evidence.",
                )
            )
        task_landmarks.extend(extract_landmarks(instruction, include_defaults=False))
        self._queries = assemble_landmark_queries(
            task_landmarks,
            prior=self.prior,
            max_queries=self.max_queries,
        )
        self._query_signature = signature
        return self._queries

    def _infer_mask(
        self,
        records: CustomMapRecords,
        observation_id: str,
        bundle: dict[str, FrameRecord],
        queries: list[Landmark],
        profiler: StageProfiler,
    ) -> FrameRecord:
        rgb_frame = bundle["rgb"]
        if records.run_dir is None or rgb_frame.path is None:
            raise RuntimeError(f"STMR RGB artifact unavailable for {observation_id}")
        session = self._session()
        perception = session.infer_frame(
            rgb_frame.path,
            queries,
            observation_id=observation_id,
            profiler=profiler,
        )
        candidates = self._dynamic_candidates(
            records,
            observation_id,
            bundle,
            perception.masks,
            perception.matches,
        )
        frame_item = {
            "dynamic_target_candidates": candidates,
        }
        policy = session.apply_dynamic_policy(
            [frame_item],
            config=self.dynamic_filter_config,
            profiler=profiler,
        )
        result = session.write_frame_artifacts(
            records.run_dir,
            {"path": rgb_frame.path, "observation_id": observation_id},
            perception.masks,
            perception.matches,
            candidates=candidates,
            dynamic_filter_enabled=self.dynamic_filter_config.enabled,
            rejected_masks=frame_item.get("rejected_masks", []),
            kept_non_primary_masks=frame_item.get("kept_non_primary_masks", []),
            primary_dynamic_track=frame_item.get("primary_dynamic_track"),
            profiler=profiler,
        )
        metadata = {**result.metadata, "dynamic_filter_summary": policy, "online_policy_scope": "single_frame"}
        return FrameRecord(
            id=f"stmr_online_{observation_id}",
            observation_id=observation_id,
            modality="stmr_open_vocab_mask",
            path=result.mask_path,
            metadata=metadata,
        )

    def _dynamic_candidates(
        self,
        records: CustomMapRecords,
        observation_id: str,
        bundle: dict[str, FrameRecord],
        masks: list[Any],
        matches: list[Any],
    ) -> list[dict[str, Any]]:
        rgb_path = bundle["rgb"].path
        if rgb_path is None:
            return []
        with Image.open(rgb_path) as image:
            image_size = image.size
        depth_frame = bundle["depth"]
        depth = _load_depth(depth_frame.path)
        pose = records.poses_by_observation_id.get(observation_id)
        center_pose = records.poses[-1] if records.poses else pose
        fov_degrees = _float_or_default(depth_frame.metadata.get("fov_degrees"), 90.0)
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

    def _publish(self, records: CustomMapRecords) -> StmrOnlineMapResult:
        processed_ids = self.state.processed_observation_ids
        if not processed_ids or records.run_dir is None:
            warnings = tuple(
                sorted(
                    {
                        *records.warnings,
                        *self._artifact_warnings,
                        *(f"stmr_failed:{item}" for item in self._failed_observation_ids),
                    }
                )
            )
            output_dir = records.run_dir / "contexts" / "stmr_map" if records.run_dir is not None else None
            json_path = output_dir / "latest_stmr_map.json" if output_dir is not None else None
            result = StmrOnlineMapResult(
                status="waiting_for_complete_observation" if not self._failed_observation_ids else "failed_no_snapshot",
                stale=False,
                latest_observation_id=None,
                processed_observation_count=0,
                point_count=self.state.point_count,
                voxel_count=self.state.voxel_count,
                center_pose={},
                map_config={},
                semantic_cell_counts={},
                trajectory_summary={},
                png_path=None,
                json_path=json_path,
                warnings=warnings,
            )
            if output_dir is not None and json_path is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                json_path.write_text(
                    json.dumps(
                        {
                            "content": result.context_content(),
                            "debug": {
                                "attempted_observation_ids": sorted(self._attempted_observation_ids),
                                "failed_observation_ids": sorted(self._failed_observation_ids),
                                "failures": self._failures,
                                "queries": [item.__dict__ for item in self._queries],
                                "profiling": self.profiler.summary(),
                                "render_history": self._render_history_summary(),
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                self.profiler.write_jsonl(output_dir / "stmr_online_timing.jsonl")
                self.profiler.write_summary(output_dir / "stmr_online_timing_summary.json")
            return result

        latest_observation_id = self._last_processed_observation_id or processed_ids[-1]
        if (
            self._last_snapshot is None
            or self._last_png_path is None
            or self._last_rendered_observation_id != latest_observation_id
        ):
            # Retry only the latest rendering when an earlier debug-artifact
            # write failed. Evidence remains idempotently processed.
            self._render_observation_snapshot(records, latest_observation_id)
        assert self._last_snapshot is not None
        assert self._last_png_path is not None
        stmr_map = self._last_snapshot
        png_path = self._last_png_path
        output_dir = records.run_dir / "contexts" / "stmr_map"
        json_path = output_dir / "latest_stmr_map.json"
        warnings = tuple(
            sorted(
                {
                    *records.warnings,
                    *stmr_map.warnings,
                    *self._artifact_warnings,
                    *(f"stmr_failed:{item}" for item in self._failed_observation_ids),
                }
            )
        )
        result = StmrOnlineMapResult(
            status="ready",
            stale=bool(self._failed_observation_ids),
            latest_observation_id=latest_observation_id,
            processed_observation_count=len(processed_ids),
            point_count=self.state.point_count,
            voxel_count=self.state.voxel_count,
            center_pose=dict(stmr_map.center_pose),
            map_config=dict(stmr_map.map_config),
            semantic_cell_counts=_semantic_cell_counts(stmr_map),
            trajectory_summary=dict(stmr_map.trajectory_summary),
            png_path=png_path,
            json_path=json_path,
            warnings=warnings,
        )
        payload = {
            "content": result.context_content(),
            "stmr_map": asdict(stmr_map),
            "debug": {
                "attempted_observation_ids": sorted(self._attempted_observation_ids),
                "failed_observation_ids": sorted(self._failed_observation_ids),
                "failures": self._failures,
                "queries": [item.__dict__ for item in self._queries],
                "profiling": self.profiler.summary(),
                "render_history": self._render_history_summary(),
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.profiler.write_jsonl(output_dir / "stmr_online_timing.jsonl")
        self.profiler.write_summary(output_dir / "stmr_online_timing_summary.json")
        return result

    def _render_observation_snapshot(self, records: CustomMapRecords, observation_id: str) -> None:
        """Render latest model input and preserve one human-facing history frame."""

        if records.run_dir is None:
            raise RuntimeError("STMR render requires run-local artifact storage")
        with self.profiler.stage("stmr_incremental_snapshot"):
            stmr_map = self.state.snapshot(debug_point_limit=0)
        output_dir = records.run_dir / "contexts" / "stmr_map"
        with self.profiler.stage("stmr_online_png_render") as stage:
            png_path = render_stmr_map(stmr_map, output_dir)
            stage.update(artifact_bytes=artifact_size_bytes(png_path))

        self._last_snapshot = stmr_map
        self._last_png_path = png_path
        self._last_rendered_observation_id = observation_id
        self._preserve_render_history(output_dir, observation_id, stmr_map, png_path)

    def _preserve_render_history(
        self,
        output_dir: Path,
        observation_id: str,
        stmr_map: StmrMap,
        latest_png_path: Path,
    ) -> None:
        """Copy the rendered map and update its compact report-only manifest.

        Failures here are deliberately fail-open: the latest PNG is already
        available to the planner, and history is a human/debug artifact only.
        """

        if any(item.get("observation_id") == observation_id for item in self._render_history):
            return
        try:
            snapshot_dir = output_dir / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            safe_observation_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", observation_id).strip("_") or "observation"
            sequence = len(self._render_history) + 1
            snapshot_path = snapshot_dir / f"{sequence:04d}_{safe_observation_id}.png"
            shutil.copy2(latest_png_path, snapshot_path)
            entry = {
                "sequence": sequence,
                "observation_id": observation_id,
                "processed_observation_count": len(self.state.processed_observation_ids),
                "point_count": self.state.point_count,
                "voxel_count": self.state.voxel_count,
                "center_pose": dict(stmr_map.center_pose),
                "semantic_cell_counts": _semantic_cell_counts(stmr_map),
                "png_path": str(snapshot_path),
            }
            self._render_history.append(entry)
            manifest_path = output_dir / "render_history.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "stmr_render_history_v1",
                        "snapshot_count": len(self._render_history),
                        "snapshots": self._render_history,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self._artifact_warnings.add(
                f"stmr_render_history_write_failed:{observation_id}:{type(exc).__name__}:{exc}"
            )

    def _render_history_summary(self) -> dict[str, Any]:
        output_dir = None
        if self._last_records is not None and self._last_records.run_dir is not None:
            output_dir = self._last_records.run_dir / "contexts" / "stmr_map"
        return {
            "snapshot_count": len(self._render_history),
            "manifest_path": str(output_dir / "render_history.json") if output_dir is not None else None,
            "snapshot_dir": str(output_dir / "snapshots") if output_dir is not None else None,
        }

    def _batch_incremental_parity(self) -> dict[str, Any] | None:
        """Build the final reference only for the case-level evaluation report."""

        if self._parity is not None:
            return dict(self._parity)
        if self._last_records is None or self._last_snapshot is None:
            return None
        try:
            with self.profiler.stage("batch_rebuild_reference"):
                reference = build_stmr_map(self._last_records)
        except Exception as exc:
            self._parity = {
                "equivalent": False,
                "status": "reference_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return dict(self._parity)
        checks = {
            "semantic_matrix": self._last_snapshot.semantic_matrix == reference.semantic_matrix,
            "semantic_legend": self._last_snapshot.semantic_legend == reference.semantic_legend,
            "trajectory_trace": self._last_snapshot.trajectory_trace == reference.trajectory_trace,
            "trajectory_summary": self._last_snapshot.trajectory_summary == reference.trajectory_summary,
            "center_pose": self._last_snapshot.center_pose == reference.center_pose,
            "point_count": self._last_snapshot.projection_debug.get("point_count")
            == reference.projection_debug.get("point_count"),
            "voxel_count": self._last_snapshot.projection_debug.get("voxel_count")
            == reference.projection_debug.get("voxel_count"),
        }
        self._parity = {
            "equivalent": all(checks.values()),
            "checks": checks,
            "mismatch_fields": [name for name, equivalent in checks.items() if not equivalent],
        }
        if self._last_records.run_dir is not None:
            output_dir = self._last_records.run_dir / "contexts" / "stmr_map"
            self.profiler.write_jsonl(output_dir / "stmr_online_timing.jsonl")
            self.profiler.write_summary(output_dir / "stmr_online_timing_summary.json")
        return dict(self._parity)


def _complete_observations(records: CustomMapRecords) -> list[tuple[str, dict[str, FrameRecord]]]:
    grouped: dict[str, dict[str, FrameRecord]] = {}
    order: list[str] = []
    for frame in records.frames:
        observation_id = str(frame.observation_id or "")
        if not observation_id:
            continue
        if observation_id not in grouped:
            grouped[observation_id] = {}
            order.append(observation_id)
        modality = frame.modality.lower()
        if modality == "rgb":
            grouped[observation_id]["rgb"] = frame
        elif modality in {"depth", "depth_perspective"}:
            grouped[observation_id]["depth"] = frame
        elif modality == "stmr_open_vocab_mask":
            grouped[observation_id]["mask"] = frame
    return [
        (observation_id, grouped[observation_id])
        for observation_id in order
        if {"rgb", "depth"}.issubset(grouped[observation_id])
        and observation_id in records.poses_by_observation_id
    ]


def _load_depth(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npy":
            depth = np.asarray(np.load(path), dtype=np.float32)
        else:
            with Image.open(path) as image:
                depth = np.asarray(image, dtype=np.float32)
    except (OSError, ValueError):
        return None
    return depth if depth.ndim == 2 else None


def _semantic_cell_counts(stmr_map: StmrMap) -> dict[str, int]:
    counts = Counter(value for row in stmr_map.semantic_matrix for value in row)
    result: dict[str, int] = {}
    for semantic_id, count in sorted(counts.items()):
        if semantic_id == -2:
            label = "unknown"
        elif semantic_id == -1:
            label = "visited_path"
        elif semantic_id == 0:
            label = "no_semantic_voxel"
        else:
            label = str(stmr_map.semantic_legend.get(semantic_id, {}).get("label") or semantic_id)
        result[label] = result.get(label, 0) + int(count)
    return result


def _compact_map_config(config: dict[str, Any]) -> dict[str, Any]:
    keep = ("grid_size", "meters_per_cell", "orientation", "local_extent_m", "voxel_size_m")
    return {key: config[key] for key in keep if key in config}


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = ["StmrOnlineMapBuilder", "StmrOnlineMapResult"]
