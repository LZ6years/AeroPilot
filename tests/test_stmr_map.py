"""Tests for the standalone STMR research module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from plugins.research.custom_map.context_modules.custom_map.records import read_custom_map_records
from plugins.research.custom_map.context_modules.custom_map.schemas import CustomMapRecords, FrameRecord, PosePoint
from plugins.research.custom_map.context_modules.custom_map.projection_geometry import (
    DEPTH_MODE_OPTICAL_Z,
    DEPTH_MODE_RAY_DISTANCE,
)
from plugins.research.custom_map.perception.semantic_palette import SEMANTIC_CLASSES
from plugins.research.stmr_map.core import (
    IncrementalStmrMapState,
    _local_to_centered_cell,
    _voxel_center,
    _world_to_local,
    build_stmr_map,
)
import plugins.research.stmr_map.core as stmr_core
from plugins.research.stmr_map.debug_payload import build_points_preview, projection_debug_summary, stratified_sample_points
from plugins.research.stmr_map.html_report import render_stmr_html
from plugins.research.stmr_map.perception.dynamic_targets import (
    DynamicTargetFilterConfig,
    apply_dynamic_target_policy,
    build_dynamic_candidate,
    should_write_candidate,
)
from plugins.research.stmr_map.perception.grounding_dino_backend import DetectionResult, GroundingDinoBackend
from plugins.research.stmr_map.perception.landmark_extractor import extract_landmarks
from plugins.research.stmr_map.perception.model_landmark_extractor import (
    SYSTEM_PROMPT,
    ModelLandmarkExtractorError,
    parse_landmarks_response,
)
from plugins.research.stmr_map.perception.open_vocab_writer import write_open_vocab_mask
from plugins.research.stmr_map.perception.sam_backend import SamMaskResult
from plugins.research.stmr_map.perception.session import StmrPerceptionSession
from plugins.research.stmr_map.perception.tfidf_filter import TfidfMatch, filter_detections_by_tfidf, tfidf_similarity
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryRecordReader


def test_landmark_extractor_returns_instruction_and_default_uav_categories():
    landmarks = extract_landmarks(
        "Fly along the road, pass the traffic light, and stop near the tower.",
        manual_landmarks=["rooftop"],
    )

    labels = [item.label for item in landmarks]
    assert labels[0] == "rooftop"
    assert "road" in labels
    assert "traffic light" in labels
    assert "tower" in labels
    assert "building" in labels
    assert len(labels) == len(set(labels))


def test_tfidf_filter_keeps_matching_phrases_and_rejects_unrelated():
    landmarks = extract_landmarks("", manual_landmarks=["traffic light", "building"], include_defaults=False)
    detections = [
        DetectionResult((0, 0, 10, 10), "traffic light", 0.9),
        DetectionResult((0, 0, 10, 10), "tree", 0.9),
    ]

    matches = filter_detections_by_tfidf(detections, landmarks, threshold=0.8)

    assert tfidf_similarity("traffic light", "traffic light") == 1.0
    assert matches[0].keep is True
    assert matches[0].landmark == "traffic light"
    assert matches[1].keep is False


def test_perception_session_reuses_injected_backends_across_frames(tmp_path):
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(rgb)

    class FakeGrounding:
        def __init__(self):
            self.warmups = 0
            self.calls = 0

        def warmup(self, *, profiler=None):
            self.warmups += 1

        def detect(self, rgb_path, prompts, *, profiler=None):
            self.calls += 1
            return [DetectionResult((1, 1, 7, 7), prompts[0], 0.9, prompts[0])]

    class FakeSam:
        def __init__(self):
            self.warmups = 0
            self.calls = 0

        def warmup(self, *, profiler=None):
            self.warmups += 1

        def segment(self, rgb_path, detections, *, profiler=None):
            self.calls += 1
            return [SamMaskResult(detections[0], np.ones((8, 8), dtype=bool), 0.9)]

    grounding = FakeGrounding()
    sam = FakeSam()
    session = StmrPerceptionSession(grounding_backend=grounding, sam_backend=sam)
    landmarks = extract_landmarks("", manual_landmarks=["building"], include_defaults=False)

    session.warmup()
    first = session.infer_frame(rgb, landmarks, observation_id="obs_1")
    second = session.infer_frame(rgb, landmarks, observation_id="obs_2")

    assert grounding.warmups == sam.warmups == 1
    assert grounding.calls == sam.calls == 2
    assert first.detections[0].phrase == "building"
    assert second.masks[0].mask.shape == (8, 8)


def test_grounding_backend_prefers_text_labels_from_new_transformers_result(tmp_path):
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(rgb)

    class FakeValue:
        def __init__(self, value):
            self.value = value

        def to(self, device):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.value)

        def item(self):
            return self.value

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTorch:
        def no_grad(self):
            return FakeNoGrad()

        def tensor(self, value, device=None):
            return FakeValue(value)

    class FakeProcessor:
        def __call__(self, **kwargs):
            return {"input_ids": FakeValue([1])}

        def post_process_grounded_object_detection(
            self,
            outputs,
            input_ids,
            *,
            text_threshold,
            target_sizes,
            threshold,
        ):
            return [{
                "boxes": [FakeValue([1, 2, 3, 4])],
                "scores": [FakeValue(0.9)],
                "labels": ["legacy-label"],
                "text_labels": ["building"],
            }]

    class FakeModel:
        def __call__(self, **kwargs):
            return object()

    backend = GroundingDinoBackend(device="cpu")
    backend._processor = FakeProcessor()
    backend._model = FakeModel()
    backend._device = "cpu"
    backend._torch = lambda: FakeTorch()

    detections = backend.detect(rgb, ["building"])

    assert detections[0].phrase == "building"


def test_model_landmark_response_parses_and_sanitizes_short_phrases():
    landmarks = parse_landmarks_response(
        {
            "landmarks": [
                {"label": "Traffic Light!!!", "reason": "mentioned in the instruction"},
                {"label": "the very large red building beside the route with many windows", "reason": "visible structure"},
                {"label": "Traffic Light", "reason": "duplicate"},
            ]
        },
        max_landmarks=3,
    )

    assert [item.label for item in landmarks] == ["traffic light", "the very large red building"]
    assert landmarks[0].source == "model_landmark_extractor"
    assert landmarks[0].reason == "mentioned in the instruction"


def test_model_landmark_response_rejects_invalid_or_empty_json():
    try:
        parse_landmarks_response({"landmarks": []})
    except ModelLandmarkExtractorError as exc:
        assert "no usable landmarks" in str(exc)
    else:
        raise AssertionError("expected empty landmark response to fail")

    try:
        parse_landmarks_response({"items": [{"label": "road"}]})
    except ModelLandmarkExtractorError as exc:
        assert "landmarks list" in str(exc)
    else:
        raise AssertionError("expected missing landmarks response to fail")


def test_model_landmark_prompt_excludes_benchmark_truth_and_debug_fields():
    lowered = SYSTEM_PROMPT.lower()

    assert "benchmark truth" in lowered
    assert "hidden labels" in lowered
    assert "phase_progress_evidence" not in lowered
    assert "current_phase" not in lowered
    assert "latest_stmr_map" not in lowered


def test_open_vocab_writer_appends_stmr_mask_frame_without_overwriting_segmentation(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(rgb_path)
    _write_base_memory(tmp_path, rgb_path)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    detection = DetectionResult((2, 2, 6, 6), "building", 0.95)

    result = write_open_vocab_mask(
        tmp_path,
        rgb_path,
        "obs_0001",
        [SamMaskResult(detection, mask, 0.9)],
        [type("Match", (), {"detection_index": 0, "phrase": "building", "landmark": "building", "similarity": 1.0, "keep": True, "score": 0.95})()],
    )

    frames = [json.loads(line) for line in (tmp_path / "memory" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result.mask_path.exists()
    assert result.overlay_path.exists()
    assert result.boxes_path.exists()
    assert any(item["modality"] == "segmentation" for item in frames)
    assert any(item["modality"] == "stmr_open_vocab_mask" for item in frames)
    assert result.label_map[SEMANTIC_CLASSES["building_structure"].color_key]["label"] == "building_structure"


def test_open_vocab_writer_records_rejected_dynamic_metadata_without_painting_it(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(rgb_path)
    _write_base_memory(tmp_path, rgb_path)
    keep_mask = np.zeros((8, 8), dtype=bool)
    keep_mask[1:4, 1:4] = True
    rejected = {
        "candidate_id": "obs_0001:1:person",
        "observation_id": "obs_0001",
        "label": "person",
        "status": "rejected_tiny_mask",
        "reject_reasons": ["tiny_mask"],
    }

    result = write_open_vocab_mask(
        tmp_path,
        rgb_path,
        "obs_0001",
        [SamMaskResult(DetectionResult((1, 1, 4, 4), "person", 0.9), keep_mask, 0.9)],
        [TfidfMatch(0, "person", "person", 1.0, True, 0.9)],
        rejected_masks=[rejected],
        dynamic_target_candidates=[rejected],
    )

    assert result.metadata["rejected_masks"] == [rejected]
    mask_pixels = np.asarray(Image.open(result.mask_path).convert("RGB"))
    person_color = np.asarray(SEMANTIC_CLASSES["person"].color, dtype=np.uint8)
    assert int(np.all(mask_pixels == person_color, axis=-1).sum()) == 9


def test_dynamic_target_policy_rejects_low_quality_person_and_keeps_near_large_person():
    config = DynamicTargetFilterConfig()
    near_large = {
        "candidate_id": "obs_0001:0:person",
        "observation_id": "obs_0001",
        "semantic_id": 10,
        "label": "person",
        "is_dynamic": True,
        "score": 0.9,
        "mask_pixel_count": 500,
        "depth_p50": 7.0,
        "depth_p95": 8.0,
        "local_centroid_forward_m": 4.0,
        "local_centroid_right_m": 0.0,
        "edge_touching": False,
    }
    far_tiny = {
        "candidate_id": "obs_0009:0:person",
        "observation_id": "obs_0009",
        "semantic_id": 10,
        "label": "person",
        "is_dynamic": True,
        "score": 0.8,
        "mask_pixel_count": 21,
        "depth_p50": 65.0,
        "depth_p95": 71.0,
        "local_centroid_forward_m": 70.0,
        "local_centroid_right_m": 8.0,
        "edge_touching": False,
    }
    low_score = {
        "candidate_id": "obs_0002:0:person",
        "observation_id": "obs_0002",
        "semantic_id": 10,
        "label": "person",
        "is_dynamic": True,
        "score": 0.3,
        "mask_pixel_count": 500,
        "depth_p50": 8.0,
        "depth_p95": 9.0,
        "local_centroid_forward_m": 5.0,
        "local_centroid_right_m": 0.3,
        "edge_touching": False,
    }
    frame_items = [{"dynamic_target_candidates": [near_large, far_tiny, low_score]}]

    summary = apply_dynamic_target_policy(frame_items, config=config)

    assert summary["rejected_count"] == 2
    assert near_large["status"] == "primary_track"
    assert "tiny_mask" in far_tiny["reject_reasons"]
    assert "low_detection_score" in low_score["reject_reasons"]
    assert should_write_candidate(near_large) is True
    assert should_write_candidate(far_tiny) is False


def test_dynamic_target_policy_keeps_primary_track_cluster_and_suppresses_other_dynamic_cluster():
    first = _candidate("obs_0001", 0.0, 3.0, score=0.9, pixels=600)
    second = _candidate("obs_0002", 0.8, 3.8, score=0.85, pixels=500)
    other = _candidate("obs_0003", 30.0, 25.0, score=0.9, pixels=450)
    frame_items = [{"dynamic_target_candidates": [first, second, other]}]

    summary = apply_dynamic_target_policy(frame_items, config=DynamicTargetFilterConfig(primary_track_radius_m=8.0))

    assert summary["primary_dynamic_track"]["candidate_count"] == 2
    assert first["status"] == "primary_track"
    assert second["status"] == "primary_track"
    assert other["status"] == "kept_non_primary"
    assert should_write_candidate(other) is False


def test_build_dynamic_candidate_adds_depth_and_local_centroid_stats():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    depth = np.full((8, 8), 8.0, dtype=np.float32)

    candidate = build_dynamic_candidate(
        observation_id="obs_0001",
        detection_index=0,
        mask_result=SamMaskResult(DetectionResult((2, 2, 6, 6), "person", 0.9), mask, 0.8),
        match=TfidfMatch(0, "person", "person", 1.0, True, 0.9),
        image_size=(8, 8),
        depth=depth,
        pose=PosePoint(0.0, 0.0, 0.0, 0.0),
        center_pose=PosePoint(0.0, 0.0, 0.0, 0.0),
    )

    assert candidate["label"] == "person"
    assert candidate["is_dynamic"] is True
    assert candidate["mask_pixel_count"] == 16
    assert candidate["depth_p95"] == 8.0
    assert "local_centroid_forward_m" in candidate


def test_stmr_top_height_projection_selects_highest_voxel_semantic(tmp_path):
    route = SEMANTIC_CLASSES["route_surface"]
    building = SEMANTIC_CLASSES["building_structure"]
    # top half of the image has smaller y_cam, hence larger world_z/up.
    mask_colors = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_colors[0:4, :, :] = building.color
    mask_colors[4:8, :, :] = route.color
    _write_stmr_observation(tmp_path, mask_colors, {route.color_key: route, building.color_key: building})

    records = _records(tmp_path)
    stmr_map = build_stmr_map(records, grid_size=20, meters_per_cell=5.0, stride=1)

    assert stmr_map.projection_debug["point_count"] > 0
    assert stmr_map.projection_debug["voxel_count"] > 0
    assert any(building.semantic_id in row for row in stmr_map.semantic_matrix)
    assert stmr_map.map_config["local_extent_m"] == 100.0
    assert stmr_map.map_config["meters_per_cell"] == 5.0
    assert stmr_map.coordinate_frame["world_frame"] == "uav_flow_pose_xy_yaw"
    assert stmr_map.projection_debug["sampled_points"]
    first_point = stmr_map.projection_debug["sampled_points"][0]
    assert {"pixel_u", "pixel_v", "raw_depth_m", "camera_x", "camera_y", "camera_z", "local_forward_m", "local_right_m", "local_up_m"} <= set(first_point)
    assert stmr_map.projection_debug["voxel_world_centers"]
    first_voxel = stmr_map.projection_debug["voxel_world_centers"][0]
    assert {"world_x", "world_y", "world_z", "local_forward_m", "local_right_m", "local_up_m"} <= set(first_voxel)
    assert stmr_map.projection_debug["depth_interpretation_used"] == DEPTH_MODE_OPTICAL_Z
    assert stmr_map.projection_debug["depth_interpretation_comparison"]["comparison_mode"] == DEPTH_MODE_RAY_DISTANCE
    assert stmr_map.projection_debug["depth_interpretation_comparison"]["samples"]
    assert stmr_map.projection_debug["frame_point_ranges"]
    assert "multi_frame_projection_drift" in stmr_map.projection_debug


def test_stmr_subgoal_priority_can_override_top_height(tmp_path):
    route = SEMANTIC_CLASSES["route_surface"]
    building = SEMANTIC_CLASSES["building_structure"]
    mask_colors = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_colors[0:4, :, :] = building.color
    mask_colors[4:8, :, :] = route.color
    _write_stmr_observation(tmp_path, mask_colors, {route.color_key: route, building.color_key: building})

    records = _records(tmp_path)
    stmr_map = build_stmr_map(
        records,
        subgoal_semantic_ids={route.semantic_id},
        grid_size=20,
        meters_per_cell=5.0,
        stride=1,
    )

    assert stmr_map.projection_debug["subgoal_priority_hits"]
    assert any(route.semantic_id in row for row in stmr_map.semantic_matrix)


def test_stmr_world_to_local_coordinate_frame_handles_heading():
    center = PosePoint(x=10.0, y=20.0, z=3.0, yaw_deg=0.0)

    forward = _world_to_local(15.0, 20.0, 5.0, center)
    right = _world_to_local(10.0, 25.0, 5.0, center)

    assert round(forward["local_forward_m"], 3) == 5.0
    assert round(forward["local_right_m"], 3) == 0.0
    assert round(forward["local_up_m"], 3) == 2.0
    assert round(right["local_forward_m"], 3) == 0.0
    assert round(right["local_right_m"], 3) == 5.0

    yaw_90 = PosePoint(x=10.0, y=20.0, z=0.0, yaw_deg=90.0)
    world_plus_y = _world_to_local(10.0, 25.0, 0.0, yaw_90)
    world_plus_x = _world_to_local(15.0, 20.0, 0.0, yaw_90)

    assert round(world_plus_y["local_forward_m"], 3) == 5.0
    assert round(world_plus_y["local_right_m"], 3) == 0.0
    assert round(world_plus_x["local_forward_m"], 3) == 0.0
    assert round(world_plus_x["local_right_m"], 3) == -5.0


def test_stmr_voxel_center_reprojects_to_top_height_cell(tmp_path):
    route = SEMANTIC_CLASSES["route_surface"]
    mask_colors = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_colors[:, :, :] = route.color
    _write_stmr_observation(tmp_path, mask_colors, {route.color_key: route})

    records = _records(tmp_path)
    stmr_map = build_stmr_map(records, grid_size=20, meters_per_cell=5.0, stride=1)

    assert stmr_map.projection_debug["top_height_projection"]
    assert stmr_map.projection_debug["alignment_warnings"] == []
    center = PosePoint(**stmr_map.center_pose)
    for item in stmr_map.projection_debug["top_height_projection"]:
        world = _voxel_center(item["voxel_key"], stmr_map.map_config["voxel_size_m"])
        local = _world_to_local(world["world_x"], world["world_y"], world["world_z"], center)
        cell = _local_to_centered_cell(local, stmr_map.map_config["grid_size"], stmr_map.map_config["meters_per_cell"])
        assert cell == {"row": item["row"], "col": item["col"]}


def test_stmr_path_is_stamped_as_minus_one(tmp_path):
    route = SEMANTIC_CLASSES["route_surface"]
    mask_colors = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_colors[:, :, :] = route.color
    _write_stmr_observation(tmp_path, mask_colors, {route.color_key: route})

    records = _records(tmp_path)
    stmr_map = build_stmr_map(records, grid_size=20, meters_per_cell=5.0, stride=1)

    assert stmr_map.current_cell == {"row": 10, "col": 10}
    assert stmr_map.semantic_matrix[10][10] == -1


def test_stmr_trajectory_trace_preserves_ordered_subcell_motion():
    poses = [
        PosePoint(x=0.0, y=0.0, z=0.0, yaw_deg=0.0, observation_id="obs_0001"),
        PosePoint(x=1.0, y=0.0, z=0.0, yaw_deg=0.0, observation_id="obs_0002"),
        PosePoint(x=2.0, y=0.0, z=0.0, yaw_deg=0.0, observation_id="obs_0003"),
    ]

    trace, summary = stmr_core._pose_trace(poses, poses[-1], 20, 5.0)

    assert len(trace) == 3
    assert {round(item["row"]) for item in trace} == {10}
    assert [item["row"] for item in trace] == [10.4, 10.2, 10.0]
    assert [item["sequence_index"] for item in trace] == [0, 1, 2]
    assert summary["visible_start_marker"] == "START"
    assert summary["recent_motion_local_m"] == {
        "forward_delta_m": 1.0,
        "right_delta_m": 0.0,
        "distance_m": 1.0,
    }


def test_stmr_trajectory_trace_recenters_yaw_and_breaks_off_map_segments():
    poses = [
        PosePoint(x=0.0, y=0.0, z=0.0, yaw_deg=0.0),
        PosePoint(x=100.0, y=0.0, z=0.0, yaw_deg=0.0),
        PosePoint(x=1.0, y=0.0, z=0.0, yaw_deg=90.0),
    ]

    trace, summary = stmr_core._pose_trace(poses, poses[-1], 20, 5.0)

    assert [item["segment_index"] for item in trace] == [0, 1]
    assert summary["visible_segment_count"] == 2
    assert summary["recent_motion_local_m"]["forward_delta_m"] == pytest.approx(0.0, abs=1e-3)
    assert summary["recent_motion_local_m"]["right_delta_m"] == pytest.approx(99.0)
    assert trace[-1]["row"] == trace[-1]["col"] == 10.0


def test_stmr_trajectory_trace_marks_entry_and_ignores_yaw_only_history():
    center = PosePoint(x=0.0, y=0.0, z=0.0, yaw_deg=90.0, observation_id="obs_0003")
    entry_trace, entry_summary = stmr_core._pose_trace(
        [PosePoint(x=-100.0, y=0.0), center],
        center,
        20,
        5.0,
    )
    stationary_trace, stationary_summary = stmr_core._pose_trace(
        [PosePoint(x=0.0, y=0.0, yaw_deg=0.0), center],
        center,
        20,
        5.0,
    )

    assert entry_trace[-1]["row"] == entry_trace[-1]["col"] == 10.0
    assert entry_summary["visible_start_marker"] == "ENTRY"
    assert stationary_trace == [
        {
            "sequence_index": 0,
            "segment_index": 0,
            "row": 10.0,
            "col": 10.0,
            "local_forward_m": 0.0,
            "local_right_m": 0.0,
            "observation_id": None,
        }
    ]
    assert stationary_summary["translation_status"] == "stationary_or_yaw_only"
    assert stationary_summary["recent_motion_local_m"]["distance_m"] == 0.0


def test_incremental_stmr_matches_batch_and_keeps_all_history(tmp_path):
    full_records = _incremental_records(tmp_path)
    first_records = _records_with_observations(full_records, {"obs_0001"})
    state = IncrementalStmrMapState(grid_size=20, meters_per_cell=5.0, stride=1)

    first = state.update(first_records)
    duplicate = state.update(first_records)
    second = state.update(full_records, ["obs_0002"])
    incremental = state.snapshot(debug_point_limit=None)
    batch = build_stmr_map(full_records, grid_size=20, meters_per_cell=5.0, stride=1)

    assert first.processed_observation_ids == ("obs_0001",)
    assert duplicate.processed_observation_ids == ()
    assert duplicate.duplicate_observation_ids == ("obs_0001",)
    assert duplicate.new_projected_point_count == 0
    assert second.processed_observation_ids == ("obs_0002",)
    assert incremental.semantic_matrix == batch.semantic_matrix
    assert incremental.trajectory_trace == batch.trajectory_trace
    assert incremental.trajectory_summary == batch.trajectory_summary
    assert incremental.center_pose == batch.center_pose
    assert incremental.projection_debug["voxel_count"] == batch.projection_debug["voxel_count"]
    assert incremental.projection_debug["semantic_cell_counts"] == batch.projection_debug["semantic_cell_counts"]
    assert incremental.projection_debug["source_observation_ids"] == ["obs_0001", "obs_0002"]
    assert incremental.projection_debug["processed_observation_count"] == 2


def test_incremental_stmr_retries_incomplete_observation(tmp_path):
    full_records = _incremental_records(tmp_path)
    incomplete = CustomMapRecords(
        run_dir=full_records.run_dir,
        actions=full_records.actions,
        poses=full_records.poses,
        frames=[
            frame
            for frame in full_records.frames
            if not (frame.observation_id == "obs_0002" and frame.modality == "stmr_open_vocab_mask")
        ],
        poses_by_observation_id=full_records.poses_by_observation_id,
        warnings=[],
    )
    state = IncrementalStmrMapState(stride=1)

    pending = state.update(incomplete, ["obs_0002"])
    recovered = state.update(full_records, ["obs_0002"])

    assert pending.processed_observation_ids == ()
    assert pending.pending_observation_ids == ("obs_0002",)
    assert recovered.processed_observation_ids == ("obs_0002",)
    assert state.processed_observation_ids == ("obs_0002",)


def test_incremental_stmr_recenters_historical_world_points(tmp_path):
    full_records = _incremental_records(tmp_path)
    first_records = _records_with_observations(full_records, {"obs_0001"})
    state = IncrementalStmrMapState(stride=1)
    state.update(first_records)
    first_snapshot = state.snapshot(debug_point_limit=None)
    first_points = {
        (item["world_x"], item["world_y"], item["world_z"]): (
            item["local_forward_m"],
            item["local_right_m"],
        )
        for item in first_snapshot.projection_debug["sampled_points"]
        if item["source_observation_id"] == "obs_0001"
    }

    state.update(full_records, ["obs_0002"])
    second_snapshot = state.snapshot(debug_point_limit=None)
    second_points = {
        (item["world_x"], item["world_y"], item["world_z"]): (
            item["local_forward_m"],
            item["local_right_m"],
        )
        for item in second_snapshot.projection_debug["sampled_points"]
        if item["source_observation_id"] == "obs_0001"
    }

    assert second_points.keys() == first_points.keys()
    assert any(second_points[key] != first_points[key] for key in first_points)
    assert second_snapshot.center_pose == {"x": 20.0, "y": 5.0, "z": 0.0, "yaw_deg": 90.0}


def test_incremental_stmr_light_snapshot_preserves_total_counts(tmp_path):
    records = _incremental_records(tmp_path)
    state = IncrementalStmrMapState(stride=1)
    state.update(records)

    snapshot = state.snapshot(debug_point_limit=3)
    preview = build_points_preview(
        stmr_projection_debug=snapshot.projection_debug,
        custom_map=None,
        stmr_point_budget=3,
        voxel_budget=3,
    )

    assert len(snapshot.projection_debug["sampled_points"]) == 3
    assert snapshot.projection_debug["debug_points_truncated"] is True
    assert preview["counts"]["stmr_points_total"] == state.point_count


def test_incremental_stmr_projection_failure_does_not_mutate_state(tmp_path, monkeypatch):
    records = _incremental_records(tmp_path)
    first_records = _records_with_observations(records, {"obs_0001"})
    state = IncrementalStmrMapState(stride=1)
    state.update(first_records)
    point_count = state.point_count
    voxel_count = state.voxel_count

    def fail_projection(*args, **kwargs):
        raise RuntimeError("synthetic projection failure")

    monkeypatch.setattr(stmr_core, "_project_frame_to_voxels", fail_projection)
    with pytest.raises(RuntimeError, match="synthetic projection failure"):
        state.update(records, ["obs_0002"])

    assert state.point_count == point_count
    assert state.voxel_count == voxel_count
    assert state.processed_observation_ids == ("obs_0001",)


def test_stmr_debug_stratified_sampling_respects_budget_and_keeps_small_classes():
    points = [
        {"semantic_id": 20, "source_observation_id": "obs_a", "local_forward_m": float(index)}
        for index in range(1000)
    ]
    points.extend(
        {"semantic_id": 10, "source_observation_id": "obs_b", "local_forward_m": float(index)}
        for index in range(8)
    )

    sampled = stratified_sample_points(points, 50)

    assert len(sampled) <= 50
    assert any(item["semantic_id"] == 10 for item in sampled)
    assert any(item["semantic_id"] == 20 for item in sampled)


def test_stmr_projection_debug_summary_removes_heavy_point_arrays():
    debug = {
        "point_count": 3,
        "sampled_points": [{"semantic_id": 10}, {"semantic_id": 20}],
        "voxel_world_centers": [{"semantic_id": 10}],
        "frame_point_ranges": [{"count": 2}],
    }

    summary = projection_debug_summary(debug)

    assert "sampled_points" not in summary
    assert "voxel_world_centers" not in summary
    assert summary["sampled_points_total"] == 2
    assert summary["voxel_world_centers_total"] == 1
    assert summary["frame_point_ranges"] == [{"count": 2}]


def test_stmr_points_preview_caps_custom_points_stmr_points_and_voxels():
    stmr_debug = {
        "sampled_points": [{"semantic_id": 10, "source_observation_id": f"obs_{index % 3}"} for index in range(100)],
        "voxel_world_centers": [{"semantic_id": 30, "observation_ids": [f"obs_{index % 4}"]} for index in range(80)],
    }
    custom_map = {
        "projection_debug": {
            "sampled_points": [{"semantic_id": 40, "source_observation_id": f"obs_{index % 2}"} for index in range(90)]
        }
    }

    preview = build_points_preview(
        stmr_projection_debug=stmr_debug,
        custom_map=custom_map,
        stmr_point_budget=12,
        custom_point_budget=10,
        voxel_budget=7,
    )

    assert len(preview["stmr_points"]) <= 12
    assert len(preview["custom_points"]) <= 10
    assert len(preview["stmr_voxels"]) <= 7
    assert preview["counts"]["stmr_points_total"] == 100
    assert preview["counts"]["custom_points_total"] == 90
    assert preview["sampling_strategy"] == "stratified_by_semantic_id_and_observation_id_v1"


def test_stmr_html_uses_lightweight_3d_preview_without_old_compare(tmp_path):
    route = SEMANTIC_CLASSES["route_surface"]
    building = SEMANTIC_CLASSES["building_structure"]
    mask_colors = np.zeros((8, 8, 3), dtype=np.uint8)
    mask_colors[0:4, :, :] = building.color
    mask_colors[4:8, :, :] = route.color
    _write_stmr_observation(tmp_path, mask_colors, {route.color_key: route, building.color_key: building})
    stmr_map = build_stmr_map(_records(tmp_path), grid_size=20, meters_per_cell=5.0, stride=1)
    preview = build_points_preview(
        stmr_projection_debug=stmr_map.projection_debug,
        custom_map=None,
        stmr_point_budget=5,
        custom_point_budget=0,
        voxel_budget=3,
    )

    html_path = render_stmr_html(
        stmr_map,
        tmp_path / "latest_stmr_map.png",
        tmp_path / "contexts" / "stmr_map",
        points_preview=preview,
    )
    html = html_path.read_text(encoding="utf-8")

    assert "STMR Lightweight 3D Preview" in html
    assert "Semantic Class Gallery" in html
    assert "Pipeline Timing" in html
    assert "pointCloud" in html
    assert "points_preview" in html
    assert '"stmr_point_budget": 5' in html
    assert '"custom_point_budget": 0' in html
    assert "3D Alignment Compare" not in html
    assert "compare3d" not in html
    assert "custom_map projected points" not in html


def _records(run_dir: Path):
    memory = FlightMemory("run", "map", "scene", "uav", memory_reader=MemoryRecordReader(run_dir))
    return read_custom_map_records(memory)


def _candidate(observation_id: str, right: float, forward: float, *, score: float, pixels: int) -> dict:
    return {
        "candidate_id": f"{observation_id}:0:person",
        "observation_id": observation_id,
        "semantic_id": SEMANTIC_CLASSES["person"].semantic_id,
        "label": "person",
        "is_dynamic": True,
        "score": score,
        "mask_pixel_count": pixels,
        "depth_p50": forward,
        "depth_p95": forward + 1.0,
        "local_centroid_forward_m": forward,
        "local_centroid_right_m": right,
        "edge_touching": False,
    }


def _write_base_memory(run_dir: Path, rgb_path: Path) -> None:
    memory_dir = run_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "actions.jsonl").write_text("", encoding="utf-8")
    (memory_dir / "observations.jsonl").write_text("", encoding="utf-8")
    (memory_dir / "poses.jsonl").write_text(
        json.dumps(
            {
                "id": "pose_0001",
                "observation_id": "obs_0001",
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frames = [
        {"id": "rgb_0001", "observation_id": "obs_0001", "modality": "rgb", "path": str(rgb_path)},
        {"id": "seg_0001", "observation_id": "obs_0001", "modality": "segmentation", "path": str(rgb_path)},
    ]
    (memory_dir / "frames.jsonl").write_text("\n".join(json.dumps(item) for item in frames) + "\n", encoding="utf-8")


def _write_stmr_observation(tmp_path: Path, mask_colors: np.ndarray, color_classes: dict[str, object]) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    mask_path = tmp_path / "stmr_mask.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(rgb_path)
    np.save(depth_path, np.full((8, 8), 8.0, dtype=np.float32))
    Image.fromarray(mask_colors.astype(np.uint8), mode="RGB").save(mask_path)
    label_map = {
        key: {
            "id": value.semantic_id,
            "label": value.label,
            "category": value.category,
            "source": "test",
        }
        for key, value in color_classes.items()
    }
    (memory_dir / "actions.jsonl").write_text("", encoding="utf-8")
    (memory_dir / "observations.jsonl").write_text("", encoding="utf-8")
    (memory_dir / "poses.jsonl").write_text(
        json.dumps(
            {
                "id": "pose_0001",
                "observation_id": "obs_0001",
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frames = [
        {"id": "rgb_0001", "observation_id": "obs_0001", "modality": "rgb", "path": str(rgb_path)},
        {
            "id": "depth_0001",
            "observation_id": "obs_0001",
            "modality": "depth",
            "path": str(depth_path),
            "metadata": {"fov_degrees": 90},
        },
        {
            "id": "mask_0001",
            "observation_id": "obs_0001",
            "modality": "stmr_open_vocab_mask",
            "path": str(mask_path),
            "metadata": {"label_map": label_map, "fov_degrees": 90},
        },
    ]
    (memory_dir / "frames.jsonl").write_text("\n".join(json.dumps(item) for item in frames) + "\n", encoding="utf-8")


def _incremental_records(tmp_path: Path) -> CustomMapRecords:
    """Create two complete observations with distinct poses and world evidence."""

    route = SEMANTIC_CLASSES["route_surface"]
    label_map = {
        route.color_key: {
            "id": route.semantic_id,
            "label": route.label,
            "category": route.category,
            "source": "test",
        }
    }
    poses = [
        PosePoint(0.0, 0.0, 0.0, 0.0, observation_id="obs_0001"),
        PosePoint(20.0, 5.0, 0.0, 90.0, observation_id="obs_0002"),
    ]
    frames: list[FrameRecord] = []
    for index, observation_id in enumerate(("obs_0001", "obs_0002"), start=1):
        rgb_path = tmp_path / f"rgb_{index:04d}.png"
        depth_path = tmp_path / f"depth_{index:04d}.npy"
        mask_path = tmp_path / f"mask_{index:04d}.png"
        Image.new("RGB", (8, 8), (20, 30, 40)).save(rgb_path)
        np.save(depth_path, np.full((8, 8), 8.0 + index, dtype=np.float32))
        mask = np.zeros((8, 8, 3), dtype=np.uint8)
        mask[:, :, :] = route.color
        Image.fromarray(mask, mode="RGB").save(mask_path)
        frames.extend(
            [
                FrameRecord(f"rgb_{index:04d}", observation_id, "rgb", rgb_path),
                FrameRecord(
                    f"depth_{index:04d}",
                    observation_id,
                    "depth",
                    depth_path,
                    {"fov_degrees": 90},
                ),
                FrameRecord(
                    f"mask_{index:04d}",
                    observation_id,
                    "stmr_open_vocab_mask",
                    mask_path,
                    {"label_map": label_map, "fov_degrees": 90},
                ),
            ]
        )
    return CustomMapRecords(
        run_dir=tmp_path,
        actions=[],
        poses=poses,
        frames=frames,
        poses_by_observation_id={pose.observation_id: pose for pose in poses if pose.observation_id},
        warnings=[],
    )


def _records_with_observations(
    records: CustomMapRecords,
    observation_ids: set[str],
) -> CustomMapRecords:
    poses = [pose for pose in records.poses if pose.observation_id in observation_ids]
    return CustomMapRecords(
        run_dir=records.run_dir,
        actions=records.actions,
        poses=poses,
        frames=[frame for frame in records.frames if frame.observation_id in observation_ids],
        poses_by_observation_id={
            observation_id: pose
            for observation_id, pose in records.poses_by_observation_id.items()
            if observation_id in observation_ids
        },
        warnings=records.warnings,
    )
