"""Tests for the standalone STMR research module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from plugins.research.custom_map.context_modules.custom_map.records import read_custom_map_records
from plugins.research.custom_map.context_modules.custom_map.schemas import PosePoint
from plugins.research.custom_map.context_modules.custom_map.projection_geometry import (
    DEPTH_MODE_OPTICAL_Z,
    DEPTH_MODE_RAY_DISTANCE,
)
from plugins.research.custom_map.perception.semantic_palette import SEMANTIC_CLASSES
from plugins.research.stmr_map.core import (
    _local_to_centered_cell,
    _voxel_center,
    _world_to_local,
    build_stmr_map,
)
from plugins.research.stmr_map.perception.dynamic_targets import (
    DynamicTargetFilterConfig,
    apply_dynamic_target_policy,
    build_dynamic_candidate,
    should_write_candidate,
)
from plugins.research.stmr_map.perception.grounding_dino_backend import DetectionResult
from plugins.research.stmr_map.perception.landmark_extractor import extract_landmarks
from plugins.research.stmr_map.perception.model_landmark_extractor import (
    SYSTEM_PROMPT,
    ModelLandmarkExtractorError,
    parse_landmarks_response,
)
from plugins.research.stmr_map.perception.open_vocab_writer import write_open_vocab_mask
from plugins.research.stmr_map.perception.sam_backend import SamMaskResult
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
