"""Tests for SegFormer-style preprocessing feeding custom_map."""

from __future__ import annotations

import json
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plugins.research.custom_map.context_modules.custom_map.projection import (
    _select_cell_semantic,
    distance_confidence,
    build_projected_semantic_layer,
)
from plugins.research.custom_map.context_modules.custom_map.records import read_custom_map_records
from plugins.research.custom_map.context_modules.custom_map.schemas import PosePoint
from plugins.research.custom_map.perception.closed_set_segmentation import (
    ClosedSetSegmentationConfig,
    run_closed_set_segmentation,
)
from plugins.research.custom_map.perception.segformer_backend import SegmentationPrediction
from plugins.research.custom_map.perception.segmentation_writer import append_segmentation_result
from plugins.research.custom_map.perception.semantic_palette import SEMANTIC_CLASSES, semantic_class_for_ade_label, semantic_label_map
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader

_REAL_COLLECTION_PROBE_PATH = Path(__file__).parent / "manual" / "custom_map_real_collection_probe.py"
_REAL_COLLECTION_PROBE_SPEC = importlib.util.spec_from_file_location("custom_map_real_collection_probe", _REAL_COLLECTION_PROBE_PATH)
assert _REAL_COLLECTION_PROBE_SPEC is not None
_REAL_COLLECTION_PROBE = importlib.util.module_from_spec(_REAL_COLLECTION_PROBE_SPEC)
assert _REAL_COLLECTION_PROBE_SPEC.loader is not None
_REAL_COLLECTION_PROBE_SPEC.loader.exec_module(_REAL_COLLECTION_PROBE)
_action_script = _REAL_COLLECTION_PROBE._action_script
_build_sweep_summary = _REAL_COLLECTION_PROBE._build_sweep_summary

_COMPARE_PROBE_PATH = Path(__file__).parent / "manual" / "custom_map_segmentation_compare_probe.py"
_COMPARE_PROBE_SPEC = importlib.util.spec_from_file_location("custom_map_segmentation_compare_probe", _COMPARE_PROBE_PATH)
assert _COMPARE_PROBE_SPEC is not None
_COMPARE_PROBE = importlib.util.module_from_spec(_COMPARE_PROBE_SPEC)
assert _COMPARE_PROBE_SPEC.loader is not None
_COMPARE_PROBE_SPEC.loader.exec_module(_COMPARE_PROBE)
_build_compare_summary = _COMPARE_PROBE._build_compare_summary
_selected_models = _COMPARE_PROBE._selected_models
PAPER_CANDIDATE_MODELS = _COMPARE_PROBE.PAPER_CANDIDATE_MODELS
STABLE_PAPER_CANDIDATE_MODELS = _COMPARE_PROBE.STABLE_PAPER_CANDIDATE_MODELS
_prepare_variant_memory = _COMPARE_PROBE._prepare_variant_memory

from plugins.research.custom_map.perception.huggingface_backend import _model_family


class FakeSegmentationBackend:
    def __init__(self, class_ids: np.ndarray, id_to_label: dict[int, str]) -> None:
        self.class_ids = class_ids
        self.id_to_label = id_to_label

    def predict(self, rgb_path: str | Path) -> SegmentationPrediction:
        return SegmentationPrediction(
            class_ids=self.class_ids,
            id_to_label=self.id_to_label,
            device="fake",
            model_name="fake-segformer",
        )


def _write_semantic_projection_observation(
    tmp_path: Path,
    *,
    segmentation: np.ndarray,
    label_names: dict[str, str],
    depth_m: float = 8.0,
) -> None:
    writer = MemoryEventWriter(tmp_path)
    depth_path = tmp_path / "depth.npy"
    segmentation_path = tmp_path / "segmentation.png"
    depth = np.full(segmentation.shape, depth_m, dtype=np.float32)
    np.save(depth_path, depth)
    image = Image.new("RGB", (segmentation.shape[1], segmentation.shape[0]), SEMANTIC_CLASSES["free"].color)
    for class_name in label_names:
        class_info = SEMANTIC_CLASSES[class_name]
        ys, xs = np.where(segmentation == class_info.semantic_id)
        for x, y in zip(xs, ys):
            image.putpixel((int(x), int(y)), class_info.color)
    image.save(segmentation_path)

    label_dir = tmp_path / "memory" / "map"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_map = {"colors": {}}
    for class_name, category in label_names.items():
        class_info = SEMANTIC_CLASSES[class_name]
        label_map["colors"][class_info.color_key] = {
            "id": class_info.semantic_id,
            "label": class_info.label,
            "category": category,
        }
    (label_dir / "mask_label_map.json").write_text(json.dumps(label_map), encoding="utf-8")

    writer.write_observation(
        {
            "scene_instance_id": "scene",
            "agent_id": "uav",
            "modalities": ["depth", "segmentation", "pose"],
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            "frames": {
                "depth": {"path": str(depth_path), "mock": False, "fov_degrees": 90},
                "segmentation": {"path": str(segmentation_path), "mock": False, "fov_degrees": 90},
            },
            "source": "semantic_projection_test",
        }
    )


def test_segformer_palette_maps_ade_labels_to_stable_custom_classes():
    assert semantic_class_for_ade_label("road").label == "route_surface"
    assert semantic_class_for_ade_label("building").label == "building_structure"
    assert semantic_class_for_ade_label("tree").label == "vegetation"
    assert semantic_class_for_ade_label("sky").label == "free"
    assert semantic_class_for_ade_label("car").label == "vehicle"
    assert semantic_class_for_ade_label("bike").label == "two_wheeler_or_small_vehicle"
    assert semantic_class_for_ade_label("motorcycle").label == "two_wheeler_or_small_vehicle"
    assert semantic_class_for_ade_label("scooter").label == "two_wheeler_or_small_vehicle"
    assert semantic_class_for_ade_label("floor").label == "route_surface"
    assert semantic_class_for_ade_label("terrain").label == "route_surface"
    assert semantic_class_for_ade_label("earth").label == "route_surface"
    assert semantic_class_for_ade_label("bench").label == "movable_obstacle"
    assert semantic_class_for_ade_label("chair").label == "movable_obstacle"
    assert semantic_class_for_ade_label("table").label == "movable_obstacle"
    assert semantic_class_for_ade_label("box").label == "movable_obstacle"
    assert semantic_class_for_ade_label("pot").label == "movable_obstacle"
    assert semantic_class_for_ade_label("ceiling").label == "overhead_structure"
    assert semantic_class_for_ade_label("canopy").label == "overhead_structure"
    assert semantic_class_for_ade_label("tunnel").label == "overhead_structure"
    assert semantic_class_for_ade_label("water").label == "water_or_hazard_surface"
    assert semantic_class_for_ade_label("river").label == "water_or_hazard_surface"
    assert semantic_class_for_ade_label("pool").label == "water_or_hazard_surface"
    assert semantic_class_for_ade_label("painting").label == "visual_clutter"
    assert semantic_class_for_ade_label("curtain").label == "visual_clutter"
    assert semantic_class_for_ade_label("poster").label == "visual_clutter"
    assert semantic_class_for_ade_label("rider").label == "person"
    assert semantic_class_for_ade_label("traffic sign").label == "street_fixture"
    assert semantic_class_for_ade_label("facade").label == "building_structure"
    assert semantic_class_for_ade_label("foliage").label == "vegetation"
    assert semantic_class_for_ade_label("cloud").label == "free"
    assert semantic_class_for_ade_label("mystery ade class").label == "unknown"
    assert "terrain" not in SEMANTIC_CLASSES
    assert "other_scene" not in SEMANTIC_CLASSES
    labels = {item["label"] for item in semantic_label_map().values()}
    assert {"terrain", "other_scene"}.isdisjoint(labels)


def test_closed_set_segmentation_writes_dense_mask_overlay_and_label_map(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4), (20, 30, 40)).save(rgb_path)
    backend = FakeSegmentationBackend(
        np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.int32,
        ),
        {0: "road", 1: "building", 2: "tree", 3: "sky"},
    )

    result = run_closed_set_segmentation(
        rgb_path,
        tmp_path / "artifacts",
        observation_id="obs_0001",
        config=ClosedSetSegmentationConfig(source="test_segformer"),
        backend=backend,
    )

    assert result.mask_path.exists()
    assert result.overlay_path.exists()
    assert result.label_map[SEMANTIC_CLASSES["route_surface"].color_key]["label"] == "route_surface"
    assert result.stats["custom_class_pixel_counts"]["route_surface"] == 4
    assert result.stats["custom_class_pixel_counts"]["building_structure"] == 4
    assert result.stats["custom_class_pixel_counts"]["free"] == 4
    image = Image.open(result.mask_path).convert("RGB")
    assert image.getpixel((0, 0)) == SEMANTIC_CLASSES["route_surface"].color
    assert image.getpixel((3, 3)) == SEMANTIC_CLASSES["free"].color


def test_segmentation_writer_appends_frame_and_merges_label_map(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(rgb_path)
    result = run_closed_set_segmentation(
        rgb_path,
        tmp_path / "artifacts",
        observation_id="obs_0001",
        backend=FakeSegmentationBackend(np.zeros((2, 2), dtype=np.int32), {0: "road"}),
    )

    ref = append_segmentation_result(tmp_path, result)

    assert ref is not None
    frames = [json.loads(line) for line in (tmp_path / "memory" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert frames[0]["observation_id"] == "obs_0001"
    assert frames[0]["modality"] == "segmentation"
    label_map = json.loads((tmp_path / "memory" / "map" / "mask_label_map.json").read_text(encoding="utf-8"))
    assert label_map["colors"][SEMANTIC_CLASSES["route_surface"].color_key]["label"] == "route_surface"


def test_projection_prefers_segmentation_frame_over_sparse_object_mask(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    depth_path = tmp_path / "depth.npy"
    object_mask_path = tmp_path / "object_mask.png"
    segmentation_path = tmp_path / "segmentation.png"
    depth = np.full((8, 8), 8.0, dtype=np.float32)
    np.save(depth_path, depth)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(object_mask_path)
    Image.new("RGB", (8, 8), SEMANTIC_CLASSES["route_surface"].color).save(segmentation_path)
    label_dir = tmp_path / "memory" / "map"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_map = {
        "colors": {
            "255,0,0": {"id": 2, "label": "object_mask_building", "category": "debug"},
            SEMANTIC_CLASSES["route_surface"].color_key: {
                "id": SEMANTIC_CLASSES["route_surface"].semantic_id,
                "label": "route_surface",
                "category": "navigation_surface",
            },
        }
    }
    (label_dir / "mask_label_map.json").write_text(json.dumps(label_map), encoding="utf-8")
    writer.write_observation(
        {
            "scene_instance_id": "scene",
            "agent_id": "uav",
            "modalities": ["depth", "mask", "segmentation", "pose"],
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            "frames": {
                "depth": {"path": str(depth_path), "mock": False, "fov_degrees": 90},
                "mask": {"path": str(object_mask_path), "mock": False, "fov_degrees": 90},
                "segmentation": {"path": str(segmentation_path), "mock": False, "fov_degrees": 90},
            },
            "source": "segmentation_priority_test",
        }
    )

    memory = FlightMemory("run", "map", "scene", "uav", memory_reader=MemoryRecordReader(tmp_path))
    records = read_custom_map_records(memory)
    layer = build_projected_semantic_layer(
        records,
        center_pose=PosePoint(x=0.0, y=0.0, yaw_deg=0.0),
        grid_size=20,
        meters_per_cell=1.0,
    )

    assert layer is not None
    assert any(SEMANTIC_CLASSES["route_surface"].semantic_id in row for row in layer.matrix)
    assert not any(2 in row for row in layer.matrix)
    assert layer.legend[SEMANTIC_CLASSES["route_surface"].semantic_id]["label"] == "route_surface"


def test_priority_pooling_prefers_vehicle_over_route_surface_background():
    selected, override = _select_cell_semantic(Counter({20: 100, 11: 10}), row=3, col=4)

    assert selected == 11
    assert override is not None
    assert override["majority_semantic_id"] == 20
    assert override["selected_semantic_id"] == 11


def test_distance_confidence_smoothstep_curve():
    assert distance_confidence(0.0) == 1.0
    assert distance_confidence(5.0) == 1.0
    assert distance_confidence(8.0) == 1.0
    assert distance_confidence(20.0) == pytest.approx(0.646, abs=0.001)
    assert distance_confidence(35.0) == 0.15
    assert distance_confidence(40.0) == 0.15


def test_confidence_pooling_allows_near_person_to_override_background():
    evidence = {
        10: {
            "count": 10,
            "weighted_score": 10.0,
            "confidence_sum": 10.0,
            "distance_sum": 50.0,
            "max_confidence": 1.0,
            "far_count": 0,
            "source_observation_ids": {"obs_1"},
        }
    }

    selected, override = _select_cell_semantic(Counter({20: 100, 10: 10}), row=3, col=4, evidence=evidence)

    assert selected == 10
    assert override is not None
    assert override["decision"] == "priority_override"
    assert override["selected_evidence"]["avg_confidence"] == 1.0


def test_confidence_pooling_suppresses_far_low_confidence_person():
    evidence = {
        10: {
            "count": 10,
            "weighted_score": 1.5,
            "confidence_sum": 1.5,
            "distance_sum": 400.0,
            "max_confidence": 0.15,
            "far_count": 10,
            "source_observation_ids": {"obs_1"},
        }
    }

    selected, decision = _select_cell_semantic(Counter({20: 100, 10: 10}), row=3, col=4, evidence=evidence)

    assert selected == 20
    assert decision is not None
    assert decision["decision"] == "low_confidence_suppressed"
    assert decision["suppressed_semantic_id"] == 10
    assert decision["evidence"]["avg_confidence"] == 0.15


def test_priority_pooling_keeps_background_when_object_is_too_sparse():
    selected, override = _select_cell_semantic(Counter({20: 100, 11: 2}), row=3, col=4)

    assert selected == 20
    assert override is None


def test_priority_pooling_prefers_building_over_route_surface_background():
    selected, override = _select_cell_semantic(Counter({20: 100, 30: 9}), row=3, col=4)

    assert selected == 30
    assert override is not None
    assert override["threshold"] == 9


def test_priority_pooling_prefers_new_navigation_risk_classes_over_background():
    for semantic_id in (12, 50, 60, 70, 30, 40, 61):
        selected, override = _select_cell_semantic(Counter({20: 100, semantic_id: 10}), row=3, col=4)

        assert selected == semantic_id
        assert override is not None


def test_priority_pooling_does_not_let_visual_clutter_override_route_surface():
    selected, override = _select_cell_semantic(Counter({20: 100, 80: 20}), row=3, col=4)

    assert selected == 20
    assert override is None


def test_priority_pooling_keeps_unknown_only_without_positive_semantics():
    selected, override = _select_cell_semantic(Counter({-2: 12}), row=3, col=4)

    assert selected == -2
    assert override is None


def test_projected_layer_records_priority_overrides(tmp_path):
    route_id = SEMANTIC_CLASSES["route_surface"].semantic_id
    vehicle_id = SEMANTIC_CLASSES["vehicle"].semantic_id
    segmentation = np.full((40, 40), route_id, dtype=np.int32)
    segmentation[13:27, 13:27] = vehicle_id
    _write_semantic_projection_observation(
        tmp_path,
        segmentation=segmentation,
        label_names={"route_surface": "navigation_surface", "vehicle": "obstacle"},
    )

    memory = FlightMemory("run", "map", "scene", "uav", memory_reader=MemoryRecordReader(tmp_path))
    records = read_custom_map_records(memory)
    layer = build_projected_semantic_layer(
        records,
        center_pose=PosePoint(x=0.0, y=0.0, yaw_deg=0.0),
        grid_size=20,
        meters_per_cell=20.0,
        stride=1,
    )

    assert layer is not None
    assert any(vehicle_id in row for row in layer.matrix)
    assert layer.projection_debug["cell_pooling_strategy"] == "priority_confidence_over_background_v1"
    assert layer.projection_debug["priority_override_count"] > 0


def test_projected_layer_keeps_full_debug_point_cloud_by_default(tmp_path):
    route_id = SEMANTIC_CLASSES["route_surface"].semantic_id
    segmentation = np.full((128, 128), route_id, dtype=np.int32)
    _write_semantic_projection_observation(
        tmp_path,
        segmentation=segmentation,
        label_names={"route_surface": "navigation_surface"},
    )

    memory = FlightMemory("run", "map", "scene", "uav", memory_reader=MemoryRecordReader(tmp_path))
    records = read_custom_map_records(memory)
    layer = build_projected_semantic_layer(
        records,
        center_pose=PosePoint(x=0.0, y=0.0, yaw_deg=0.0),
        grid_size=40,
        meters_per_cell=1.0,
    )

    assert layer is not None
    sampled_points = layer.projection_debug["sampled_points"]
    assert len(sampled_points) > 3000
    assert len(sampled_points) == layer.projection_debug["point_count"]
    assert layer.projection_debug["debug_point_limit"] is None
    assert sampled_points[0]["distance_m"] > 0
    assert 0.15 <= sampled_points[0]["distance_confidence"] <= 1.0
    assert layer.projection_debug["semantic_confidence_summary"][str(route_id)]["avg_confidence"] > 0
    assert layer.projection_debug["distance_confidence_config"]["function"] == "smoothstep"


def test_unknown_ade_label_projects_as_unknown_semantic_id(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(rgb_path)
    np.save(depth_path, np.full((8, 8), 8.0, dtype=np.float32))
    write_result = writer.write_observation(
        {
            "scene_instance_id": "scene",
            "agent_id": "uav",
            "modalities": ["rgb", "depth", "pose"],
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            "frames": {
                "rgb": {"path": str(rgb_path), "mock": False},
                "depth": {"path": str(depth_path), "mock": False, "fov_degrees": 90},
            },
            "source": "unknown_projection_test",
        }
    )
    result = run_closed_set_segmentation(
        rgb_path,
        tmp_path / "artifacts",
        observation_id=write_result["observation_ref"]["id"],
        backend=FakeSegmentationBackend(np.full((8, 8), 9, dtype=np.int32), {9: "mystery ade class"}),
    )
    append_segmentation_result(tmp_path, result)

    memory = FlightMemory("run", "map", "scene", "uav", memory_reader=MemoryRecordReader(tmp_path))
    records = read_custom_map_records(memory)
    layer = build_projected_semantic_layer(
        records,
        center_pose=PosePoint(x=0.0, y=0.0, yaw_deg=0.0),
        grid_size=20,
        meters_per_cell=1.0,
    )

    assert result.stats["custom_class_pixel_counts"]["unknown"] == 64
    assert layer is not None
    assert any(-2 in row for row in layer.matrix)
    assert {point["semantic_id"] for point in layer.projection_debug["sampled_points"]} == {-2}


def test_explore_20_action_script_has_twenty_valid_steps():
    script = _action_script("explore_20_v1")

    assert len(script) == 20
    assert {item[0] for item in script}.issubset({"turn_relative", "move_body"})
    assert {item[1] for item in script}.issubset({"uav.turn_relative", "uav.move_body"})
    assert any(step[2].get("vz", 0.0) > 0 for step in script if step[0] == "move_body")
    assert any(step[2].get("vz", 0.0) < 0 for step in script if step[0] == "move_body")


def test_sweep_summary_aggregates_class_counts_and_unknown_ratios():
    summary = _build_sweep_summary(
        [
            {
                "class_name": "Move",
                "projection_point_count": 100,
                "sampled_point_count": 100,
                "priority_override_count": 2,
                "observation_count": 21,
                "action_count": 20,
                "semantic_point_counts_by_label": {"route_surface": 80, "unknown": 20},
                "segmentation_pixel_counts": {"route_surface": 800, "unknown": 200},
                "semantic_matrix_cell_counts_by_label": {"route_surface": 30, "unknown": 10},
            },
            {
                "class_name": "Shift",
                "projection_point_count": 50,
                "sampled_point_count": 50,
                "priority_override_count": 1,
                "observation_count": 21,
                "action_count": 20,
                "semantic_point_counts_by_label": {"building_structure": 40, "unknown": 10},
                "segmentation_pixel_counts": {"building_structure": 900, "unknown": 100},
                "semantic_matrix_cell_counts_by_label": {"building_structure": 25, "unknown": 5},
            },
        ]
    )

    assert summary["run_count"] == 2
    assert summary["totals"]["projection_point_count"] == 150
    assert summary["totals"]["sampled_point_count"] == 150
    assert summary["totals"]["action_count"] == 40
    assert summary["semantic_point_counts_by_label"]["unknown"] == 30
    assert summary["segmentation_pixel_counts"]["unknown"] == 300
    assert summary["semantic_matrix_cell_counts_by_label"]["unknown"] == 15
    assert summary["ratios"]["unknown_projection_ratio"] == 0.2
    assert summary["ratios"]["sampled_to_projected_ratio"] == 1.0


def test_huggingface_backend_classifies_supported_model_families():
    assert _model_family("nvidia/segformer-b2-finetuned-cityscapes-1024-1024") == "segformer"
    assert _model_family("facebook/mask2former-swin-large-cityscapes-semantic") == "mask2former"
    assert _model_family("shi-labs/oneformer_ade20k_dinat_large") == "oneformer"


def test_compare_paper_candidate_preset_excludes_previous_default_models():
    selected = _selected_models(["paper_candidates_4"])

    assert tuple(selected) == PAPER_CANDIDATE_MODELS
    assert "segformer_ade_b0" not in selected
    assert "segformer_cityscapes_b2" not in selected
    assert "mask2former_cityscapes_large" not in selected
    assert "mask2former_mapillary_large" not in selected
    assert {
        "segformer_ade_b5",
        "mask2former_ade_large",
        "oneformer_ade_swin_large",
        "oneformer_cityscapes_swin_large",
    }.issubset(selected)


def test_compare_stable_paper_candidate_preset_uses_runnable_models():
    selected = _selected_models(["paper_candidates_2"])

    assert tuple(selected) == STABLE_PAPER_CANDIDATE_MODELS
    assert set(selected) == {"segformer_ade_b5", "mask2former_ade_large"}
    assert "oneformer_ade_swin_large" not in selected
    assert "oneformer_cityscapes_swin_large" not in selected


def test_closed_set_segmentation_uses_model_specific_output_dirs(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(rgb_path)
    backend = FakeSegmentationBackend(np.zeros((2, 2), dtype=np.int32), {0: "road"})

    first = run_closed_set_segmentation(
        rgb_path,
        tmp_path / "artifacts",
        observation_id="obs_0001",
        config=ClosedSetSegmentationConfig(source="model_a", mask_dir_name="semantic_mask_model_a", overlay_dir_name="semantic_overlay_model_a"),
        backend=backend,
    )
    second = run_closed_set_segmentation(
        rgb_path,
        tmp_path / "artifacts",
        observation_id="obs_0001",
        config=ClosedSetSegmentationConfig(source="model_b", mask_dir_name="semantic_mask_model_b", overlay_dir_name="semantic_overlay_model_b"),
        backend=backend,
    )

    assert first.mask_path.parent.name == "semantic_mask_model_a"
    assert second.mask_path.parent.name == "semantic_mask_model_b"
    assert first.mask_path != second.mask_path
    assert first.overlay_path.exists()
    assert second.overlay_path.exists()


def test_compare_variant_memory_excludes_existing_segmentation_without_touching_source(tmp_path):
    run_dir = tmp_path / "run"
    memory_dir = run_dir / "memory"
    memory_dir.mkdir(parents=True)
    rgb_path = run_dir / "rgb.png"
    depth_path = run_dir / "depth.npy"
    segmentation_path = run_dir / "old_segmentation.png"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(rgb_path)
    Image.new("RGB", (2, 2), (128, 64, 128)).save(segmentation_path)
    np.save(depth_path, np.ones((2, 2), dtype=np.float32))
    (memory_dir / "actions.jsonl").write_text("", encoding="utf-8")
    (memory_dir / "observations.jsonl").write_text("", encoding="utf-8")
    (memory_dir / "poses.jsonl").write_text("", encoding="utf-8")
    frames = [
        {"id": "rgb", "observation_id": "obs", "modality": "rgb", "path": str(rgb_path)},
        {"id": "depth", "observation_id": "obs", "modality": "depth", "path": str(depth_path)},
        {"id": "seg", "observation_id": "obs", "modality": "segmentation", "path": str(segmentation_path)},
    ]
    (memory_dir / "frames.jsonl").write_text("\n".join(json.dumps(item) for item in frames) + "\n", encoding="utf-8")

    variant_dir = tmp_path / "variant"
    _prepare_variant_memory(run_dir, variant_dir)

    source_frames = [json.loads(line) for line in (memory_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    variant_frames = [json.loads(line) for line in (variant_dir / "memory" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(item["modality"] == "segmentation" for item in source_frames)
    assert not any(item["modality"] == "segmentation" for item in variant_frames)
    assert {item["modality"] for item in variant_frames} == {"rgb", "depth"}


def test_compare_summary_aggregates_models_and_html_refs(tmp_path):
    summary = _build_compare_summary(
        tmp_path,
        [
            {
                "run_dir": "run_a",
                "model_slug": "segformer_cityscapes_b2",
                "projection_point_count": 100,
                "semantic_point_counts_by_label": {"route_surface": 80, "unknown": 20},
                "custom_map_html": "run_a/contexts/custom_map_compare/segformer_cityscapes_b2/latest_custom_map.html",
            },
            {
                "run_dir": "run_a",
                "model_slug": "mask2former_cityscapes_large",
                "projection_point_count": 100,
                "semantic_point_counts_by_label": {"route_surface": 95, "unknown": 5},
                "custom_map_html": "run_a/contexts/custom_map_compare/mask2former_cityscapes_large/latest_custom_map.html",
            },
        ],
    )

    assert summary["model_count"] == 2
    assert summary["run_count"] == 1
    assert summary["by_model"]["segformer_cityscapes_b2"]["unknown_projection_ratio"] == 0.2
    assert summary["by_model"]["mask2former_cityscapes_large"]["unknown_projection_ratio"] == 0.05
    assert summary["results"][0]["custom_map_html"].endswith("latest_custom_map.html")
