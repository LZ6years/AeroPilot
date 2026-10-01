"""Tests for the research.custom_map context plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plugins.research.custom_map.context_modules.custom_map import PluginContextModule
from plugins.research.custom_map.context_modules.custom_map.mask_semantics import build_semantic_layer
from plugins.research.custom_map.context_modules.custom_map.projection import build_projected_semantic_layer
from plugins.research.custom_map.context_modules.custom_map.projection_geometry import (
    DEPTH_MODE_OPTICAL_Z,
    DEPTH_MODE_RAY_DISTANCE,
    camera_intrinsics,
    project_depth_pixel,
)
from plugins.research.custom_map.context_modules.custom_map.records import read_custom_map_records
from plugins.research.custom_map.context_modules.custom_map.schemas import PosePoint
from plugins.research.custom_map.context_modules.custom_map.trajectory import build_trajectory
from src.memory.context.registry import ContextModuleConfigError, ContextModuleRegistry
from src.memory.context.contracts import MemorySourceBundle, TaskContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.plugins.core.discovery import PluginDiscovery
from src.plugins.core.registry import PluginRegistry
from src.plugins.context import load_plugin_context_module


def test_custom_map_declares_semantic_evidence_boundary():
    evidence = PluginContextModule().describe_output()["semantic_evidence"]

    assert len(evidence) == 1
    assert "semantic cells and trajectory" in evidence[0]
    assert "not current visual truth or verified free space" in evidence[0]


def _memory(tmp_path: Path) -> FlightMemory:
    reader = MemoryRecordReader(tmp_path)
    return FlightMemory(
        "run_custom_map",
        "map_1",
        "scene_1",
        "uav_1",
        active_instruction="Fly past the building.",
        memory_reader=reader,
    )


def _stage_context() -> TaskContextInput:
    return TaskContextInput(
        active_instruction="Fly past the building.",
    )


def _write_pose_action_history(tmp_path: Path, memory: FlightMemory) -> None:
    writer = MemoryEventWriter(tmp_path)
    for index, x in enumerate([0.0, 5.0, 10.0]):
        result = writer.write_action(
            "uav.move_body",
            {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 1.0},
            {"ok": True, "data": {"pose": {"x": x, "y": 0.0, "z": -2.0, "yaw_deg": 0.0}}},
        )
        memory.append_action_ref(result["action_ref"])
        memory.append_pose_ref(result["pose_ref"])
    failed = writer.write_action(
        "uav.move_body",
        {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 1.0},
        {"ok": False, "data": {"pose": {"x": 10.0, "y": 5.0, "z": -2.0, "yaw_deg": 0.0}}},
    )
    memory.append_action_ref(failed["action_ref"])
    memory.append_pose_ref(failed["pose_ref"])


def _write_mask_observation(tmp_path: Path, memory: FlightMemory, *, with_label: bool) -> Path:
    mask_path = tmp_path / "mask.png"
    image = Image.new("RGB", (40, 40), (0, 0, 0))
    for y in range(20):
        for x in range(40):
            image.putpixel((x, y), (255, 0, 0))
    image.save(mask_path)
    if with_label:
        label_dir = tmp_path / "memory" / "map"
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / "mask_label_map.json").write_text(
            json.dumps(
                {
                    "colors": {
                        "255,0,0": {
                            "id": 2,
                            "label": "building",
                            "category": "landmark",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
    writer = MemoryEventWriter(tmp_path)
    obs = writer.write_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["mask"],
            "frames": {"mask": {"path": str(mask_path), "mock": False}},
            "source": "test",
        }
    )
    memory.append_observation_ref(obs["observation_ref"])
    memory.append_frame_refs(obs["frame_refs"])
    return mask_path


def _write_depth_mask_observation(tmp_path: Path, *, yaw_deg: float = 0.0, with_label: bool = True) -> None:
    mask_path = tmp_path / f"mask_{int(yaw_deg)}.png"
    depth_path = tmp_path / f"depth_{int(yaw_deg)}.npy"
    image = Image.new("RGB", (40, 40), (0, 0, 0))
    depth = np.zeros((40, 40), dtype=np.float32)
    for y in range(14, 26):
        for x in range(16, 24):
            image.putpixel((x, y), (255, 0, 0))
            depth[y, x] = 8.0
    image.save(mask_path)
    np.save(depth_path, depth)
    if with_label:
        label_dir = tmp_path / "memory" / "map"
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / "mask_label_map.json").write_text(
            json.dumps({"colors": {"255,0,0": {"id": 2, "label": "building", "category": "landmark"}}}),
            encoding="utf-8",
        )
    MemoryEventWriter(tmp_path).write_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["depth", "mask", "pose"],
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": yaw_deg},
            "frames": {
                "depth": {"path": str(depth_path), "mock": False, "fov_degrees": 90},
                "mask": {"path": str(mask_path), "mock": False, "fov_degrees": 90},
            },
            "source": "projection_test",
        }
    )


def _write_unknown_depth_mask_observation(
    tmp_path: Path,
    name: str,
    *,
    colors: list[tuple[int, int, int]],
    x: float = 0.0,
) -> None:
    mask_path = tmp_path / f"{name}_mask.png"
    depth_path = tmp_path / f"{name}_depth.npy"
    image = Image.new("RGB", (40, 40), (0, 0, 0))
    depth = np.zeros((40, 40), dtype=np.float32)
    for index, color in enumerate(colors):
        x0 = 14 + index * 8
        x1 = x0 + 6
        for y in range(16, 24):
            for px in range(x0, x1):
                image.putpixel((px, y), color)
                depth[y, px] = 8.0
    image.save(mask_path)
    np.save(depth_path, depth)
    MemoryEventWriter(tmp_path).write_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["depth", "mask", "pose"],
            "pose": {"x": x, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
            "frames": {
                "depth": {"path": str(depth_path), "mock": False, "fov_degrees": 90},
                "mask": {"path": str(mask_path), "mock": False, "fov_degrees": 90},
            },
            "source": "projection_test",
        }
    )


def test_plugin_discovery_supports_package_style_context_modules():
    result = PluginDiscovery(
        roots=["plugins/research"],
        enabled=["research.custom_map"],
    ).discover()

    by_plugin_slot = {(item.plugin_id, item.slot): item for item in result.contributions.context_modules}

    custom_map = by_plugin_slot[("research.custom_map", "custom_map")]
    assert custom_map.module_path == "plugins.research.custom_map.context_modules.custom_map"
    assert custom_map.source_path.name == "__init__.py"


def test_removed_multimodal_image_provider_is_not_implicitly_restored():
    with pytest.raises(ContextModuleConfigError, match="research.multimodal_image"):
        ContextModuleRegistry.from_dict(
            {"context_modules": {"image": "research.multimodal_image"}}
        )


def test_custom_map_builds_fragment_image_matrix_and_path_summary(tmp_path):
    memory = _memory(tmp_path)
    _write_pose_action_history(tmp_path, memory)
    _write_mask_observation(tmp_path, memory, with_label=True)

    result = PluginContextModule().build(MemorySourceBundle(memory), _stage_context())
    fragment = result.fragment.to_dict()
    content = fragment["content"]

    assert fragment["name"] == "custom_map"
    assert content["path_summary"]["pose_count"] == 4
    assert content["path_summary"]["failed_action_count"] == 1
    assert content["center_pose"]["x"] == 10.0
    assert content["semantic_legend"]["-1"]["label"] == "visited_path"
    assert content["semantic_legend"]["0"]["label"] == "free"
    assert content["semantic_legend"]["2"]["label"] == "building"
    assert any(2 in row for row in content["semantic_matrix"])
    assert any(-1 in row for row in content["semantic_matrix"])
    assert content["map_config"]["semantic_projection"] == "mask_downsampled_without_depth_projection"
    assert result.audit.metadata["html_report"].endswith("latest_custom_map.html")
    assert fragment["media_refs"][0]["type"] == "image"
    assert fragment["media_refs"][0]["media_ref"]["role"] == "custom_map"
    rendered_path = tmp_path / fragment["media_refs"][0]["media_ref"]["ref"]
    html_path = Path(result.audit.metadata["html_report"])
    assert rendered_path.exists()
    assert html_path.exists()

    image = Image.open(rendered_path).convert("RGB")
    assert image.size == (900, 560)
    non_background = sum(1 for pixel in image.getdata() if pixel != (248, 250, 252))
    assert non_background > 1000
    assert _has_green_uav_marker(image)
    assert _has_blue_path_marker(image)
    html = html_path.read_text(encoding="utf-8")
    assert "CUSTOM_MAP_DATA" in html
    assert "<canvas id=\"pointCloud\"" in html
    assert "semanticClassSelector" in html
    assert "Semantic Class Gallery" in html
    assert "building" in html
    assert "mask_downsampled_without_depth_projection" in html


def test_custom_map_keeps_unknown_mask_colors_unknown(tmp_path):
    memory = _memory(tmp_path)
    _write_pose_action_history(tmp_path, memory)
    _write_mask_observation(tmp_path, memory, with_label=False)

    records = read_custom_map_records(memory)
    semantic_layer = build_semantic_layer(records)

    assert semantic_layer.status == "mask_downsampled_without_depth_projection"
    assert semantic_layer.legend[-2]["label"] == "unknown"
    assert any(-2 in row for row in semantic_layer.matrix)
    assert any(warning.startswith("custom_map_mask_label_missing:255,0,0") for warning in semantic_layer.warnings)


def test_depth_mask_projection_keeps_unknown_color_ids_stable_across_frames(tmp_path):
    memory = _memory(tmp_path)
    green = (0, 255, 0)
    _write_unknown_depth_mask_observation(tmp_path, "first", colors=[(255, 0, 0), green])
    _write_unknown_depth_mask_observation(tmp_path, "second", colors=[green], x=0.1)

    records = read_custom_map_records(memory)
    layer = build_projected_semantic_layer(
        records,
        center_pose=PosePoint(x=0.0, y=0.0, yaw_deg=0.0),
        grid_size=20,
        meters_per_cell=1.0,
    )

    assert layer is not None
    green_ids = {
        item["semantic_id"]
        for item in layer.projection_debug["sampled_points"]
        if item["color"] == [0, 255, 0]
    }
    assert green_ids == {-2}
    assert layer.legend[-2]["label"] == "unknown"


def test_depth_mask_projection_places_semantic_region_in_front_cells(tmp_path):
    memory = _memory(tmp_path)
    _write_depth_mask_observation(tmp_path, yaw_deg=0.0)

    records = read_custom_map_records(memory)
    layer = build_projected_semantic_layer(
        records,
        center_pose=PosePoint(x=0.0, y=0.0, yaw_deg=0.0),
        grid_size=20,
        meters_per_cell=1.0,
    )

    assert layer is not None
    assert layer.status == "depth_mask_projected_to_topdown"
    assert layer.legend[2]["label"] == "building"
    assert layer.projection_debug["point_count"] > 0
    assert layer.projection_debug["sampled_points"][0]["semantic_id"] == 2
    assert {"pixel_u", "pixel_v", "raw_depth_m", "camera_x", "camera_y", "camera_z", "local_forward_m", "local_right_m", "local_up_m"} <= set(layer.projection_debug["sampled_points"][0])
    assert layer.projection_debug["semantic_point_counts"]["2"] > 0
    assert layer.projection_debug["semantic_source_observations"]["2"]
    assert layer.projection_debug["semantic_classes"][0]["label"] == "building"
    assert layer.projection_debug["source_frames_by_observation_id"]
    assert layer.projection_debug["depth_interpretation_used"] == DEPTH_MODE_OPTICAL_Z
    assert layer.projection_debug["depth_interpretation_comparison"]["comparison_mode"] == DEPTH_MODE_RAY_DISTANCE
    assert layer.projection_debug["depth_interpretation_comparison"]["samples"]
    assert layer.projection_debug["frame_point_ranges"]
    assert "multi_frame_projection_drift" in layer.projection_debug
    projected_cells = [(row_index, col_index) for row_index, row in enumerate(layer.matrix) for col_index, value in enumerate(row) if value == 2]
    assert projected_cells
    assert min(row for row, _ in projected_cells) < 10


def test_custom_map_html_contains_projected_point_cloud_debug(tmp_path):
    memory = _memory(tmp_path)
    _write_pose_action_history(tmp_path, memory)
    _write_depth_mask_observation(tmp_path, yaw_deg=0.0)

    result = PluginContextModule().build(MemorySourceBundle(memory), _stage_context())
    fragment = result.fragment.to_dict()
    content = fragment["content"]

    assert content["map_config"]["semantic_projection"] == "depth_mask_projected_to_topdown"
    projection_debug = result.audit.metadata["projection_debug"]
    assert projection_debug["point_count"] > 0
    assert projection_debug["sampled_points"]
    assert projection_debug["semantic_point_counts"]["2"] > 0
    assert projection_debug["semantic_point_ratios"]["2"] > 0
    assert "semantic_raw_label_examples" in projection_debug
    assert projection_debug["semantic_classes"][0]["label"] == "building"
    assert projection_debug["source_frames_by_observation_id"]
    html_path = Path(result.audit.metadata["html_report"])
    html = html_path.read_text(encoding="utf-8")
    assert "CUSTOM_MAP_DATA" in html
    assert "depth_mask_projected_to_topdown" in html
    assert "pointCloud" in html
    assert "semanticClassSelector" in html
    assert "classGallery" in html
    assert "sourceFrames" in html
    assert "semantic_point_counts" in html
    assert "semantic_point_ratios" in html
    assert "semantic_raw_label_examples" in html
    assert "Depth Projection Diagnostics" in html
    assert "depthModeCompare" in html
    assert "depth_interpretation_comparison" in html
    assert "building" in html


def test_depth_projection_modes_are_distinct_for_off_center_pixels():
    fx, fy, cx, cy = camera_intrinsics(40, 40, 90.0)
    pose = PosePoint(x=0.0, y=0.0, z=0.0, yaw_deg=0.0)

    optical = project_depth_pixel(
        pixel_u=30,
        pixel_v=10,
        raw_depth_m=8.0,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        pose=pose,
        center_pose=pose,
        depth_mode=DEPTH_MODE_OPTICAL_Z,
    )
    ray = project_depth_pixel(
        pixel_u=30,
        pixel_v=10,
        raw_depth_m=8.0,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        pose=pose,
        center_pose=pose,
        depth_mode=DEPTH_MODE_RAY_DISTANCE,
    )

    assert optical["camera_z"] == 8.0
    assert ray["camera_z"] < optical["camera_z"]
    assert ray["distance_m"] == 8.0
    assert optical["local_forward_m"] > ray["local_forward_m"]


def test_depth_mask_projection_rotates_observation_yaw_into_world_y(tmp_path):
    memory = _memory(tmp_path)
    _write_depth_mask_observation(tmp_path, yaw_deg=90.0)

    records = read_custom_map_records(memory)
    layer = build_projected_semantic_layer(
        records,
        center_pose=PosePoint(x=0.0, y=0.0, yaw_deg=0.0),
        grid_size=20,
        meters_per_cell=1.0,
    )

    assert layer is not None
    projected_cells = [(row_index, col_index) for row_index, row in enumerate(layer.matrix) for col_index, value in enumerate(row) if value == 2]
    assert projected_cells
    assert max(col for _, col in projected_cells) > 10


def test_trajectory_compresses_actions_without_raw_log_exposure(tmp_path):
    memory = _memory(tmp_path)
    _write_pose_action_history(tmp_path, memory)

    records = read_custom_map_records(memory)
    trajectory = build_trajectory(records)

    assert trajectory.summary["pose_count"] == 4
    assert trajectory.summary["approx_path_distance_m"] == 15.0
    assert trajectory.summary["history_compressed"] is True
    assert len(trajectory.segments) == 2
    assert trajectory.segments[0]["kind"] == "move"
    assert trajectory.segments[0]["count"] == 3
    assert trajectory.segments[1]["ok"] is False
    assert len(trajectory.failed_markers) == 1


def test_configured_registry_includes_custom_map_schema(tmp_path):
    memory = _memory(tmp_path)
    _write_pose_action_history(tmp_path, memory)
    registry = PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/research"],
                "enabled": ["research.custom_map"],
            },
            "context_modules": {
                "custom_map": "research.custom_map",
            },
        }
    )
    contribution = registry.context_module_for("custom_map", "research.custom_map")
    assert contribution is not None
    module_registry = ContextModuleRegistry.from_dict(
        {"context_modules": {"custom_map": "research.custom_map"}},
        plugin_modules={"custom_map": load_plugin_context_module(contribution)},
    )

    result = module_registry.build_fragments(
        MemorySourceBundle(memory), _stage_context()
    )[0]

    assert result.fragment.name == "custom_map"
    schema = module_registry.describe_modules()["custom_map"]["model_fields"]
    assert "semantic_matrix" in schema
    assert schema["semantic_matrix"] == PluginContextModule().describe_output()["model_fields"]["semantic_matrix"]
    assert result.fragment.media_refs[0]["media_ref"]["role"] == "custom_map"


def _has_green_uav_marker(image: Image.Image) -> bool:
    return any(pixel[1] > 140 and pixel[0] < 80 and pixel[2] < 120 for pixel in image.getdata())


def _has_blue_path_marker(image: Image.Image) -> bool:
    return any(pixel[2] > 180 and pixel[0] < 80 and pixel[1] < 140 for pixel in image.getdata())
