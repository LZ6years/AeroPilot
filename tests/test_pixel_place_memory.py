"""Deterministic exact-pixel, task isolation, topology and archive contracts."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from plugins.capabilities.pixel_depth.measure_pixel_depth.scripts.handler import measure_pixel_depth
from plugins.capabilities.place_memory.store import PlaceStore
from plugins.capabilities.place_memory.render import render_map
from plugins.capabilities.place_memory.context_modules.map import PluginContextModule
from plugins.capabilities.place_memory.inspect_place.scripts.handler import handle as inspect_place
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.memory.flight_memory import FlightMemory
from src.memory.context.contracts import MemorySourceBundle, TaskContextInput
from src.plugins.core.discovery import PluginDiscovery
from src.plugins.core.registry import PluginRegistry
from src.tools.registry import default_registry


def observation(tmp_path, values=None):
    rgb = tmp_path / "rgb.png"; Image.new("RGB", (6, 4)).save(rgb)
    depth = tmp_path / "depth.npy"
    np.save(depth, values if values is not None else np.full((4, 6), 7.3, dtype=np.float32))
    return {"observation_id": "obs_1", "frames": {
        "rgb": {"path": str(rgb)}, "depth": {"path": str(depth), "metadata": {
            "depth_unit": "meter", "capture_source": "depth_perspective"}}}}


def test_exact_pixel_and_marker_without_retargeting(tmp_path):
    values = np.arange(24, dtype=np.float32).reshape(4, 6) + 0.31
    result = measure_pixel_depth(observation(tmp_path, values), "obs_1", 5, 3, run_dir=str(tmp_path), tool_call_id="c1")
    assert result.data["distance_m"] == 23.3
    assert result.data["depth_geometry"] == "ray_range"
    assert result.data["audit_details"]["raw_depth_m"] == pytest.approx(values[3, 5])
    assert len(result.model_inputs) == 1
    assert Image.open(result.artifact_refs[0]).size == (6, 4)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_center_not_replaced_by_valid_neighbors(tmp_path, value):
    a = np.ones((4, 6), dtype=np.float32) * 7
    a[2, 3] = value
    result = measure_pixel_depth(observation(tmp_path, a), "obs_1", 3, 2)
    assert result.data["distance_m"] is None
    assert result.data["distance_status"] == "invalid_depth_at_point"


@pytest.mark.parametrize("args", [("old", 1, 1), ("obs_1", -1, 0), ("obs_1", 6, 0), ("obs_1", True, 0), ("obs_1", 1.5, 0)])
def test_pixel_invalid_arguments(tmp_path, args):
    with pytest.raises(ValueError):
        measure_pixel_depth(observation(tmp_path), *args)


def test_pixel_rejects_dimension_and_unit_mismatch(tmp_path):
    obs = observation(tmp_path, np.ones((3, 6)))
    with pytest.raises(ValueError, match="dimensions"):
        measure_pixel_depth(obs, "obs_1", 0, 0)
    obs = observation(tmp_path)
    obs["frames"]["depth"]["metadata"]["depth_unit"] = "normalized"
    assert measure_pixel_depth(obs, "obs_1", 0, 0).data["distance_m"] is None


def flight(tmp_path, positions):
    writer = MemoryEventWriter(tmp_path)
    instruction = writer.write_instruction("inspect")
    memory = FlightMemory("r", "m", "s", "u", memory_reader=MemoryRecordReader(tmp_path))
    memory.update_instruction_ref(instruction, "inspect")
    for index, p in enumerate(positions):
        rgb = tmp_path / f"rgb_{index}.png"; Image.new("RGB", (16, 16), (index % 255, 0, 0)).save(rgb)
        if index:
            action = writer.write_action("uav.move_body", {}, {"ok": True, "data": {"pose": {"position": dict(zip("xyz", p)), "euler_rad": {"yaw": 0}}}})
            memory.append_action_ref(action["action_ref"]); memory.append_pose_ref(action["pose_ref"])
        result = writer.write_observation({"pose": {"position": dict(zip("xyz", p)), "euler_rad": {"yaw": 0}}, "frames": {"rgb": {"path": str(rgb)}}, "source": "test"})
        memory.append_observation_ref(result["observation_ref"])
        memory.append_pose_ref(result["pose_ref"]); memory.append_frame_refs(result["frame_refs"])
        memory.set_observation_anchor("task_initial", result["observation_ref"], overwrite=False)
    return memory


def test_topology_revisits_heights_long_history_and_photos(tmp_path):
    memory = flight(tmp_path, [(0, 0, 0), (20, 0, 0), (0, 0, 0), (0, 0, 10)] + [(i * 20, 0, 10) for i in range(1, 60)])
    module = PluginContextModule()
    result = module.build(MemorySourceBundle(memory), TaskContextInput("inspect"))
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    graph = store.load()
    assert len(graph["observations"]) == 63
    assert len(graph["trajectory"]) == 125  # includes intermediate measured actions
    assert graph["nodes"][0]["visits"] == ["obs_0001", "obs_0003"]
    assert len(graph["nodes"][0]["photos"]) == 2
    assert graph["nodes"][2]["position"][2] == 10
    assert graph["edges"][1]["to"] == "P001"
    assert result.fragment.media_refs[0]["media_ref"]["latest_group"] == "task_place_map"
    assert "data:image" not in (store.directory / "index.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        PlaceStore(tmp_path, "different_task").load()


def test_annotation_idempotence_immutable_map_and_historical_inspection(tmp_path):
    memory = flight(tmp_path, [(0, 0, 0), (20, 0, 0)])
    PluginContextModule().build(MemorySourceBundle(memory), TaskContextInput("inspect"))
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    first, _ = render_map(store, store.load())
    old_bytes = (tmp_path / first["media_ref"]["ref"]).read_bytes()
    args = {"place_id": "P001", "summary": "红色门", "landmarks": ["门"], "evidence_refs": ["obs_0001"], "photo_observation_ids": ["obs_0001"]}
    context = {"run_dir": str(tmp_path), "instruction_id": memory.active_instruction_id, "tool_call_id": "c1", "evidence_refs": ["obs_0001"]}
    graph = store.annotate(args, context)
    assert graph == store.annotate(args, context)
    second, _ = render_map(store, graph)
    assert first["media_ref"]["ref"] != second["media_ref"]["ref"]
    assert (tmp_path / first["media_ref"]["ref"]).read_bytes() == old_bytes
    inspected = inspect_place({"runtime_context": context, "args": {"place_id": "P001"}})
    assert inspected.data["evidence_origin"] == "historical_memory"
    assert inspected.model_inputs[0]["media_ref"]["observation_id"] == "obs_0001"
    with pytest.raises(ValueError, match="evidence"):
        store.annotate({**args, "evidence_refs": ["other_task"]}, {**context, "tool_call_id": "c2"})
    with pytest.raises(ValueError, match="photos"):
        store.annotate({**args, "photo_observation_ids": ["obs_0002"]}, {**context, "tool_call_id": "c3"})


def test_plugin_configuration_registers_only_new_tools():
    config = json.loads(Path("configs/plugins.pixel_map.json").read_text(encoding="utf-8"))
    discovered = PluginDiscovery(**config["plugins"]).discover()
    assert not discovered.warnings
    specs = default_registry(PluginRegistry.discover(config)).model_specs()
    names = {s.name for s in specs}
    assert {"skill.pixel_depth.measure_pixel_depth", "skill.place_memory.save_place", "skill.place_memory.inspect_place"} <= names
    assert "skill.visual_grounding.measure_visible_depth" not in names


@pytest.mark.parametrize("channels", [3, 4])
def test_airsim_raw_bgr_decode_preserves_color_and_alpha(tmp_path, channels):
    from types import SimpleNamespace
    from src.adapters.airsim.capture import AirSimCapture
    from src.perception.frame_payload import load_rgb_frame
    capture = AirSimCapture("front", "uav")
    # A red pixel from Unreal's raw BGR/BGRA transport, not an RGB fixture.
    raw = bytes([0, 0, 255] + ([128] if channels == 4 else []))
    response = SimpleNamespace(width=1, height=1, image_data_uint8=raw, image_data_float=[], time_stamp=123)
    frame = capture._frame_metadata("rgb", "scene", capture._capture_spec("scene"), response)
    assert load_rgb_frame(frame).getpixel((0, 0)) == (255, 0, 0)
    assert frame["camera_name"] == "front" and frame["capture_timestamp_ns"] == 123
