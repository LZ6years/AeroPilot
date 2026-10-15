"""Geometry, provenance and archive regressions for the simple map renderer.

Uses synthetic observations and archived snapshots. No Provider, simulator,
ground-truth navigation data or pixel-snapshot comparison is required.
"""
import hashlib
import json
import math

from PIL import Image
import pytest

from plugins.capabilities.place_memory.layout import draw_map, IMAGE_SIZE, MAP_RECT, INSET_RECT
from plugins.capabilities.place_memory.render import render_map
from plugins.capabilities.place_memory.store import PlaceStore
from src.agent.context.media import MediaHydrator
from src.agent.context.request_assembler import RequestAssembler
from src.memory.context.contracts import MemorySourceBundle, TaskContextInput
from plugins.capabilities.place_memory.context_modules.map import PluginContextModule
from evaluation.benchmarks.aerial_vln.pixel_map_campaign import redraw_preview, controller_entry, OVERLAY
from tests.test_pixel_place_memory import flight


def test_map_has_only_spatial_labels_and_no_peripheral_teaching(tmp_path, monkeypatch):
    from PIL import ImageDraw
    texts = []
    original = ImageDraw.ImageDraw.text
    def capture(self, xy, text, *args, **kwargs):
        texts.append(str(text))
        return original(self, xy, text, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture)
    _, _, graph = measured_graph(tmp_path, [(0, 0, 0), (30, 0, 0)])
    layout = draw_map(graph)
    assert MAP_RECT[1] < 50
    assert any("P001" in text for text in texts)
    assert any("heading" in text for text in texts)
    assert not any("inspect_place" in text or "list_places" in text or "SIMPLE PLACE MAP" in text for text in texts)
    assert layout.hit_regions and layout.label_boxes


def measured_graph(path, positions):
    memory = flight(path, positions)
    store = PlaceStore(path, memory.active_instruction_id)
    graph = store.update("obs_0001", f"obs_{len(positions):04d}")
    return memory, store, graph


@pytest.mark.parametrize("yaw,direction", [(0, (0, -1)), (math.pi/2, (1, 0)),
                                          (-math.pi/2, (-1, 0)), (math.pi, (0, 1))])
def test_current_arrow_uses_actual_pose_and_fixed_heading_frame(tmp_path, yaw, direction):
    _, _, graph = measured_graph(tmp_path, [(0, 0, 0), (8, 0, 0)])
    graph["trajectory"][-1]["yaw_rad"] = yaw
    layout = draw_map(graph)
    assert len(graph["nodes"]) == 1  # current node still anchored at the start
    assert layout.current_position != layout.node_positions["P001"]
    assert layout.current_position == layout.projection.project(graph["trajectory"][-1]["position"])
    delta = tuple((b-a)/32 for a, b in zip(layout.current_position, layout.current_tip))
    assert delta == pytest.approx(direction)
    assert layout.inset_projection.project(graph["trajectory"][-1]["position"]) == layout.inset_projection.centre_px
    assert layout.inset_projection.pixels_per_m * 80 == pytest.approx(206)


def test_long_path_grid_uniform_scale_and_vertical_node_groups(tmp_path):
    _, _, graph = measured_graph(tmp_path, [(0, 0, 0), (0, 0, 10), (1200, 0, 10), (1200, 400, 10), (0, 0, 0)])
    layout = draw_map(graph)
    assert layout.image.size == IMAGE_SIZE
    assert layout.grid_spacing_m >= 10
    a, b, c = (layout.projection.project(p) for p in ((0, 0, 0), (10, 0, 0), (0, 10, 0)))
    assert math.dist(a, b) == pytest.approx(math.dist(a, c))
    assert any({"P001", "P002"}.issubset(hit["ids"]) for hit in layout.hit_regions)
    assert {pid for h in layout.hit_regions for pid in h["ids"]} == {n["id"] for n in graph["nodes"]}
    for point in layout.node_positions.values():
        assert MAP_RECT[0] < point[0] < MAP_RECT[2] and MAP_RECT[1] < point[1] < MAP_RECT[3]
        assert not (INSET_RECT[0] < point[0] < INSET_RECT[2] and INSET_RECT[1] < point[1] < INSET_RECT[3])


def test_dense_chinese_english_labels_are_bounded_and_not_invented(tmp_path):
    _, _, graph = measured_graph(tmp_path, [(i*12, (i%3)*16, 0) for i in range(25)])
    for index, node in enumerate(graph["nodes"]):
        node["summary"] = "这里检查过屋顶入口，右侧尚未确认。" * 12
        node["landmarks"] = ["Long narrow road beside unfinished concrete columns and a red entrance" if index % 2 else "红色入口旁边有尚未检查的狭窄道路与混凝土立柱" * 5]
    layout = draw_map(graph)
    assert layout.label_boxes
    assert graph["current_place_id"] in layout.label_boxes[0]["ids"]
    for index, label in enumerate(layout.label_boxes):
        x1, y1, x2, y2 = label["box"]
        assert MAP_RECT[0] <= x1 < x2 <= MAP_RECT[2]
        assert MAP_RECT[1] <= y1 < y2 <= MAP_RECT[3]
        notes = [n["landmarks"][0] for n in graph["nodes"] if n["id"] in label["ids"]]
        assert any(label["excerpt"].rstrip("…") in note for note in notes)
        for other in layout.label_boxes[index+1:]:
            a, b = label["box"], other["box"]
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


def test_map_only_request_has_context_label_and_camera_bounds_stay_intact(tmp_path):
    memory, _, _ = measured_graph(tmp_path, [(0, 0, 0)])
    result = PluginContextModule().build(MemorySourceBundle(memory), TaskContextInput("inspect"))
    assert result.fragment.content == {}
    assembler = RequestAssembler(MediaHydrator(base_dir=tmp_path))
    content = assembler.checkpoint_content({"fragments": [{"name":"map", "content": result.fragment.content}]},
                                          model_inputs=list(result.fragment.media_refs), timeline_role="CURRENT")
    text = "\n".join(p["text"] for p in content if p["type"] == "text")
    assert "CONTEXT IMAGE" in text and "image_size=1600x1100" in text
    assert all(term not in text for term in ("current_place", "is_annotated", "places", "scene_bounds", "CURRENT establishes"))
    camera = dict(result.fragment.media_refs[0]["media_ref"], latest_group=None, role="current_observation")
    content = assembler.checkpoint_content({}, model_inputs=[{"type":"image", "media_ref":camera}], timeline_role="CURRENT")
    assert "scene_bounds_in_display_pixels" in content[0]["text"]
    assert "CURRENT establishes" in content[0]["text"]


def test_renderer_version_preserves_old_image_and_html_group_targets(tmp_path):
    _, store, graph = measured_graph(tmp_path, [(0, 0, 0), (0, 0, 10)])
    old_digest = hashlib.sha256(json.dumps({k:v for k,v in graph.items() if k != "committed_calls"}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    old_path = store.directory / f"map_{old_digest}.png"
    old_path.write_bytes(b"previous evidence")
    first, page = render_map(store, graph)
    assert old_path.read_bytes() == b"previous evidence"
    path = tmp_path / first["media_ref"]["ref"]
    before = path.read_bytes()
    assert Image.open(path).size == IMAGE_SIZE
    assert render_map(store, graph)[0] == first and path.read_bytes() == before
    text = page.read_text(encoding="utf-8")
    assert 'id="group-0"' in text and 'href="#P001"' in text and 'href="#P002"' in text
    assert "left:" in text and "width:" in text and "data:image" not in text
    hit = draw_map(graph).hit_regions[0]
    assert f'left:{hit["box"][0]/1600*100:.6f}%' in text
    assert "\ufffd" not in text


def test_offline_redraw_never_updates_source_or_duplicates_photos(tmp_path):
    source, output = tmp_path / "old", tmp_path / "new"
    run = source / "attempt/runs/task"
    _, store, graph = measured_graph(run, [(0, 0, 0), (20, 0, 0)])
    render_map(store, graph)
    (source / "progress.json").write_text(json.dumps({"tasks":{"one":{"task":{"episode_id":"one"}, "status":"scored", "result":{"run_dir":str(run)}}}}), encoding="utf-8")
    before = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    result = redraw_preview(source, output)
    after = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    assert before == after and result["snapshots"] == 3
    assert len(list(output.rglob("*.png"))) == 3
    assert "未用于旧模型决策" in (output / "offline_preview/one/final/index.html").read_text(encoding="utf-8")
    assert "src/agent/context/request_assembler.py" in OVERLAY
    with pytest.raises(ValueError):
        redraw_preview(source, source)


def test_launch_uses_complete_verified_controller_snapshot(tmp_path):
    directory = tmp_path / "parallel/controller_source/revision"
    directory.mkdir(parents=True)
    script, recovery = directory / "parallel_campaign.py", directory / "parallel_recovery.py"
    script.write_text("# frozen controller", encoding="utf-8")
    recovery.write_text("# frozen recovery", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"fingerprint":"experiment"}), encoding="utf-8")
    overlay = {"base_fingerprint":"experiment", "controller_script":str(script),
               "source_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (script,recovery)}}
    path = tmp_path / "parallel/execution_overlay.json"
    path.write_text(json.dumps(overlay), encoding="utf-8")
    assert controller_entry(tmp_path) == script.resolve()
    recovery.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        controller_entry(tmp_path)
    overlay["base_fingerprint"] = "other"
    path.write_text(json.dumps(overlay), encoding="utf-8")
    with pytest.raises(ValueError, match="match"):
        controller_entry(tmp_path)
