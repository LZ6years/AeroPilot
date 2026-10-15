"""Saved directory, corrective photo feedback and historical pose contracts."""
import json
import math
import pytest

from plugins.capabilities.place_memory.store import PlaceStore, atomic_json, place_summary
from plugins.capabilities.place_memory.list_places.scripts.handler import handle as list_places
from plugins.capabilities.place_memory.inspect_place.scripts.handler import handle as inspect_place
from plugins.capabilities.place_memory.save_place.scripts.handler import handle as save_place
from src.plugins.contracts.skill import SkillCapabilityFailure
from tests.test_pixel_place_memory import flight
from tests.test_proactive_place_memory import save_args, save_context


def test_directory_saved_only_pagination_and_live_refresh(tmp_path):
    memory = flight(tmp_path, [(i * 20, 0, 0) for i in range(12)])
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    graph = store.update("obs_0001", "obs_0012")
    context = save_context(tmp_path, memory)
    payload = {"runtime_context": context, "args": {"offset": 0, "limit": 1}}
    assert list_places(payload).data["places"] == []
    for i in (10, 2, 1):
        node = graph["nodes"][i - 1]
        node.update(summary=f"Uncertain site {i}", landmarks=[])
    atomic_json(store.path, graph)
    before = store.path.read_bytes()
    result = list_places(payload)
    assert result.data["places"] == [{"place_id": "P001", "summary": "Uncertain site 1"}]
    assert result.data["total"] == 3 and result.data["next_offset"] == 1
    assert not result.model_inputs and store.path.read_bytes() == before
    assert [n["place_id"] for n in store.list_saved_places(1, 20)["places"]] == ["P002", "P010"]
    assert store.list_saved_places(50, 10)["places"] == []
    store.annotate(save_args(), context)
    assert list_places(payload).data["places"][0]["summary"] == save_args()["summary"]


def test_missing_corrupt_and_legacy_directory(tmp_path):
    store = PlaceStore(tmp_path, "task")
    with pytest.raises(ValueError):
        store.list_saved_places(0, 10)
    atomic_json(store.path, {"instruction_id": "task", "nodes": [
        {"id": "P001", "summary": "", "landmarks": ["possibly red"], "visits": []}]})
    assert store.list_saved_places(0, 10)["places"][0]["summary"] == "possibly red"
    store.path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        store.list_saved_places(0, 10)


def test_photo_receipt_rebuild_repair_and_historical_pose(tmp_path):
    memory = flight(tmp_path, [(0, 0, 0), (1, 0, 0), (20, 0, 0), (2, 0, 0)])
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    store.update("obs_0001", "obs_0004")
    context = save_context(tmp_path, memory)
    args = save_args()
    graph = store.annotate(args, context)
    assert place_summary(graph, graph["nodes"][0])["photo_observation_ids"] == ["obs_0001"]
    before = store.path.read_bytes()
    with pytest.raises(SkillCapabilityFailure) as error:
        save_place({"runtime_context": {**context, "tool_call_id": "bad"},
                    "args": {**args, "summary": "must not commit", "photo_observation_ids": ["obs_0003"]}})
    assert error.value.data["invalid_photo_observation_ids"] == ["obs_0003"]
    assert error.value.data["available_photo_total"] == 3
    assert store.path.read_bytes() == before
    graph = store.update("obs_0001", "obs_0004")
    assert place_summary(graph, graph["nodes"][0])["photo_observation_ids"] == ["obs_0001"]
    payload = {"runtime_context": context, "args": {"place_id": "P001"}}
    result = inspect_place(payload)
    assert result.data["photos"][0]["position_right_forward_up_m"] == [0, 0, 0]
    assert result.data["photos"][0]["uav_yaw_relative_initial_rad"] == 0
    assert result.data["evidence_refs"] == ["obs_0001"]
    store.annotate({**args, "summary": "Corrected", "photo_observation_ids": []},
                   {**context, "tool_call_id": "correction"})
    assert inspect_place(payload).data["summary"] == "Corrected"
    assert inspect_place({**payload, "args": {"place_id": "P002"}}).data["is_annotated"] is False
    # Missing historical pose stays unknown, never borrowed from a later visit.
    graph = store.load()
    graph["observations"]["obs_0001"].pop("position")
    graph["trajectory"] = []
    atomic_json(store.path, graph)
    assert inspect_place(payload).data["photos"][0]["position_right_forward_up_m"] is None


def test_selected_observation_uses_one_rgb_and_measured_historical_heading(tmp_path):
    memory = flight(tmp_path, [(0, 0, 0), (1, 2, 3)])
    pose_path = tmp_path / "memory/poses.jsonl"
    poses = [json.loads(line) for line in pose_path.read_text().splitlines()]
    for pose in poses:
        pose["pose"]["euler_rad"]["yaw"] = math.pi / 2 if pose["pose"]["position"]["x"] == 0 else math.pi
    pose_path.write_text("\n".join(json.dumps(p) for p in poses), encoding="utf-8")
    frame_path, obs_path = tmp_path / "memory/frames.jsonl", tmp_path / "memory/observations.jsonl"
    frames = [json.loads(line) for line in frame_path.read_text().splitlines()]
    observations = [json.loads(line) for line in obs_path.read_text().splitlines()]
    extra = {**frames[0], "id": "extra_rgb"}
    frames.append(extra)
    observations[0]["frame_refs"].append({"id": "extra_rgb"})
    frame_path.write_text("\n".join(json.dumps(f) for f in frames), encoding="utf-8")
    obs_path.write_text("\n".join(json.dumps(o) for o in observations), encoding="utf-8")
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    store.update("obs_0001", "obs_0002")
    args = {**save_args(), "photo_observation_ids": ["obs_0001", "obs_0002"]}
    context = save_context(tmp_path, memory)
    before = store.annotate(args, context)["nodes"][0]["photos"]
    assert store.update("obs_0001", "obs_0002")["nodes"][0]["photos"] == before
    photos = inspect_place({"runtime_context": context, "args": {"place_id": "P001"}}).data["photos"]
    assert photos[1]["position_right_forward_up_m"] == pytest.approx([-1, 2, 3])
    assert photos[1]["uav_yaw_relative_initial_rad"] == pytest.approx(math.pi / 2)
