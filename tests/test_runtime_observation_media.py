"""Exercise original-media ownership through real Runtime and request assembly.

Fixtures use a scene-owned capture directory outside each episode, matching
reused simulators. No model service, graphics process or network is required.
"""

import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from plugins.capabilities.pixel_depth.measure_pixel_depth.scripts.handler import handle
from plugins.capabilities.place_memory.store import PlaceStore
from src.agent.context.media import MediaHydrator
from src.agent.context.projection import HistoryProjector
from src.agent.context.request_assembler import RequestAssembler
from src.env_client import Observation
from src.memory.stores import MemoryRecordReader
from src.perception.frame_payload import load_depth_frame, load_rgb_frame
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.context.observation_media import ObservationMediaStore
from src.runtime.factory import RuntimeFactory
from src.runtime.logging.async_run_logger import AsyncRunLogger
from tests.support.planner import StubPlanner, response_with_calls
from tests.support.runtime import harness, move, run_loop
from tests.test_unified_tool_loop import ManagedFakeBackend


def capture(folder, index=1):
    """Write aligned, color-preserving PNG/NPY payloads with real metadata."""
    folder.mkdir(parents=True, exist_ok=True)
    rgb = folder / f"rgb_{index}.png"
    depth = folder / f"depth_{index}.npy"
    preview = folder / f"preview_{index}.png"
    Image.new("RGB", (8, 6), (20 + index, 40, 60)).save(rgb)
    np.save(depth, np.full((6, 8), 7.0 + index, dtype=np.float32))
    Image.new("RGB", (8, 6), "gray").save(preview)
    common = {"width": 8, "height": 6, "capture_group": f"capture_{index}"}
    return {
        "rgb": {**common, "artifact_path": str(rgb), "capture_source": "scene"},
        "depth": {**common, "artifact_path": str(depth), "preview_path": str(preview),
                  "capture_source": "depth_perspective", "depth_unit": "meter"},
    }


@pytest.mark.parametrize("relative", [False, True])
def test_shared_capture_import_is_exact_deduplicated_and_does_not_mutate_input(tmp_path, monkeypatch, relative):
    monkeypatch.chdir(tmp_path)
    frames = capture(Path("shared") if relative else tmp_path / "shared")
    original = copy.deepcopy(frames)
    originals = {path: path.read_bytes() for path in (tmp_path / "shared").iterdir()}
    store = ObservationMediaStore(tmp_path / "episode")
    imported = store.materialize_frames(frames)
    for modality in ("rgb", "depth"):
        item = imported[modality]
        path = Path(item["artifact_path"])
        source = Path(frames[modality]["artifact_path"])
        assert path.is_relative_to(store.run_dir)
        assert path == Path(item["path"])
        assert path.read_bytes() == source.read_bytes()
        assert item["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert item["source_artifact_path"] == str(source.resolve())
        assert item["capture_group"] == frames[modality]["capture_group"]
    assert imported["depth"]["depth_unit"] == "meter"
    assert Path(imported["depth"]["preview_path"]).read_bytes() == Path(frames["depth"]["preview_path"]).read_bytes()
    written = {path: path.stat().st_mtime_ns for path in store.root.rglob("*") if path.is_file()}
    assert store.materialize_frames(frames) == imported
    assert written == {path: path.stat().st_mtime_ns for path in store.root.rglob("*") if path.is_file()}
    assert frames == original
    assert all(path.read_bytes() == raw for path, raw in originals.items())


def test_existing_run_media_is_reused_without_another_copy(tmp_path):
    frames = capture(tmp_path / "artifacts" / "scene")
    store = ObservationMediaStore(tmp_path)
    imported = store.materialize_frames(frames)
    assert imported["rgb"]["artifact_path"] == str(Path(frames["rgb"]["artifact_path"]).resolve())
    assert imported["depth"]["preview_path"] == frames["depth"]["preview_path"]
    assert not store.root.exists()


def test_inline_only_payloads_survive_source_removal(tmp_path):
    frames = capture(tmp_path / "capture")
    for frame in frames.values():
        path = Path(frame.pop("artifact_path"))
        frame["base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        path.unlink()
    preview = Path(frames["depth"].pop("preview_path"))
    preview.unlink()
    imported = ObservationMediaStore(tmp_path / "episode").materialize_frames(frames)
    assert load_rgb_frame(imported["rgb"]).getpixel((0, 0)) == (21, 40, 60)
    assert np.all(load_depth_frame(imported["depth"], require_metric=True) == 8.0)


def test_genuinely_missing_images_and_mock_metadata_are_preserved(tmp_path):
    frames = {"rgb": {"mock": True, "artifact_path": "not-a-real-image.png"},
              "depth": {"width": 8, "height": 6}}
    store = ObservationMediaStore(tmp_path)
    assert store.materialize_frames(frames) == frames
    assert store.materialize_frames({}) == {}
    assert not store.root.exists()


@pytest.mark.parametrize("failure", ["missing", "bad_base64", "corrupt_image", "wrong_dimensions", "bad_depth", "missing_preview"])
def test_supplied_invalid_media_fails_explicitly(tmp_path, failure):
    frames = capture(tmp_path / "shared")
    rgb = frames["rgb"]
    if failure == "missing":
        Path(rgb["artifact_path"]).unlink()
    elif failure == "bad_base64":
        rgb.pop("artifact_path")
        rgb["base64"] = "!!!"
    elif failure == "corrupt_image":
        Path(rgb["artifact_path"]).write_bytes(b"not an image")
    elif failure == "wrong_dimensions":
        rgb["width"] = 9
    elif failure == "bad_depth":
        np.save(frames["depth"]["artifact_path"], np.array(["invalid"], dtype="U7"))
    else:
        Path(frames["depth"]["preview_path"]).unlink()
    with pytest.raises((ValueError, OSError)):
        ObservationMediaStore(tmp_path / "episode").materialize_frames(frames)


def test_disk_failure_leaves_no_published_or_temporary_media(tmp_path, monkeypatch):
    import src.runtime.context.observation_media as media

    frames = capture(tmp_path / "shared")
    store = ObservationMediaStore(tmp_path / "episode")
    def fail_replace(*args):
        raise PermissionError("test disk write denied")
    monkeypatch.setattr(media.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="disk write denied"):
        store.materialize_frames(frames)
    assert not any(path.is_file() for path in store.root.rglob("*"))
    assert Path(frames["rgb"]["artifact_path"]).is_file()


def test_failed_import_does_not_commit_an_observation(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [])
    before = (tmp_path / "memory/observations.jsonl").read_bytes()
    frames = {"rgb": {"artifact_path": str(tmp_path / "absent.png")}}
    backend.get_observation = lambda *args: Observation("scene", "uav", ["rgb"], {"frames": frames})
    with pytest.raises(ValueError, match="no readable artifact"):
        runtime.evidence.record_observation(runtime.session, ["rgb"], "instruction_start")
    assert (tmp_path / "memory/observations.jsonl").read_bytes() == before
    assert state.latest_observation["observation_id"] == "obs_0001"


def test_after_action_import_failure_preserves_execution_receipts(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [[move("move_once")]])
    frames = {"rgb": {"artifact_path": str(tmp_path / "missing-capture.png")}}
    backend.get_observation = lambda *args: Observation("scene", "uav", ["rgb"], {"frames": frames})
    with pytest.raises(ValueError, match="no readable artifact"):
        run_loop(runtime, state, scene)
    results = [event.payload for event in runtime.agent_transcript.items if event.kind == "tool_result"]
    assert [(result.tool_call_id, result.status) for result in results] == [("move_once", "ok")]
    assert len([action for action, _ in backend.actions if action == "move_body"]) == 1
    window = next(payload for name, payload in runtime.recorder.events if name == "physical_window_recorded")
    assert window["status"] == "error" and window["observation_after_id"] is None
    assert window["observation_error"]
    assert not state.completed


def test_two_runtimes_hydrate_shared_rgb_and_keep_depth_and_place_photos(tmp_path):
    """Exercise real request assembly, then remove every shared source file."""
    shared = tmp_path / "scene_session_runtime" / "artifacts"

    class CaptureBackend(ManagedFakeBackend):
        def get_observation(self, scene, agent, modalities):
            self.observations += 1
            return Observation(scene, agent, modalities, {
                "pose": dict(self.pose), "frames": capture(shared, self.observations)})

    class TwoTurnPlanner(StubPlanner):
        def __init__(self):
            super().__init__({})
            self.responses = iter([
                response_with_calls(("move_once", "uav_move_body", {"vx": 1, "vy": 0, "vz": 0, "duration": 2})),
                response_with_calls(("done", "runtime_report_completed", {
                    "reason": "Endpoint observed", "evidence_refs": ["obs_0002"]})),
            ])

        def _request(self, payload):
            self.response = next(self.responses)
            return super()._request(payload)

        def plan_turn(self, transcript, registry, **kwargs):
            self.transcript = transcript
            return super().plan_turn(transcript, registry, **kwargs)

    runs = []
    for name in ("episode_a", "episode_b"):
        run_dir = tmp_path / name
        planner = TwoTurnPlanner()
        recorder = AsyncRunLogger(run_dir)
        runtime = RuntimeFactory("configs/plugins.pixel_map.json").create_task_runtime(
            CaptureBackend(), recorder, planner=planner,
            stabilization_config=UavStabilizationConfig(enabled=False), shutdown_policy="none")
        try:
            state = runtime.run_task("map", "Inspect the view and move two meters")
        finally:
            recorder.close()
        assert state.completed and state.physical_window_count == 1
        assert len(planner.payloads) == 2
        checkpoints = [event.payload for event in planner.transcript.items if event.kind == "state_checkpoint"]
        for checkpoint in checkpoints:
            image = next(fragment for fragment in checkpoint.fragments if fragment.name == "image")
            assert not image.content and len(image.media_refs) == 1
        second_request = json.dumps(planner.payloads[1])
        assert "HISTORY" in second_request and "obs_0001" in second_request
        assert "CURRENT" in second_request and "obs_0002" in second_request
        assert "current_rgb_unavailable" not in second_request
        graph = PlaceStore(run_dir, "inst_0001").load()
        assert all(record["photos"] for record in graph["observations"].values())
        runs.append((run_dir, runtime, planner, graph))

    for path in shared.iterdir():
        path.unlink()
    for run_dir, runtime, planner, graph in runs:
        reader = MemoryRecordReader(run_dir)
        latest = reader.read_observation("obs_0003")
        measured = handle({"args": {"observation_id": "obs_0001", "pixel_x": 3, "pixel_y": 2},
            "runtime_context": {"run_dir": str(run_dir), "tool_call_id": "historical_depth",
                                "latest_observation": latest, "evidence_refs": ["obs_0001"]}})
        assert measured.data["distance_m"] == 8.0
        assert measured.data["observation_role"] == "HISTORY"
        assert latest["observation_id"] == "obs_0003"
        for observation in graph["observations"].values():
            for photo in observation["photos"]:
                with Image.open(run_dir / photo["ref"]) as image:
                    assert image.size == (8, 6)
        projected = HistoryProjector(assembler=RequestAssembler(MediaHydrator(base_dir=run_dir)),
            media_policy={"max_request_images": 20, "max_tool_evidence_images": 4,
                          "keep_recent_observations": 3, "pin_instruction_start": True,
                          "pin_current": True}).project(planner.transcript, runtime.tools.registry)
        selected = projected.audit["selected_visual_evidence"]
        assert len([item for item in selected if str(item["frame_id"]).startswith("frame_")]) == 3
        assert len([item for item in selected if str(item["frame_id"]).startswith("place_map_")]) == 1
        assert projected.audit["total_image_count"] <= 20
        assert projected.audit["missing_images"] == []
        assert "data:image/png;base64," in json.dumps(projected.messages)
        assert "data:image/" not in (run_dir / "memory/agent_transcript.jsonl").read_text(encoding="utf-8")
