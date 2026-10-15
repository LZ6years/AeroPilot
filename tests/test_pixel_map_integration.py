"""Production tool-loop and media-selection integration, without real services."""
from dataclasses import replace
import json

import pytest
from PIL import Image

from src.agent.context.media import MediaHydrator
from src.agent.context.projection import HistoryProjector, _candidate, _select_visual_candidates
from src.agent.context.request_assembler import RequestAssembler
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult
from src.env_client import Observation
from src.memory.context.media_refs import image_media_ref
from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from src.runtime.control.actions.terminal import environmental_evidence_refs
from tests.test_history_projection import _transcript, _add_image_checkpoint
from tests.test_unified_tool_loop import public_runtime, call, move, done
from evaluation.benchmarks.aerial_vln.pixel_map_campaign import select_cases
from evaluation.benchmarks.aerial_vln import parallel_campaign as parallel


def test_latest_map_after_annotation_and_compressed_prefix(tmp_path):
    transcript = _transcript(tmp_path)
    for i in range(1, 26):
        _add_image_checkpoint(transcript, tmp_path, i)
    for i in range(3):
        path = tmp_path / f"map_{i}.png"; Image.new("RGB", (3, 3), (i, 25, 0)).save(path)
        ref = image_media_ref(path, run_dir=tmp_path, observation_id="obs_0025", frame_id=f"map_{i}", role="map", latest_group="map")
        transcript.append_assistant(AssistantTurn(f"a{i}", "annotate", [AgentToolCall(f"c{i}", "uav.turn_relative", {"yaw_deg": 1})]))
        transcript.append_tool_result(ToolResult(f"c{i}", "uav.turn_relative", "ok", model_inputs=[ref.to_model_input()]))
    projector = HistoryProjector(assembler=RequestAssembler(MediaHydrator(base_dir=tmp_path)),
        media_policy={"max_request_images":20,"max_tool_evidence_images":4,"keep_recent_observations":3,
                      "pin_instruction_start":True,"pin_current":True})
    projected = projector.project(transcript, default_registry())
    selected = projected.audit["selected_visual_evidence"]
    assert sum(i["selection_tier"] == "latest_group" for i in selected) == 1
    assert next(i for i in selected if i["selection_tier"] == "latest_group")["frame_id"] == "map_2"
    assert projected.audit["total_image_count"] <= 20
    # A latest cumulative snapshot remains eligible even if its source event
    # is entirely in a compressed, uncited prefix.
    candidate = _candidate(ref.to_dict(), 1, evidence_ref="c", source_kind="tool_evidence")
    chosen = _select_visual_candidates([candidate], projector.media_policy, compacted_through=50,
                                      current_observation_id="obs_0025", cited_refs=frozenset())
    assert chosen[0].candidate.frame_id == "map_2"


def test_production_runtime_plugin_write_and_history_evidence(tmp_path):
    plugins = PluginRegistry.from_config("configs/plugins.pixel_map.json")
    from src.runtime.factory import RuntimeFactory
    from src.runtime.configs.stabilization_config import UavStabilizationConfig
    from tests.test_unified_tool_loop import ManagedFakeBackend, Recorder, ScriptedPlanner
    backend = ManagedFakeBackend()
    runtime = RuntimeFactory("configs/plugins.pixel_map.json").create_task_runtime(backend, Recorder(tmp_path),
        stabilization_config=UavStabilizationConfig(enabled=False), planner=ScriptedPlanner([
        [call("save", "skill.place_memory.save_place", place_id="P001", summary="Red doorway", landmarks=["door"],
              evidence_refs=["obs_0001"], photo_observation_ids=["obs_0001"])],
        [call("inspect", "skill.place_memory.inspect_place", place_id="P001")],
        [move("m")], [done()],
    ]))
    image = tmp_path / "original.png"; Image.new("RGB", (8, 8), "red").save(image)
    def observe(scene, agent, modalities):
        backend.observations += 1
        return Observation(scene, agent, modalities, {"pose": dict(backend.pose), "frames": {"rgb": {"path": str(image)}}})
    backend.get_observation = observe
    state = runtime.run_task("map", "Inspect the door and move two meters")
    assert state.completed and state.physical_window_count == 1
    events = runtime.planner.transcript.snapshot()
    results = [e.payload for e in events if e.kind == "tool_result"]
    assert [r.status for r in results] == ["ok", "ok", "ok", "ok"]
    assert results[0].model_inputs[0]["media_ref"]["latest_group"] == "task_place_map"
    assert results[0].data["summary"] == "Red doorway"
    assert results[0].data["current_place"]["is_annotated"] is True
    assert results[0].data["new_observations_since_save"] == 0
    projection = HistoryProjector(assembler=RequestAssembler(MediaHydrator(base_dir=tmp_path)),
        media_policy={"max_request_images":20,"max_tool_evidence_images":4,"keep_recent_observations":3,
                      "pin_instruction_start":True,"pin_current":True}).project(runtime.planner.transcript, runtime.tools.registry)
    selected = [s for s in projection.audit["selected_visual_evidence"] if s["selection_tier"] == "latest_group"]
    assert len(selected) == 1
    assert projection.audit["total_image_count"] <= 20
    available = environmental_evidence_refs(events, runtime.tools.registry)
    assert "inspect" not in available and "save" not in available
    assert "obs_0001" in available and "m" in available
    assert len([x for x in backend.actions if x[0] == "move_body"]) == 1


def test_sampling_is_fixed_and_outcome_independent():
    population = {"2": (3, 22, 20), "3": (24, 5, 0), "5": (0, 15, 31), "8": (22, 3, 0),
                  "10": (1, 25, 55), "12": (21, 24, 2), "14": (15, 13, 3), "17": (26, 3, 0)}
    tasks = [{"split": "val_seen", "scene_id": scene, "difficulty": difficulty, "episode_id": f"{scene}_{difficulty}_{i}"}
             for scene, numbers in population.items() for difficulty, count in zip(("low", "medium", "high"), numbers) for i in range(count)]
    selected, pilot = select_cases(tasks)
    assert len(selected) == len({t["episode_id"] for t in selected}) == 100
    assert selected == select_cases(list(reversed(tasks)))[0]
    assert [sum(t["difficulty"] == d for t in selected) for d in ("low", "medium", "high")] == [34, 33, 33]
    assert [sum(t["difficulty"] == d for t in pilot) for d in ("low", "medium", "high")] == [5, 5, 5]
    assert len({t["scene_id"] for t in pilot}) == 8


def test_disabled_detector_preload_never_imports_weights(tmp_path, monkeypatch):
    config = tmp_path / "source/configs/plugins.json"; config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"plugins": {"enabled": ["capabilities.pixel_depth"]}}), encoding="utf-8")
    monkeypatch.setattr(parallel, "cached_models", lambda *a: pytest.fail("disabled detector preloaded"))
    overlay = {}
    parallel.offline_models(tmp_path, overlay)
    assert overlay["model_cache"] == []


def test_capture_calibration_uses_installed_sdk_and_paired_geometry(tmp_path):
    from io import BytesIO
    from types import SimpleNamespace
    import numpy as np
    from evaluation.benchmarks.aerial_vln.capture_validation import capture_checks
    h = w = 16
    color = np.zeros((h, w, 3), dtype=np.uint8); color[:, :, 0] = np.arange(w)
    color[:, :, 1] = np.arange(h)[:, None]; color[:, :, 2] = 100
    buffer = BytesIO(); Image.fromarray(color).save(buffer, format="PNG")
    y, x = np.mgrid[:h, :w]
    depths = np.ones((h, w)) * 10
    ranges = depths * np.sqrt(1 + ((x - 7.5) / 8) ** 2 + ((y - 7.5) / 8) ** 2)
    def response(raw=b"", values=()):
        return SimpleNamespace(width=w, height=h, image_data_uint8=raw, image_data_float=list(np.array(values).ravel()),
            camera_position=SimpleNamespace(x_val=0, y_val=0, z_val=0), camera_name="front", time_stamp=123)
    rpc = SimpleNamespace(simIsPause=lambda: False, simPause=lambda value: None,
        simGetCameraInfo=lambda *a, **k: SimpleNamespace(fov=90),
        simGetImages=lambda *a, **k: [response(color[:, :, ::-1].tobytes()), response(buffer.getvalue()), response(values=ranges), response(values=depths)])
    report = capture_checks(rpc, "front", "uav", tmp_path)
    assert report["passed"] and report["best_raw_transform"] == "identity"
