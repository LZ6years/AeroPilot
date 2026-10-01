"""Tests for stateful STMR map-context integration without real model weights."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from plugins.research.stmr_map.context_modules.map import PluginContextModule, STMR_MAP_USAGE
from plugins.research.stmr_map.online import StmrOnlineMapBuilder, StmrOnlineMapResult
from plugins.research.stmr_map.perception import StmrPerceptionSession
from plugins.research.stmr_map.perception.grounding_dino_backend import DetectionResult
from plugins.research.stmr_map.perception.sam_backend import SamMaskResult
from src.agent.context.request_assembler import RequestAssembler
from src.memory.context.contracts import MemorySourceBundle, TaskContextInput
from src.memory.context.media_refs import image_media_ref
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MapMemoryStore, MemoryRecordReader


def test_stmr_context_declares_semantic_evidence_boundary():
    evidence = PluginContextModule(mode="planner").describe_output()["semantic_evidence"]

    assert len(evidence) == 1
    assert "semantic history" in evidence[0]
    assert "not verified free space or exact target pose" in evidence[0]


def test_online_builder_processes_completed_observation_once(tmp_path):
    _write_complete_memory(tmp_path, include_pose=True)
    grounding = _FakeGrounding()
    sam = _FakeSam()
    builder = StmrOnlineMapBuilder(
        session=StmrPerceptionSession(grounding_backend=grounding, sam_backend=sam),
        device="cpu",
    )
    memory = _memory(tmp_path)
    stage = _stage_context()

    first = builder.build(memory, stage)
    second = builder.build(memory, stage)

    assert first.status == second.status == "ready"
    assert first.processed_observation_count == second.processed_observation_count == 1
    assert first.point_count == second.point_count
    assert grounding.calls == sam.calls == 1
    assert first.png_path and first.png_path.exists()
    assert first.json_path and first.json_path.exists()
    manifest_path = first.json_path.parent / "render_history.json"
    history = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert history["snapshot_count"] == 1
    assert history["snapshots"][0]["observation_id"] == "obs_0001"
    assert Path(history["snapshots"][0]["png_path"]).exists()
    debug = builder.debug_summary()
    frame_total = debug["profiling"]["stages"]["stmr_online_frame_total"]
    assert frame_total["sample_count"] == 1
    assert debug["profiling"]["stages"]["stmr_online_perception_map_total"]["sample_count"] == 1
    assert debug["profiling"]["stages"]["stmr_online_png_render"]["sample_count"] == 1
    assert debug["render_history"]["snapshot_count"] == 1
    assert debug["batch_incremental_parity"]["equivalent"] is True


def test_online_builder_preserves_one_render_for_each_successful_observation(tmp_path):
    _write_complete_memory(tmp_path, include_pose=True, observation_count=2)
    grounding = _FakeGrounding()
    sam = _FakeSam()
    builder = StmrOnlineMapBuilder(
        session=StmrPerceptionSession(grounding_backend=grounding, sam_backend=sam),
        device="cpu",
    )

    result = builder.build(_memory(tmp_path), _stage_context())
    repeated = builder.build(_memory(tmp_path), _stage_context())
    manifest = json.loads((result.json_path.parent / "render_history.json").read_text(encoding="utf-8"))

    assert result.processed_observation_count == repeated.processed_observation_count == 2
    assert grounding.calls == sam.calls == 2
    assert manifest["snapshot_count"] == 2
    assert [item["observation_id"] for item in manifest["snapshots"]] == ["obs_0001", "obs_0002"]
    assert [item["processed_observation_count"] for item in manifest["snapshots"]] == [1, 2]
    assert len(list((result.json_path.parent / "snapshots").glob("*.png"))) == 2


def test_online_builder_keeps_latest_map_when_history_copy_fails(tmp_path, monkeypatch):
    _write_complete_memory(tmp_path, include_pose=True)
    builder = StmrOnlineMapBuilder(
        session=StmrPerceptionSession(grounding_backend=_FakeGrounding(), sam_backend=_FakeSam()),
        device="cpu",
    )

    def fail_copy(*args, **kwargs):
        raise OSError("synthetic history storage failure")

    monkeypatch.setattr("plugins.research.stmr_map.online.shutil.copy2", fail_copy)
    result = builder.build(_memory(tmp_path), _stage_context())
    debug = builder.debug_summary()

    assert result.status == "ready"
    assert result.processed_observation_count == 1
    assert result.png_path and result.png_path.exists()
    assert debug["render_history"]["snapshot_count"] == 0
    assert any("stmr_render_history_write_failed:obs_0001" in item for item in debug["warnings"])


def test_online_builder_waits_for_pose_then_retries(tmp_path):
    _write_complete_memory(tmp_path, include_pose=False)
    grounding = _FakeGrounding()
    sam = _FakeSam()
    builder = StmrOnlineMapBuilder(
        session=StmrPerceptionSession(grounding_backend=grounding, sam_backend=sam),
        device="cpu",
    )

    pending = builder.build(_memory(tmp_path), _stage_context())
    _write_pose(tmp_path)
    recovered = builder.build(_memory(tmp_path), _stage_context())
    repeated = builder.build(_memory(tmp_path), _stage_context())

    assert pending.status == "waiting_for_complete_observation"
    assert grounding.calls == sam.calls == 1
    assert recovered.status == repeated.status == "ready"
    assert recovered.processed_observation_count == repeated.processed_observation_count == 1


def test_online_builder_reads_aerial_vln_nested_pose_and_artifact_paths(tmp_path):
    _write_complete_memory(tmp_path, include_pose=False)
    memory_dir = tmp_path / "memory"
    pose = {
        "id": "pose_0001",
        "observation_id": "obs_0001",
        "coordinate_frame": "canonical_unspecified",
        "pose": {
            "position": {"x": 48.25, "y": -7.0, "z": -9.45},
            "euler_rad": {"pitch": 0.0, "roll": 0.0, "yaw": math.pi / 2.0},
        },
    }
    (memory_dir / "poses.jsonl").write_text(json.dumps(pose) + "\n", encoding="utf-8")
    frame_rows = [json.loads(line) for line in (memory_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in frame_rows:
        row["metadata"] = {**dict(row.get("metadata") or {}), "artifact_path": row.pop("path")}
        row["path"] = None
    (memory_dir / "frames.jsonl").write_text(
        "\n".join(json.dumps(item) for item in frame_rows) + "\n",
        encoding="utf-8",
    )
    builder = StmrOnlineMapBuilder(
        session=StmrPerceptionSession(
            grounding_backend=_FakeGrounding(), sam_backend=_FakeSam()
        ),
        device="cpu",
    )

    result = builder.build(_memory(tmp_path), _stage_context())

    assert result.status == "ready"
    assert result.center_pose == {
        "x": 48.25,
        "y": -7.0,
        "z": -9.45,
        "yaw_deg": 90.0,
    }
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["stmr_map"]["coordinate_frame"]["world_frame"] == "canonical_unspecified"


def test_online_builder_rejects_unknown_or_mixed_pose_frames(tmp_path):
    _write_complete_memory(tmp_path, include_pose=True, observation_count=2)
    pose_path = tmp_path / "memory" / "poses.jsonl"
    rows = [json.loads(line) for line in pose_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["coordinate_frame"] = "unrealcv_world_m"
    rows[1]["coordinate_frame"] = "canonical_unspecified"
    pose_path.write_text(
        "\n".join(json.dumps(item) for item in rows) + "\n",
        encoding="utf-8",
    )
    builder = StmrOnlineMapBuilder(
        session=StmrPerceptionSession(
            grounding_backend=_FakeGrounding(), sam_backend=_FakeSam()
        ),
        device="cpu",
    )

    with pytest.raises(ValueError, match="mixed pose coordinate frames"):
        builder.build(_memory(tmp_path), _stage_context())


def test_online_builder_persists_failure_summary_without_snapshot(tmp_path):
    class FailingGrounding(_FakeGrounding):
        def detect(self, rgb_path, prompts, *, profiler=None):
            self.calls += 1
            raise RuntimeError("synthetic detector failure")

    _write_complete_memory(tmp_path, include_pose=True)
    builder = StmrOnlineMapBuilder(
        session=StmrPerceptionSession(grounding_backend=FailingGrounding(), sam_backend=_FakeSam()),
        device="cpu",
    )

    result = builder.build(_memory(tmp_path), _stage_context())
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert result.status == "failed_no_snapshot"
    assert result.png_path is None
    assert payload["debug"]["failures"][0]["error_type"] == "RuntimeError"
    assert payload["debug"]["failures"][0]["error"] == "synthetic detector failure"


def test_map_context_shadow_is_baseline_and_planner_adds_one_image(tmp_path):
    png = tmp_path / "stmr.png"
    Image.new("RGB", (4, 4), (20, 40, 60)).save(png)
    result = StmrOnlineMapResult(
        status="ready",
        stale=False,
        latest_observation_id="obs_0001",
        processed_observation_count=1,
        point_count=16,
        voxel_count=3,
        center_pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
        map_config={"grid_size": 20, "meters_per_cell": 5.0, "orientation": "uav_forward_up"},
        semantic_cell_counts={"building": 2},
        trajectory_summary={
            "pose_count": 1,
            "recent_motion_local_m": {"forward_delta_m": 0.0, "right_delta_m": 0.0, "distance_m": 0.0},
            "translation_status": "stationary_or_yaw_only",
        },
        png_path=png,
        json_path=None,
    )
    fake = _FakeOnlineBuilder(result)
    sources = MemorySourceBundle(
        FlightMemory(
            "run",
            "map",
            "scene",
            "uav",
            memory_reader=MemoryRecordReader(tmp_path),
        ),
        MapMemoryStore(),
    )

    shadow = PluginContextModule(mode="shadow", online_builder=fake).build(sources, _stage_context())
    planner = PluginContextModule(mode="planner", online_builder=fake).build(sources, _stage_context())

    assert shadow.fragment.media_refs == ()
    assert "stmr" not in shadow.fragment.content
    assert len(planner.fragment.media_refs) == 1
    ref = planner.fragment.media_refs[0]["media_ref"]
    assert ref["usage"] == STMR_MAP_USAGE
    assert "START/ENTRY to NOW" in ref["usage"]
    assert "image top is current forward" in ref["usage"]
    assert planner.fragment.content["stmr"]["representation"] == "semantic_history_not_traversability"
    assert planner.fragment.content["stmr"]["trajectory_summary"]["pose_count"] == 1
    assert "trajectory_trace" not in planner.fragment.content["stmr"]


def test_map_context_failure_falls_back_without_model_input():
    sources = MemorySourceBundle(FlightMemory("run", "map", "scene", "uav"), MapMemoryStore())

    planner = PluginContextModule(mode="planner", online_builder=_ExplodingOnlineBuilder()).build(
        sources,
        _stage_context(),
    )
    shadow = PluginContextModule(mode="shadow", online_builder=_ExplodingOnlineBuilder()).build(
        sources,
        _stage_context(),
    )

    assert planner.fragment.media_refs == ()
    assert "stmr_map_failed:RuntimeError" in planner.audit.warnings
    assert shadow.fragment.media_refs == ()
    assert shadow.audit.warnings == ()


def test_stmr_planner_summary_stays_bounded_as_history_grows(tmp_path):
    png = tmp_path / "stmr.png"
    Image.new("RGB", (4, 4), (20, 40, 60)).save(png)
    sources = MemorySourceBundle(
        FlightMemory(
            "run",
            "map",
            "scene",
            "uav",
            memory_reader=MemoryRecordReader(tmp_path),
        ),
        MapMemoryStore(),
    )

    def fragment_for(observation_count: int):
        result = StmrOnlineMapResult(
            status="ready",
            stale=False,
            latest_observation_id=f"obs_{observation_count:04d}",
            processed_observation_count=observation_count,
            point_count=observation_count * 100,
            voxel_count=observation_count * 10,
            center_pose={
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "yaw_deg": 45.0,
            },
            map_config={
                "grid_size": 20,
                "meters_per_cell": 5.0,
                "orientation": "uav_forward_up",
            },
            semantic_cell_counts={"building": 12},
            trajectory_summary={
                "pose_count": observation_count,
                "total_translation_m": 20.0,
                "recent_motion_local_m": {
                    "forward_delta_m": 1.0,
                    "right_delta_m": 0.0,
                    "distance_m": 1.0,
                },
            },
            png_path=png,
            json_path=None,
        )
        return PluginContextModule(
            mode="planner",
            online_builder=_FakeOnlineBuilder(result),
        ).build(sources, _stage_context())

    short = fragment_for(1)
    long = fragment_for(1000)
    short_text = json.dumps(short.fragment.content, sort_keys=True)
    long_text = json.dumps(long.fragment.content, sort_keys=True)

    assert len(long.fragment.media_refs) == len(short.fragment.media_refs) == 1
    assert "trajectory_trace" not in long_text
    assert "frame_history" not in long_text
    assert len(long_text) - len(short_text) < 32


def test_stmr_media_follows_rgb_and_precedes_context_json(tmp_path):
    rgb_path = _tiny_png(tmp_path / "rgb.png", (10, 20, 30))
    map_path = _tiny_png(tmp_path / "map.png", (30, 20, 10))
    rgb_ref = image_media_ref(
        rgb_path, observation_id="obs", frame_id="rgb", role="current_observation",
        usage="NOW | CURRENT OBSERVATION.",
    )
    map_ref = image_media_ref(
        map_path, observation_id="obs", frame_id="map", role="stmr_semantic_map",
        usage=STMR_MAP_USAGE,
    )
    checkpoint = {
        "schema_version": "state_checkpoint_v3",
        "checkpoint_id": "checkpoint_0001",
        "reason": "instruction_start",
        "observation_id": "obs",
        "fragments": [
            {
                "name": "image",
                "content": {},
                "media_refs": [rgb_ref.to_model_input()],
            },
            {
                "name": "map",
                "content": {"stmr": {"representation": "semantic_history_not_traversability"}},
                "media_refs": [map_ref.to_model_input()],
            },
        ],
    }

    content = RequestAssembler().checkpoint_content(
        checkpoint,
        model_inputs=[rgb_ref.to_model_input(), map_ref.to_model_input()],
        timeline_role="CURRENT",
    )

    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text", "image_url", "text", "image_url", "text"]
    assert "NOW | CURRENT OBSERVATION." in content[0]["text"]
    assert STMR_MAP_USAGE in content[2]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[3]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "data:image" not in content[-1]["text"]
    assert json.loads(content[-1]["text"])["fragments"][-1]["name"] == "map"


class _FakeGrounding:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, rgb_path, prompts, *, profiler=None):
        self.calls += 1
        return [DetectionResult((1, 1, 7, 7), "building", 0.95, "building")]

    def warmup(self, *, profiler=None):
        return None


class _FakeSam:
    def __init__(self) -> None:
        self.calls = 0

    def segment(self, rgb_path, detections, *, profiler=None):
        self.calls += 1
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True
        return [SamMaskResult(detections[0], mask, 0.95)]

    def warmup(self, *, profiler=None):
        return None


class _FakeOnlineBuilder:
    def __init__(self, result: StmrOnlineMapResult) -> None:
        self.result = result
        self.calls = 0

    def build(self, memory, stage_context):
        self.calls += 1
        return self.result

    def debug_summary(self):
        return self.result.context_content()


class _ExplodingOnlineBuilder:
    def build(self, memory, stage_context):
        raise RuntimeError("synthetic STMR failure")


def _write_complete_memory(run_dir: Path, *, include_pose: bool, observation_count: int = 1) -> None:
    memory_dir = run_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(1, observation_count + 1):
        observation_id = f"obs_{index:04d}"
        rgb = run_dir / f"rgb_{index:04d}.png"
        depth = run_dir / f"depth_{index:04d}.npy"
        Image.new("RGB", (8, 8), (10 * index, 20, 30)).save(rgb)
        np.save(depth, np.full((8, 8), 8.0 + index, dtype=np.float32))
        frames.extend(
            [
                {
                    "id": f"rgb_{index:04d}",
                    "observation_id": observation_id,
                    "modality": "rgb",
                    "path": str(rgb),
                },
                {
                    "id": f"depth_{index:04d}",
                    "observation_id": observation_id,
                    "modality": "depth",
                    "path": str(depth),
                    "metadata": {"fov_degrees": 90.0},
                },
            ]
        )
    (memory_dir / "frames.jsonl").write_text("\n".join(json.dumps(item) for item in frames) + "\n", encoding="utf-8")
    (memory_dir / "actions.jsonl").write_text("", encoding="utf-8")
    (memory_dir / "observations.jsonl").write_text("", encoding="utf-8")
    if include_pose:
        _write_pose(run_dir, observation_count=observation_count)
    else:
        (memory_dir / "poses.jsonl").write_text("", encoding="utf-8")


def _write_pose(run_dir: Path, *, observation_count: int = 1) -> None:
    path = run_dir / "memory" / "poses.jsonl"
    poses = [
        {
            "id": f"pose_{index:04d}",
            "observation_id": f"obs_{index:04d}",
            "pose": {"x": float(index - 1), "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
        }
        for index in range(1, observation_count + 1)
    ]
    path.write_text("\n".join(json.dumps(item) for item in poses) + "\n", encoding="utf-8")


def _memory(run_dir: Path) -> FlightMemory:
    return FlightMemory(
        "run",
        "map",
        "scene",
        "uav",
        active_instruction="follow the building",
        memory_reader=MemoryRecordReader(run_dir),
    )


def _stage_context() -> TaskContextInput:
    return TaskContextInput(
        active_instruction="follow the building",
    )


def _tiny_png(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (2, 2), color).save(path)
    return path
