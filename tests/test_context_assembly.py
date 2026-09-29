"""Ownership, control-prefix and frame traversal equivalence, without engines."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from src.agent.context.projection import HistoryProjector
from src.agent.transcript import AgentTranscript
from src.memory.context.modules import image as images
from src.memory.context.modules.image import frames as image_frames
from src.memory.context.modules.image import module as image_module
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    MemorySourceBundle,
    StateCheckpoint,
    TaskContextInput,
    checkpoint_fingerprint,
)
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.tools.registry import default_registry
from tests.support.runtime import call, done, move, public_runtime


def make_memory(tmp_path, observations):
    writer = MemoryEventWriter(tmp_path)
    memory = FlightMemory(
        "run", "map", "scene", "uav", memory_reader=MemoryRecordReader(tmp_path)
    )
    memory.update_instruction_ref(writer.write_instruction("Inspect"), "Inspect")
    for observation in observations:
        refs = writer.write_observation(observation)
        memory.append_observation_ref(refs["observation_ref"])
        memory.append_pose_ref(refs["pose_ref"])
        memory.append_frame_refs(refs["frame_refs"])
    return memory


def transcript():
    result = AgentTranscript(SimpleNamespace(write_transcript_event=lambda **kw: None))
    result.start_instruction(
        "Inspect",
        "instruction",
        InstructionEnvelope("instruction", "", {}, {}, (), ()),
    )
    fragment = ContextFragment("test", {"nested": {"list": [1, 2]}})
    result.append_state(
        StateCheckpoint(
            "checkpoint_0001",
            "instruction_start",
            "obs_1",
            None,
            (),
            (fragment,),
            checkpoint_fingerprint("obs_1", None, (), (fragment,)),
        )
    )
    return result


@pytest.mark.parametrize("boundary", [True, False, -1, 4, 1.5, "1", None])
def test_prefix_rejects_invalid_boundary(boundary):
    with pytest.raises(ValueError):
        transcript().snapshot_before(boundary)


def test_prefix_isolated_and_excludes_subsequent_events():
    history = transcript()
    prefix = history.snapshot_before(3)
    assert history.snapshot_before(0) == ()
    assert prefix == history.snapshot()
    history.append_protocol_feedback("later")
    assert history.snapshot_before(3) == prefix
    prefix[2].payload.fragments[0].model_content["nested"]["list"].append(9)
    assert history.items[2].payload.fragments[0].model_content["nested"]["list"] == [1, 2]


def test_projection_public_inputs_and_outputs_remain_isolated():
    history = transcript()
    original = copy.deepcopy(history.items)
    external = history.snapshot()
    projector = HistoryProjector()
    registry = default_registry()
    normal = projector.project(history, registry)
    explicit = projector.project_snapshot(external, registry)
    assert normal == explicit
    normal.messages[0]["content"] = "changed"
    normal.messages[1]["content"][0]["text"] = "changed"
    normal.audit["policies"]["changed"] = []
    assert history.items == original
    assert external == tuple(original)
    assert projector.project_snapshot(external, registry) == explicit


def test_normal_projection_copies_entire_history_once(monkeypatch):
    history = transcript()
    original = copy.deepcopy
    roots = []
    def counted(value, *args, **kwargs):
        if value is history.items or (isinstance(value, tuple) and value and
                                     type(value[0]).__name__ == "TranscriptEvent"):
            roots.append(type(value))
        return original(value, *args, **kwargs)
    monkeypatch.setattr(copy, "deepcopy", counted)
    HistoryProjector().project(history, default_registry())
    assert roots == [list]
    roots.clear()
    HistoryProjector().project_snapshot(tuple(history.items), default_registry())
    assert roots == [tuple]


@pytest.mark.parametrize("terminal", ["completed", "blocked"])
def test_runtime_only_copies_validated_control_prefix(tmp_path, monkeypatch, terminal):
    import src.runtime.task.turn_executor as module
    batches = [
        [move("mixed"), call("mixed_plan", "runtime.update_plan", steps=[])],
        [call("read", "knowledge.load", name="missing")],
        [call("plan", "runtime.update_plan", explanation="Inspect", steps=[
            {"id": "inspect", "description": "Inspect", "status": "in_progress"}])],
        [move("move")],
        [call("terminal", f"runtime.report_{terminal}", reason="Observed", evidence_refs=["obs_0002"],
              **({"blocking_condition": "insufficient_evidence"} if terminal == "blocked" else {}))],
    ]
    runtime, backend = public_runtime(tmp_path, batches)
    original_snapshot = AgentTranscript.snapshot
    # ScriptedPlanner explicitly asks for one snapshot per decision. No other
    # full snapshots may be added for control execution or rejected/read turns.
    snapshots, prefixes, controls = [], [], []
    def snapshot(self):
        snapshots.append(len(self.items))
        return original_snapshot(self)
    original_prefix = AgentTranscript.snapshot_before
    def before(self, boundary):
        result = original_prefix(self, boundary)
        prefixes.append((boundary, len(self.items), result))
        return result
    original_control = module.execute_builtin_control
    def control(tool_call, state, registry, history):
        assert history == tuple(runtime.agent_transcript.items[:-1])
        assert runtime.agent_transcript.items[-1].kind == "assistant_turn"
        controls.append(tool_call.tool_call_id)
        return original_control(tool_call, state, registry, history)
    monkeypatch.setattr(AgentTranscript, "snapshot", snapshot)
    monkeypatch.setattr(AgentTranscript, "snapshot_before", before)
    monkeypatch.setattr(module, "execute_builtin_control", control)
    runtime.run_task("map", "Inspect the endpoint")
    assert controls == ["plan", "terminal"]
    assert len(snapshots) == 5 and len(prefixes) == 2
    assert all(boundary == length - 1 for boundary, length, _ in prefixes)
    results = [e.payload.tool_call_id for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert results == ["mixed", "mixed_plan", "read", "plan", "move", "terminal"]
    assert backend.stops == ["scene"]


@pytest.mark.parametrize("count", [1, 3, 50])
@pytest.mark.parametrize("depth", [False, True])
def test_image_fragment_equal_and_overlapping_batches_read_once(tmp_path, monkeypatch, count, depth):
    image = tmp_path / "frame.png"
    Image.new("RGB", (24, 16), (50, 80, 130)).save(image)
    observations = [{"source": "after", "modalities": ["rgb"], "frames": {
        "rgb": {"path": str(image), "width": 24, "height": 16, "mock": False},
        "depth": {"mock": True}}} for _ in range(count)]
    # Latest observation has no RGB: V2 must explicitly avoid historical
    # fallback while preserving one-build record-cache behavior.
    if count > 1:
        observations[-1]["frames"] = {"depth": {"mock": True}}
    memory = make_memory(tmp_path, observations)
    sources, stage = MemorySourceBundle(memory), TaskContextInput("Inspect")
    module = images.DefaultImageContextModule(include_current_depth=depth)
    original = memory.memory_reader.read_refs
    reads = []
    def counted(refs):
        reads.append(id(refs))
        return original(refs)
    monkeypatch.setattr(memory.memory_reader, "read_refs", counted)
    with memory.memory_reader.read_scope():
        candidate = module.build(sources, stage)
    assert len(reads) == 1 + min(count, module.max_items)
    reads.clear()
    with patch.object(image_module, "_BuildFrameReader", lambda reader: reader):
        with memory.memory_reader.read_scope():
            baseline = module.build(sources, stage)
    assert len(reads) > 1 + min(count, module.max_items)
    assert baseline == candidate
    assert module.build(sources, stage) == candidate


def test_frame_batch_missing_is_retried_errors_propagate_and_output_annotations_isolated():
    refs = [{"ref": "frames#1"}, {"ref": "frames#2"}]
    records = [{"modality": "rgb", "id": "rgb", "observation_id": "obs", "metadata": {"mock": True}},
               {"modality": "depth", "id": "depth", "observation_id": "obs"}]
    class Reader:
        attempts = 0
        def read_refs(self, value):
            self.attempts += 1
            if self.attempts == 1:
                return records[:1]
            if self.attempts == 2:
                raise json.JSONDecodeError("bad", "{", 0)
            return copy.deepcopy(records)
    source = Reader()
    reader = image_frames._BuildFrameReader(source)
    assert len(reader.read_refs(refs)) == 1
    with pytest.raises(json.JSONDecodeError):
        reader.read_refs(refs)
    observation = {"frame_refs": refs}
    recent = image_frames._recent_rgb_frames(reader, [observation], 3)
    selected = image_frames.selected_rgb_frames_from_observations(reader, [observation])
    depth = image_frames._depth_frame_for_current_rgb(reader, [observation], selected[0])
    assert source.attempts == 3
    assert "selection_role" not in recent[0]
    assert depth["selection_role"] == "current_depth"
    assert reader.read_refs(refs) == records


def test_frame_cache_preserves_custom_batch_reader_and_multi_frame_order():
    refs = [{"ref": "x"}, {"ref": "y"}]
    calls = []
    def read(refs):
        calls.append(refs)
        return [{"modality": "rgb", "id": name, "metadata": {"mock": True}} for name in ("a", "b")]
    reader = image_frames._BuildFrameReader(SimpleNamespace(read_refs=read))
    observation = {"frame_refs": refs}
    assert [r["id"] for r in image_frames._recent_rgb_frames(reader, [observation], 3)] == ["a", "b"]
    assert image_frames.selected_rgb_frames_from_observations(reader, [observation])[0]["id"] == "b"
    assert calls == [refs]
