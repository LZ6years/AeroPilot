from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from src.agent.context.media import MediaHydrationError, MediaHydrator
from src.agent.context.projection import HistoryProjector
from src.agent.context.request_assembler import RequestAssembler
from src.agent.transcript import AgentTranscript
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    MediaRef,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.context.builder import ModelContextBuilder
from src.memory.context.documents import ContextDocuments
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.media_refs import image_media_ref, run_relative_media_ref
from src.memory.stores import MapMemoryStore, MemoryEventWriter
from src.runtime.context.checkpoints import (
    ContextCheckpointService,
    UnsupportedRunSchema,
    load_state_checkpoint,
)
from src.tools.registry import ToolRegistry, default_registry
from tests.support.context import _memory, _record_observation, _state


class Recorder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.writes: list[str] = []

    def write_json(self, relative: str, value: dict) -> None:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        self.writes.append(relative)


class CountingBuilder:
    def __init__(self) -> None:
        self.build_count = 0
        self.dynamic = 0

    def build_instruction_envelope(self, state, memory) -> InstructionEnvelope:
        return InstructionEnvelope(
            instruction_id=memory.active_instruction_id,
            protocol_context="protocol-once",
            identity={"name": "FlyAgent"},
            user_preferences={"language": "zh"},
            knowledge_catalog=({"name": "flight"},),
            static_module_guidance=({"name": "pose"},),
        )

    def build_checkpoint(self, state, memory, *, checkpoint_id, reason):
        self.build_count += 1
        fragment = ContextFragment("pose", {"sequence": self.dynamic})
        fingerprint = checkpoint_fingerprint(None, None, (), (fragment,))
        return (
            StateCheckpoint(
                checkpoint_id=checkpoint_id,
                reason=reason,
                observation_id=None,
                task_plan=None,
                loaded_knowledge=(),
                fragments=(fragment,),
                fingerprint=fingerprint,
            ),
            (),
        )


class State:
    user_instruction = "inspect"


class Memory:
    active_instruction_id = "inst_1"


def _checkpoint(index: int) -> StateCheckpoint:
    fragment = ContextFragment("pose", {"index": index})
    return StateCheckpoint(
        checkpoint_id=f"checkpoint_{index:04d}",
        reason="instruction_start" if index == 1 else "physical_window_completed",
        observation_id=f"obs_{index}",
        task_plan=None,
        loaded_knowledge=(),
        fragments=(fragment,),
        fingerprint=checkpoint_fingerprint(f"obs_{index}", None, (), (fragment,)),
    )


def test_checkpoint_service_builds_once_per_publish_and_deduplicates(tmp_path):
    recorder = Recorder(tmp_path)
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    builder = CountingBuilder()
    service = ContextCheckpointService(builder, transcript, recorder)
    service.begin_instruction(State(), Memory())

    first = service.publish(State(), Memory(), "instruction_start")
    duplicate = service.publish(State(), Memory(), "plan_updated")
    builder.dynamic = 1
    second = service.publish(State(), Memory(), "plan_updated")

    assert first is not None and second is not None
    assert duplicate is None
    assert builder.build_count == 3
    assert [item.kind for item in transcript.items] == [
        "instruction_envelope",
        "user_instruction",
        "state_checkpoint",
        "state_checkpoint",
    ]
    assert recorder.writes.count("contexts/latest_checkpoint.json") == 2


def test_projection_emits_static_instruction_context_once(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    envelope = InstructionEnvelope(
        instruction_id="inst_1",
        protocol_context="protocol-once",
        identity={"name": "FlyAgent"},
        user_preferences={"language": "zh"},
        knowledge_catalog=({"name": "flight"},),
        static_module_guidance=({"name": "pose"},),
    )
    transcript.start_instruction("inspect", "inst_1", envelope)
    transcript.append_state(_checkpoint(1))
    transcript.append_state(_checkpoint(2))

    projection = HistoryProjector().project(transcript, ToolRegistry())
    serialized = json.dumps(projection.messages, ensure_ascii=False)

    assert serialized.count("inspect") == 1
    assert serialized.count("protocol-once") == 1
    assert serialized.count("knowledge_catalog") == 1
    assert serialized.count("static_module_guidance") == 1


def test_media_hydration_is_lazy_and_sha_cached(tmp_path):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    path = tmp_path / "frame.png"
    path.write_bytes(raw)
    ref = MediaRef(
        ref="frame.png",
        sha256=hashlib.sha256(raw).hexdigest(),
        mime_type="image/png",
        observation_id="obs_1",
        frame_id="rgb_1",
        role="current_rgb",
    )
    hydrator = MediaHydrator(base_dir=tmp_path)

    assert hydrator.stats()["reads"] == 0
    prepared = hydrator.hydrate(ref)
    assert prepared.data_url.startswith("data:image/png;base64,")
    assert hydrator.hydrate(ref).provider_sha256
    assert hydrator.stats() == {
        "entries": 1,
        "payload_bytes": len(prepared.data_url),
        "reads": 1,
        "encodes": 1,
        "hits": 1,
        "evictions": 0,
    }


def test_runtime_media_refs_are_run_relative_and_contained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = Path("runs") / "portable"
    image = run_dir / "artifacts" / "scene" / "rgb" / "frame.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (3, 2), (10, 20, 30)).save(image)

    assert run_relative_media_ref(image, run_dir) == "artifacts/scene/rgb/frame.png"
    assert run_relative_media_ref(image.resolve(), run_dir) == "artifacts/scene/rgb/frame.png"
    durable = image_media_ref(
        image,
        run_dir=run_dir,
        observation_id="obs_1",
        frame_id="rgb_1",
        role="current_observation",
    )
    assert durable.ref == "artifacts/scene/rgb/frame.png"
    assert MediaHydrator(base_dir=run_dir).hydrate(durable).data_url.startswith(
        "data:image/png;base64,"
    )


@pytest.mark.parametrize(
    "ref_value",
    ["../outside.png", "artifacts\\outside.png"],
)
def test_runtime_media_hydrator_rejects_nonportable_refs(tmp_path, ref_value):
    digest = hashlib.sha256(b"outside").hexdigest()
    ref = MediaRef(
        ref=ref_value,
        sha256=digest,
        mime_type="image/png",
        observation_id="obs_1",
        frame_id="rgb_1",
        role="current_observation",
    )

    with pytest.raises(MediaHydrationError):
        MediaHydrator(base_dir=tmp_path).hydrate(ref)


def test_runtime_media_hydrator_rejects_absolute_ref(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (2, 2)).save(image)
    ref = image_media_ref(
        image,
        observation_id="obs_1",
        frame_id="rgb_1",
        role="current_observation",
    )

    with pytest.raises(MediaHydrationError, match="run-relative"):
        MediaHydrator(base_dir=tmp_path).hydrate(ref)


def test_v2_artifacts_are_binary_free_and_v1_is_rejected(tmp_path):
    checkpoint = _checkpoint(1)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    assert "data:image/" not in checkpoint_path.read_text(encoding="utf-8")
    assert load_state_checkpoint(checkpoint_path)["checkpoint_id"] == "checkpoint_0001"

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"schema_version": "context_v1"}), encoding="utf-8")
    with pytest.raises(UnsupportedRunSchema):
        load_state_checkpoint(legacy)


def test_real_runtime_interactive_single_window_builds_three_checkpoints(tmp_path):
    from src.runtime.runners.interactive_session import InteractiveTaskSession
    from tests.support.runtime import done, move, public_runtime

    runtime, _ = public_runtime(
        tmp_path,
        [[move("move_once")], [done(observation="obs_0002")]],
    )
    original = runtime.context_builder.build_checkpoint
    calls: list[str] = []

    def counted(*args, **kwargs):
        calls.append(str(kwargs["reason"]))
        return original(*args, **kwargs)

    runtime.context_builder.build_checkpoint = counted
    session = InteractiveTaskSession(runtime, "map")
    session.start_scene()
    summary = session.submit_instruction("Move once, then stop.")

    assert summary["completed"] is True
    assert calls == [
        "instruction_start",
        "physical_window_completed",
        "task_terminal",
    ]


def test_runtime_factory_hydrates_workspace_relative_capture_path(
    tmp_path, monkeypatch
):
    """Cover the production path shape that previously duplicated run_dir."""

    from src.env_client import ActionResult, Observation, SceneInstance
    from src.plugins import PluginRegistry
    from src.runtime.configs.stabilization_config import UavStabilizationConfig
    from src.runtime.factory import RuntimeFactory
    from src.runtime.logging.async_run_logger import AsyncRunLogger
    from tests.support.planner import StubPlanner, response_with_calls

    monkeypatch.chdir(tmp_path)
    run_dir = Path("runs") / "relative_media"

    class ImageBackend:
        def __init__(self) -> None:
            self.scene = SceneInstance(
                "scene", "map", "default", "fake", "", 0, None, None, "running"
            )
            self.observation_count = 0

        def start_scene(self, map_id, scene_id=None):
            return self.scene

        def get_or_start_scene(self, map_id, scene_id=None):
            return self.scene

        def stop_scene(self, scene_id):
            return None

        def execute_action(self, scene_id, agent_id, action, args):
            return ActionResult(True, action, "ok")

        def get_observation(self, scene_id, agent_id, modalities):
            self.observation_count += 1
            image = (
                run_dir
                / "artifacts"
                / "scene"
                / "rgb"
                / f"frame_{self.observation_count:06d}.png"
            )
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), (10, 20, 30)).save(image)
            return Observation(
                scene_id,
                agent_id,
                list(modalities),
                {
                    "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
                    "frames": {
                        "rgb": {
                            "artifact_path": str(image),
                            "width": 8,
                            "height": 6,
                        }
                    },
                },
            )

    planner = StubPlanner(
        response_with_calls(
            (
                "done",
                "runtime_report_completed",
                {"reason": "image received", "evidence_refs": ["obs_0001"]},
            )
        )
    )
    recorder = AsyncRunLogger(run_dir)
    runtime = RuntimeFactory(plugin_registry=PluginRegistry()).create_task_runtime(
        ImageBackend(),
        recorder,
        planner=planner,
        shutdown_policy="none",
        stabilization_config=UavStabilizationConfig(enabled=False),
    )

    state = runtime.run_task("map", "Inspect the current RGB image.")
    recorder.close()

    projection = planner.last_exchange["history_projection"]
    assert state.completed is True
    assert projection["observation_image_count"] == 1
    assert projection["missing_images"] == []
    assert "data:image/png;base64," in json.dumps(planner.payloads[0])
    checkpoint = json.loads(
        (run_dir / "debug/context_checkpoints/checkpoint_0001.json").read_text(
            encoding="utf-8"
        )
    )
    image_fragment = next(
        fragment
        for fragment in checkpoint["checkpoint"]["fragments"]
        if fragment["name"] == "image"
    )
    ref = image_fragment["model_inputs"][0]["media_ref"]["ref"]
    assert ref == "artifacts/scene/rgb/frame_000001.png"


def test_checkpoint_build_defers_media_read_until_projection(
    tmp_path, monkeypatch
):
    image = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), (20, 80, 130)).save(image)
    state = _state()
    memory, writer = _memory(tmp_path, state)
    _record_observation(writer, memory, state, frame_path=image)
    builder = ModelContextBuilder(
        registry=ContextModuleRegistry.default(),
        documents=ContextDocuments(identity={}, user_preferences={}),
        map_memory=MapMemoryStore(),
    )
    original_read_bytes = type(image).read_bytes
    media_reads = []

    def tracked(path):
        if path == image:
            media_reads.append(str(path))
        return original_read_bytes(path)

    monkeypatch.setattr(type(image), "read_bytes", tracked)
    envelope = builder.build_instruction_envelope(state, memory)
    checkpoint, _ = builder.build_checkpoint(
        state,
        memory,
        checkpoint_id="checkpoint_0001",
        reason="instruction_start",
    )
    assert media_reads == []

    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction(
        state.user_instruction, memory.active_instruction_id, envelope
    )
    transcript.append_state(checkpoint)
    hydrator = MediaHydrator(base_dir=tmp_path)
    projector = HistoryProjector(
        assembler=RequestAssembler(hydrator)
    )
    projector.project(transcript, default_registry())
    projector.project(transcript, default_registry())

    assert media_reads == [str(image)]
    assert hydrator.stats()["reads"] == 1
    assert hydrator.stats()["encodes"] == 1
    assert hydrator.stats()["hits"] == 1


def test_checkpoint_fingerprint_is_stable_for_unchanged_model_state(tmp_path):
    state = _state()
    memory, writer = _memory(tmp_path, state)
    _record_observation(writer, memory, state)
    builder = ModelContextBuilder(
        registry=ContextModuleRegistry.default(),
        documents=ContextDocuments(identity={}, user_preferences={}),
        map_memory=MapMemoryStore(),
    )

    first, _ = builder.build_checkpoint(
        state,
        memory,
        checkpoint_id="checkpoint_0001",
        reason="instruction_start",
    )
    second, _ = builder.build_checkpoint(
        state,
        memory,
        checkpoint_id="checkpoint_0002",
        reason="plan_updated",
    )

    assert first.fingerprint == second.fingerprint
    assert first.checkpoint_id != second.checkpoint_id
