"""Model Context V3 builder and registry contract tests."""

from __future__ import annotations

import json
from io import BytesIO

import pytest
from PIL import Image

from src.agent.mission_state import MissionState
from src.agent.planning.task_plan import TaskPlan
from src.memory.context.builder import ModelContextBuilder
from src.memory.context.contracts import ContextModuleResult, MemorySourceBundle
from src.memory.context.documents import ContextDocuments
from src.memory.context.registry import ContextModuleConfigError, ContextModuleRegistry
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MapMemoryStore, MemoryEventWriter, MemoryRecordReader
from src.uav.config import UavSafetyConfig


def _state(instruction: str = "inspect") -> MissionState:
    return MissionState("run", instruction, protocol_context="caller protocol")


def _memory(tmp_path, state: MissionState) -> tuple[FlightMemory, MemoryEventWriter]:
    writer = MemoryEventWriter(tmp_path)
    memory = FlightMemory(
        "run", "map", "scene", "uav", memory_reader=MemoryRecordReader(tmp_path)
    )
    memory.update_instruction_ref(writer.write_instruction(state.user_instruction), state.user_instruction)
    return memory, writer


def _builder(**kwargs) -> ModelContextBuilder:
    return ModelContextBuilder(
        registry=kwargs.pop("registry", ContextModuleRegistry.default()),
        documents=kwargs.pop(
            "documents",
            ContextDocuments(identity={"name": "FlyAgent"}, user_preferences={"language": "zh"}),
        ),
        map_memory=kwargs.pop("map_memory", MapMemoryStore()),
        **kwargs,
    )


def _record_observation(
    writer: MemoryEventWriter,
    memory: FlightMemory,
    state: MissionState,
    *,
    frame_path=None,
) -> str:
    frames = {}
    if frame_path is not None:
        frames["rgb"] = {
            "artifact_path": str(frame_path),
            "width": 4,
            "height": 3,
            "mock": False,
        }
    observation = {
        "source": "test",
        "modalities": ["rgb", "pose"] if frames else ["pose"],
        "pose": {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "yaw_deg": 10.0,
            "agl_m": 2.5,
            "coordinate_frame": "canonical_world_m",
        },
        "frames": frames,
    }
    refs = writer.write_observation(observation)
    observation_id = refs["observation_ref"]["id"]
    observation["observation_id"] = observation_id
    memory.append_observation_ref(refs["observation_ref"])
    memory.append_pose_ref(refs["pose_ref"])
    memory.append_frame_refs(refs["frame_refs"])
    state.record_observation(observation)
    return observation_id


def test_instruction_envelope_owns_static_content_without_raw_instruction(tmp_path):
    state = _state("find the entrance")
    memory, _ = _memory(tmp_path, state)
    builder = _builder(knowledge_catalog=[{"name": "guide", "description": "Guide"}])

    envelope = builder.build_instruction_envelope(state, memory).to_dict()

    assert envelope["schema_version"] == "instruction_envelope_v3"
    assert envelope["instruction_id"] == memory.active_instruction_id
    assert envelope["protocol_context"] == "caller protocol"
    assert envelope["identity"]["name"] == "FlyAgent"
    assert envelope["knowledge_catalog"] == [{"name": "guide", "description": "Guide"}]
    assert "find the entrance" not in json.dumps(envelope)
    assert [item["name"] for item in envelope["static_module_guidance"]] == [
        "skills", "image", "pose", "motion", "map"
    ]


def test_checkpoint_contains_only_dynamic_model_fields_and_separate_audit(tmp_path):
    state = _state()
    state.task_plan = TaskPlan().updated(
        {
            "explanation": "inspect safely",
            "steps": [{"id": "look", "description": "Look", "status": "in_progress",
                       "evidence_refs": []}],
        }
    )
    state.loaded_knowledge = [{"name": "guide", "version": "1"}]
    memory, writer = _memory(tmp_path, state)
    observation_id = _record_observation(writer, memory, state)

    checkpoint, audits = _builder().build_checkpoint(
        state, memory, checkpoint_id="checkpoint_0001", reason="instruction_start"
    )
    payload = checkpoint.to_dict()

    assert payload["schema_version"] == "state_checkpoint_v3"
    assert payload["observation_id"] == observation_id
    assert "task_plan" not in payload
    assert "loaded_knowledge" not in payload
    assert [item["name"] for item in payload["fragments"]] == [
        "skills", "image", "pose", "motion", "map"
    ]
    assert all(set(item) == {"name", "content", "media_refs"} for item in payload["fragments"])
    assert all("source" in item and "build_duration_ms" in item for item in audits)
    assert "data:image/" not in json.dumps(payload)


def test_motion_fragment_keeps_dynamic_safety_and_observation_policy(tmp_path):
    state = _state()
    state.observation_modalities = ("rgb", "pose")
    state.yaw_scan_extent_deg = 15.0
    memory, writer = _memory(tmp_path, state)
    _record_observation(writer, memory, state)
    builder = _builder(safety_config=UavSafetyConfig(max_yaw_scan_extent_deg=45.0))

    checkpoint, _ = builder.build_checkpoint(
        state, memory, checkpoint_id="checkpoint_0001", reason="instruction_start"
    )
    motion = next(item for item in checkpoint.fragments if item.name == "motion")

    assert motion.content["observation_policy"]["modalities"] == ["rgb", "pose"]
    assert motion.content["safety_constraints"]["yaw_scan_remaining_deg"] == 30.0


def test_image_fragment_is_media_ref_and_does_not_embed_base64(tmp_path):
    output = BytesIO()
    Image.new("RGB", (4, 3), (20, 30, 40)).save(output, format="PNG")
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(output.getvalue())
    state = _state()
    memory, writer = _memory(tmp_path, state)
    observation_id = _record_observation(
        writer, memory, state, frame_path=frame_path
    )

    checkpoint, _ = _builder().build_checkpoint(
        state, memory, checkpoint_id="checkpoint_0001", reason="instruction_start"
    )
    image = next(item for item in checkpoint.fragments if item.name == "image")
    ref = image.media_refs[0]["media_ref"]

    assert ref["ref"] == "frame.png"
    assert ref["observation_id"] == observation_id
    assert len(ref["sha256"]) == 64
    assert "data:image/" not in json.dumps(image.to_dict())


def test_missing_current_rgb_never_reuses_older_frame(tmp_path):
    output = BytesIO()
    Image.new("RGB", (4, 3), (1, 2, 3)).save(output, format="PNG")
    frame_path = tmp_path / "old.png"
    frame_path.write_bytes(output.getvalue())
    state = _state()
    memory, writer = _memory(tmp_path, state)
    old_id = _record_observation(writer, memory, state, frame_path=frame_path)
    current_id = _record_observation(writer, memory, state)

    checkpoint, _ = _builder().build_checkpoint(
        state, memory, checkpoint_id="checkpoint_0002", reason="physical_window_completed"
    )
    image = next(item for item in checkpoint.fragments if item.name == "image")

    assert current_id != old_id
    assert image.media_refs == ()
    assert image.content == {"visual_evidence": "current_rgb_unavailable"}


def test_registry_returns_module_results_and_ignores_disabled_modules(tmp_path):
    state = _state()
    memory, _ = _memory(tmp_path, state)
    registry = ContextModuleRegistry.default()
    registry.disabled.append("map")
    results = registry.build_fragments(
        MemorySourceBundle(memory),
        _builder(registry=registry)._stage_context_from_state(state),
    )

    assert all(isinstance(result, ContextModuleResult) for result in results)
    assert "map" not in [result.fragment.name for result in results]
    assert "map" not in registry.describe_modules()


def test_registry_rejects_unconstructed_plugin_provider():
    with pytest.raises(ContextModuleConfigError, match="did not supply plugin"):
        ContextModuleRegistry.from_dict({"context_modules": {"map": "research.map"}})


def test_semantic_evidence_capabilities_are_deduplicated_and_bounded():
    registry = ContextModuleRegistry.default()
    capabilities = registry.semantic_evidence_capabilities()

    assert capabilities
    assert len(capabilities) <= registry.MAX_SEMANTIC_EVIDENCE_ITEMS
    assert len(capabilities) == len(set(capabilities))
    assert all(len(item) <= registry.MAX_SEMANTIC_EVIDENCE_CHARS for item in capabilities)
