"""V2 history selection and lazy media projection tests."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from src.agent.context.media import MediaHydrator
from src.agent.context.projection import HistoryProjector
from src.agent.context.request_assembler import RequestAssembler
from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn, ToolResult
from src.memory.context.contracts import (
    ContextFragment,
    HistoryImageCandidate,
    InstructionEnvelope,
    MediaRef,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.context.modules.base import ContextModule
from src.memory.context.modules.image import DefaultImageContextModule
from src.memory.stores import MemoryEventWriter
from src.tools.registry import default_registry


def _transcript(tmp_path) -> AgentTranscript:
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction(
        "inspect",
        "inst_1",
        InstructionEnvelope(
            instruction_id="inst_1",
            protocol_context="protocol",
            identity={},
            user_preferences={},
            knowledge_catalog=(),
            static_module_guidance=(),
        ),
    )
    return transcript


def _add_image_checkpoint(transcript, tmp_path, index: int) -> MediaRef:
    output = BytesIO()
    Image.new("RGB", (2, 2), (index, index, index)).save(output, format="PNG")
    raw = output.getvalue()
    path = tmp_path / f"frame_{index}.png"
    path.write_bytes(raw)
    observation_id = f"obs_{index:04d}"
    ref = MediaRef(
        ref=path.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        mime_type="image/png",
        observation_id=observation_id,
        frame_id=f"frame_{index}",
        role="current_observation",
    )
    fragment = ContextFragment("image", {}, (ref.to_model_input(),))
    transcript.append_state(
        StateCheckpoint(
            checkpoint_id=f"checkpoint_{index:04d}",
            reason="instruction_start" if index == 1 else "physical_window_completed",
            observation_id=observation_id,
            task_plan=None,
            loaded_knowledge=(),
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint(observation_id, None, (), (fragment,)),
        )
    )
    return ref


def test_recency_log_selector_is_identity_only_and_bounded():
    module = DefaultImageContextModule()
    candidates = tuple(
        HistoryImageCandidate(f"obs_{index}", index, (f"frame_{index}",))
        for index in range(20)
    )
    selected = module.select_history_images(
        candidates,
        {
            "mode": "recency_log",
            "max_observation_images": 6,
            "keep_recent_observations": 2,
        },
    )

    assert len(selected) == 6
    assert selected[0] == "obs_0"
    assert selected[-2:] == ("obs_18", "obs_19")


def test_projector_reads_and_encodes_only_policy_selected_media(tmp_path):
    transcript = _transcript(tmp_path)
    for index in range(1, 11):
        _add_image_checkpoint(transcript, tmp_path, index)
    hydrator = MediaHydrator(base_dir=tmp_path)
    module = DefaultImageContextModule()
    projector = HistoryProjector(
        {
            "image": (
                module,
                {
                    "mode": "recency_log",
                    "max_observation_images": 4,
                    "keep_recent_observations": 1,
                },
            )
        },
        assembler=RequestAssembler(hydrator),
    )

    result = projector.project(transcript, default_registry())

    assert hydrator.stats()["reads"] == 4
    assert hydrator.stats()["encodes"] == 4
    assert result.audit["observation_image_count"] == 4
    assert result.audit["policies"]["image"]["omitted_count"] == 6


def test_same_sha_is_encoded_once_per_instruction_cache(tmp_path):
    transcript = _transcript(tmp_path)
    ref = _add_image_checkpoint(transcript, tmp_path, 1)
    fragment = ContextFragment(
        "image",
        {},
        (
            MediaRef(
                **{**ref.to_dict(), "observation_id": "obs_0002", "frame_id": "frame_2"}
            ).to_model_input(),
        ),
    )
    transcript.append_state(
        StateCheckpoint(
            checkpoint_id="checkpoint_0002",
            reason="physical_window_completed",
            observation_id="obs_0002",
            task_plan=None,
            loaded_knowledge=(),
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint("obs_0002", None, (), (fragment,)),
        )
    )
    hydrator = MediaHydrator(base_dir=tmp_path)
    projector = HistoryProjector(
        {"image": (DefaultImageContextModule(), {"mode": "all"})},
        assembler=RequestAssembler(hydrator),
    )

    result = projector.project(transcript, default_registry())

    assert hydrator.stats()["reads"] == 1
    assert hydrator.stats()["encodes"] == 1
    assert hydrator.stats()["hits"] == 1
    assert result.audit["observation_image_count"] == 2


def test_projection_does_not_mutate_transcript_or_return_shared_objects(tmp_path):
    transcript = _transcript(tmp_path)
    _add_image_checkpoint(transcript, tmp_path, 1)
    before = transcript.to_dict()
    result = HistoryProjector(
        assembler=RequestAssembler(MediaHydrator(base_dir=tmp_path))
    ).project(transcript, default_registry())
    result.messages[0]["content"] = "mutated"

    assert transcript.to_dict() == before
    assert transcript.provider_messages(default_registry())[0]["content"] == "inspect"


def test_invalid_module_selection_is_rejected(tmp_path):
    class InvalidSelector(DefaultImageContextModule):
        def select_history_images(self, candidates, config):
            return ("not_an_observation",)

    transcript = _transcript(tmp_path)
    _add_image_checkpoint(transcript, tmp_path, 1)

    with pytest.raises(ValueError, match="Invalid history selection"):
        HistoryProjector({"image": (InvalidSelector(), {"mode": "all"})}).project(
            transcript, default_registry()
        )


def test_missing_selected_media_is_explicit_and_never_substitutes_old_image(tmp_path):
    transcript = _transcript(tmp_path)
    _add_image_checkpoint(transcript, tmp_path, 1)
    missing_raw = b"missing"
    missing = MediaRef(
        ref="does_not_exist.png",
        sha256=hashlib.sha256(missing_raw).hexdigest(),
        mime_type="image/png",
        observation_id="obs_0002",
        frame_id="frame_2",
        role="current_observation",
    )
    fragment = ContextFragment("image", {}, (missing.to_model_input(),))
    transcript.append_state(
        StateCheckpoint(
            checkpoint_id="checkpoint_0002",
            reason="physical_window_completed",
            observation_id="obs_0002",
            task_plan=None,
            loaded_knowledge=(),
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint("obs_0002", None, (), (fragment,)),
        )
    )

    result = HistoryProjector(
        assembler=RequestAssembler(MediaHydrator(base_dir=tmp_path))
    ).project(transcript, default_registry())
    rendered = json.dumps(result.messages, ensure_ascii=False)

    assert rendered.count("data:image/png;base64,") == 1
    assert "CURRENT observation_id=obs_0002" not in rendered
    assert "no older image substitutes CURRENT" in rendered
    assert result.audit["missing_images"]


def test_global_media_budget_includes_observation_and_tool_evidence(tmp_path):
    transcript = _transcript(tmp_path)
    _add_image_checkpoint(transcript, tmp_path, 1)
    calls = []
    results = []
    for index in range(25):
        output = BytesIO()
        Image.new("RGB", (2, 2), (index, 0, 0)).save(output, format="PNG")
        raw = output.getvalue()
        path = tmp_path / f"tool_{index}.png"
        path.write_bytes(raw)
        call_id = f"tool_{index}"
        ref = MediaRef(
            ref=path.name,
            sha256=hashlib.sha256(raw).hexdigest(),
            mime_type="image/png",
            observation_id=f"tool_obs_{index}",
            frame_id=f"tool_frame_{index}",
            role="tool_evidence",
        )
        calls.append(AgentToolCall(call_id, "uav.turn_relative", {"yaw_deg": 1}))
        results.append(
            ToolResult(
                call_id,
                "uav.turn_relative",
                "ok",
                model_inputs=[ref.to_model_input()],
            )
        )
    transcript.append_assistant(AssistantTurn("turn", "Inspect", calls))
    for result in results:
        transcript.append_tool_result(result)
    projector = HistoryProjector(
        assembler=RequestAssembler(MediaHydrator(base_dir=tmp_path)),
        media_policy={
            "max_request_images": 20,
            "keep_recent_visual_events": 3,
            "pin_instruction_start": True,
            "pin_current": True,
        },
    )

    projected = projector.project(transcript, default_registry())
    rendered = json.dumps(projected.messages, ensure_ascii=False)

    assert projected.audit["total_image_count"] == 20
    assert projected.audit["observation_image_count"] == 1
    assert projected.audit["tool_image_count"] == 19
    assert rendered.count("data:image/png;base64,") == 20
