"""V2 history selection and lazy media projection tests."""

from __future__ import annotations

import hashlib
import copy
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
from src.agent.planning.task_plan import TaskPlan


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


def test_plan_receipts_are_compact_but_native_arguments_and_durable_history_are_exact(tmp_path):
    transcript = _transcript(tmp_path)
    args = {"operation": "replace", "explanation": None, "steps": [{"id": "gate", "description": "Reach the gate", "status": "in_progress"}]}
    plan = TaskPlan().updated(args)
    result = ToolResult("plan", "runtime.update_plan", "ok", data={
        "previous_revision": 0, "plan": plan.to_dict(), "changed_step_ids": ["gate"],
    }, requested_args=copy.deepcopy(args), executed_args=copy.deepcopy(args))
    transcript.append_assistant(AssistantTurn("turn", "Track route", [AgentToolCall("plan", "runtime.update_plan", args)]))
    transcript.append_tool_result(result)
    transcript.append_assistant(AssistantTurn("bad", "Repair plan", [
        AgentToolCall("bad_plan", "runtime.update_plan", {"operation": "patch", "explanation": None, "steps": [{"description": None, "id": "missing", "status": "completed"}]})
    ]))
    rejected = ToolResult("bad_plan", "runtime.update_plan", "rejected", "unknown step",
                          requested_args={"operation": "patch", "explanation": None, "steps": [{"description": None, "id": "missing", "status": "completed"}]},
                          data={"rejection": {"source_tool_call_id": "bad_plan"}})
    transcript.append_tool_result(rejected)
    before = transcript.to_dict()
    projection = HistoryProjector().project(transcript, default_registry())
    calls = [m for m in projection.messages if m["role"] == "assistant"]
    assert json.loads(calls[0]["tool_calls"][0]["function"]["arguments"]) == args
    receipts = [json.loads(m["content"]) for m in projection.messages if m["role"] == "tool"]
    assert receipts[0]["data"] == {"previous_revision": 0, "revision": 1, "changed_step_ids": ["gate"]}
    assert "requested_args" not in receipts[0] and "executed_args" not in receipts[0]
    expected = rejected.to_provider_dict()
    expected.pop("requested_args")
    assert receipts[1] == expected
    assert transcript.to_dict() == before
    assert len(json.dumps(receipts[0])) < len(json.dumps(result.to_provider_dict()))


def test_current_camera_label_does_not_exclude_historical_route_evidence():
    from src.agent.context.request_assembler import _media_label
    label = _media_label({"observation_id": "obs_2", "frame_id": "frame_2"}, "CURRENT")
    assert "Only CURRENT establishes current camera visibility" in label
    assert "next action" not in label
    context = _media_label({"observation_id": "obs_2", "frame_id": "map", "latest_group": "places",
                            "usage": "Measured path and model-authored labels, not camera visibility."}, "CURRENT")
    assert context.startswith("CONTEXT IMAGE")
    assert "Only CURRENT" not in context


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
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint(observation_id, (fragment,)),
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


def test_retrieved_old_photos_keep_latest_tool_priority_and_unique_anchors(tmp_path):
    transcript = _transcript(tmp_path)
    refs = {index: _add_image_checkpoint(transcript, tmp_path, index) for index in range(1, 41)}
    projector = HistoryProjector(
        assembler=RequestAssembler(MediaHydrator(base_dir=tmp_path)),
        media_policy={"max_request_images": 20, "max_tool_evidence_images": 4,
                      "keep_recent_observations": 3, "pin_instruction_start": True, "pin_current": True},
    )
    initial = projector.project(transcript, default_registry()).audit["selected_visual_evidence"]
    absent = [i for i in range(2, 30) if refs[i].observation_id not in {v["observation_id"] for v in initial}]
    assert len(absent) >= 5
    # Old retrievals compete with an explicit re-read; duplicate pinned images
    # must neither steal evidence slots nor change the current camera source.
    for number, indices in enumerate((absent[:2], absent[2:5], [1, 40, *absent[:2]])):
        call_id = f"retrieve_{number}"
        transcript.append_assistant(AssistantTurn(call_id, "Compare earlier views", [
            AgentToolCall(call_id, "uav.turn_relative", {"yaw_deg": 1})]))
        transcript.append_tool_result(ToolResult(call_id, "uav.turn_relative", "ok",
            model_inputs=[refs[i].to_model_input() for i in indices]))
    before = transcript.to_dict()
    projection = projector.project(transcript, default_registry())
    selected = projection.audit["selected_visual_evidence"]
    assert projection.audit["total_image_count"] == 20
    assert len({v["observation_id"] for v in selected}) == 20
    for index in absent[:2]:
        value = next(v for v in selected if v["observation_id"] == refs[index].observation_id)
        assert value["source_kind"] == "tool_evidence"
        assert value["selection_tier"] == "tail_tool_evidence"
        assert value["event_index"] == transcript.event_count - 1
    assert projection.audit["tool_image_count"] == 4
    for index in (1, 40):
        assert next(v for v in selected if v["observation_id"] == refs[index].observation_id)["source_kind"] == "observation"
    assert transcript.to_dict() == before


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
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint("obs_0002", (fragment,)),
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
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint("obs_0002", (fragment,)),
        )
    )

    result = HistoryProjector(
        assembler=RequestAssembler(MediaHydrator(base_dir=tmp_path))
    ).project(transcript, default_registry())
    rendered = json.dumps(result.messages, ensure_ascii=False)

    assert rendered.count("data:image/png;base64,") == 1
    assert "CURRENT observation_id=obs_0002" not in rendered
    assert "No older image establishes CURRENT visibility" in rendered
    assert "STATE CHECKPOINT CURRENT" in rendered
    assert "Published image media is unavailable" in rendered
    assert result.audit["missing_images"]


def test_global_media_budget_caps_tool_evidence_without_crowding_observation(tmp_path):
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
            "max_tool_evidence_images": 4,
            "keep_recent_observations": 3,
            "pin_instruction_start": True,
            "pin_current": True,
        },
    )

    projected = projector.project(transcript, default_registry())
    rendered = json.dumps(projected.messages, ensure_ascii=False)

    assert projected.audit["total_image_count"] == 5
    assert projected.audit["observation_image_count"] == 1
    assert projected.audit["tool_image_count"] == 4
    assert rendered.count("data:image/png;base64,") == 5
    selected_tools = [
        item
        for item in projected.audit["selected_visual_evidence"]
        if item["source_kind"] == "tool_evidence"
    ]
    assert len(selected_tools) == 4
    assert {item["selection_tier"] for item in selected_tools} == {
        "tail_tool_evidence"
    }


def test_layered_media_budget_keeps_start_current_recent_and_log_history(tmp_path):
    transcript = _transcript(tmp_path)
    for index in range(1, 26):
        _add_image_checkpoint(transcript, tmp_path, index)
    calls = []
    results = []
    for index in range(8):
        output = BytesIO()
        Image.new("RGB", (2, 2), (0, index, 0)).save(output, format="PNG")
        raw = output.getvalue()
        path = tmp_path / f"evidence_{index}.png"
        path.write_bytes(raw)
        call_id = f"evidence_{index}"
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
            "max_tool_evidence_images": 4,
            "keep_recent_observations": 3,
            "pin_instruction_start": True,
            "pin_current": True,
        },
    )

    projected = projector.project(transcript, default_registry())
    selected = projected.audit["selected_visual_evidence"]
    observation_ids = {
        item["observation_id"]
        for item in selected
        if item["source_kind"] == "observation"
    }

    assert projected.audit["total_image_count"] == 20
    assert projected.audit["observation_image_count"] == 16
    assert projected.audit["tool_image_count"] == 4
    assert {"obs_0001", "obs_0022", "obs_0023", "obs_0024", "obs_0025"} <= observation_ids
    assert any(item["selection_tier"] == "log_history" for item in selected)
