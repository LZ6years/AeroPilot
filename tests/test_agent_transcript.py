"""Tests for the binary-free V2 chronological Agent transcript."""

from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn, ToolResult
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    MediaRef,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.context.media_refs import image_media_ref
from src.memory.stores import MemoryEventWriter
from src.tools.registry import default_registry


def _start(transcript: AgentTranscript, instruction: str = "inspect") -> None:
    transcript.start_instruction(
        instruction,
        "inst_0001",
        InstructionEnvelope(
            instruction_id="inst_0001",
            protocol_context="protocol",
            identity={},
            user_preferences={},
            knowledge_catalog=(),
            static_module_guidance=(),
        ),
    )


def _append_checkpoint(
    transcript: AgentTranscript,
    observation_id: str,
    *,
    inputs: tuple[dict, ...] = (),
    index: int = 1,
) -> StateCheckpoint:
    fragment = ContextFragment("image", {"index": index}, inputs)
    checkpoint = StateCheckpoint(
        checkpoint_id=f"checkpoint_{index:04d}",
        reason="instruction_start" if index == 1 else "physical_window_completed",
        observation_id=observation_id,
        task_plan=None,
        loaded_knowledge=(),
        fragments=(fragment,),
        fingerprint=checkpoint_fingerprint(observation_id, None, (), (fragment,)),
    )
    transcript.append_state(checkpoint)
    return checkpoint


def _media(tmp_path, observation_id: str, color: tuple[int, int, int]) -> MediaRef:
    output = BytesIO()
    Image.new("RGB", (2, 2), color).save(output, format="PNG")
    raw = output.getvalue()
    path = tmp_path / f"{observation_id}.png"
    path.write_bytes(raw)
    return MediaRef(
        ref=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        mime_type="image/png",
        observation_id=observation_id,
        frame_id=f"frame_{observation_id}",
        role="current_observation",
        width=2,
        height=2,
        display_width=2,
        display_height=3,
        color_bar_height=1,
    )


def test_transcript_projects_checkpoints_calls_and_results_in_order(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    _start(transcript, "find the person")
    checkpoint = _append_checkpoint(transcript, "obs_0001")
    transcript.append_assistant(
        AssistantTurn(
            "assistant_1",
            "Confirm the current view.",
            [AgentToolCall("call_detect", "uav.hover", {})],
        )
    )
    transcript.append_tool_result(ToolResult("call_detect", "uav.hover", "ok", "stable"))

    messages = transcript.provider_messages(default_registry())

    assert [message["role"] for message in messages] == [
        "user", "user", "user", "assistant", "tool"
    ]
    state_json = json.loads(messages[2]["content"][-1]["text"])
    assert state_json["schema_version"] == "state_checkpoint_v2"
    assert transcript.latest_checkpoint() == checkpoint
    assert messages[3]["tool_calls"][0]["id"] == "call_detect"
    assert json.loads(messages[4]["content"])["status"] == "ok"
    assert transcript.latest_observation_id() == "obs_0001"


def test_transcript_rejects_inline_images_and_persists_media_refs_only(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    _start(transcript)
    ref = _media(tmp_path, "obs_0001", (1, 2, 3))
    _append_checkpoint(transcript, "obs_0001", inputs=(ref.to_model_input(),))

    transcript_text = (tmp_path / "memory" / "agent_transcript.jsonl").read_text(
        encoding="utf-8"
    )
    messages = transcript.provider_messages(default_registry())

    assert "data:image/" not in transcript_text
    assert ref.sha256 in transcript_text
    assert "data:image/png;base64," in json.dumps(messages)
    with pytest.raises(ValueError, match="inline image data"):
        transcript.append_tool_result(
            ToolResult(
                "bad",
                "uav.hover",
                "ok",
                model_inputs=[{"type": "image", "data": "data:image/png;base64,AAAA"}],
            )
        )


def test_transcript_projects_each_observation_image_once_in_time_order(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    _start(transcript, "inspect over time")
    refs = [
        _media(tmp_path, "obs_0001", (255, 0, 0)),
        _media(tmp_path, "obs_0002", (0, 255, 0)),
        _media(tmp_path, "obs_0003", (0, 0, 255)),
    ]
    for index, ref in enumerate(refs, 1):
        _append_checkpoint(
            transcript, ref.observation_id, inputs=(ref.to_model_input(),), index=index
        )

    rendered = json.dumps(transcript.provider_messages(default_registry()), ensure_ascii=False)

    assert rendered.count("data:image/png;base64,") == 3
    assert rendered.count("HISTORY observation_id=obs_0001") == 1
    assert rendered.count("HISTORY observation_id=obs_0002") == 1
    assert rendered.count("CURRENT observation_id=obs_0003") == 1
    assert rendered.count("scene_size=2x2") == 3
    assert rendered.count("bottom color bar is not scene content") == 3


def test_repeated_observation_media_is_sent_at_first_checkpoint_only(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    _start(transcript, "measure")
    ref = _media(tmp_path, "obs_0001", (10, 20, 30))
    _append_checkpoint(transcript, "obs_0001", inputs=(ref.to_model_input(),), index=1)
    transcript.append_assistant(AssistantTurn("turn_1", "Measure.", []))
    _append_checkpoint(transcript, "obs_0001", inputs=(ref.to_model_input(),), index=2)

    messages = transcript.provider_messages(default_registry())
    rendered = json.dumps(messages, ensure_ascii=False)

    assert rendered.count("data:image/png;base64,") == 1
    assert "CURRENT observation_id=obs_0001" in rendered
    first_state = next(message for message in messages if isinstance(message.get("content"), list))
    assert "data:image/png;base64," in json.dumps(first_state)


def test_tool_result_provider_projection_hides_internal_audit_details():
    result = ToolResult(
        "call_depth",
        "skill.metric_depth.sample_region",
        "ok",
        data={
            "observation_id": "obs_0001",
            "distance_m": 2.9,
            "audit_details": {"mad_m": 0.07},
        },
    )
    assert result.to_dict()["data"]["audit_details"] == {"mad_m": 0.07}
    assert result.to_provider_dict()["data"] == {
        "observation_id": "obs_0001",
        "distance_m": 2.9,
    }


def test_tool_result_projects_media_ref_visual_evidence_after_tool_role(tmp_path):
    evidence_path = tmp_path / "grounding_candidates.png"
    evidence_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )
    evidence_path.write_bytes(evidence_bytes)
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    _start(transcript, "measure")
    transcript.append_assistant(
        AssistantTurn(
            "turn_depth",
            "Measure the selected surface.",
            [AgentToolCall("call_depth", "uav.hover", {})],
        )
    )
    result = ToolResult(
        "call_depth",
        "uav.hover",
        "ok",
        data={"observation_id": "obs_0001", "distance_m": 2.9},
        artifact_refs=[str(evidence_path)],
        model_inputs=[
            image_media_ref(
                evidence_path,
                observation_id="obs_0001",
                frame_id="grounding_candidates",
                role="visual_grounding_candidates",
                usage="Confirm candidate_01 against the instruction.",
            ).to_model_input()
        ],
    )
    transcript.append_tool_result(result)

    messages = transcript.provider_messages(default_registry())

    assert [message["role"] for message in messages] == [
        "user", "user", "assistant", "tool", "user"
    ]
    assert json.loads(messages[3]["content"])["data"]["distance_m"] == 2.9
    assert "model_inputs" not in json.loads(messages[3]["content"])
    assert "TOOL EVIDENCE tool_call_id=call_depth" in messages[4]["content"][0]["text"]
    assert messages[4]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_terminal_event_is_audit_only_and_prior_refs_exclude_current_turn(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    _start(transcript)
    _append_checkpoint(transcript, "obs_0001")
    transcript.append_assistant(
        AssistantTurn(
            "assistant_1", "Need evidence.", [AgentToolCall("call_prior", "uav.hover", {})]
        )
    )
    transcript.append_tool_result(
        ToolResult("call_prior", "uav.hover", "ok", result_ref="memory/actions.jsonl#1")
    )
    transcript.append_assistant(
        AssistantTurn(
            "assistant_2", "Blocked.", [AgentToolCall("call_terminal", "runtime.report_blocked", {})]
        )
    )
    prior_refs = transcript.prior_evidence_reference_ids()
    transcript.append_task_terminal({"stopped": True})

    assert prior_refs == {"obs_0001", "call_prior", "memory/actions.jsonl#1"}
    assert "task_terminal" not in json.dumps(
        transcript.provider_messages(default_registry()), ensure_ascii=False
    )


def test_fallback_intent_is_audited_but_not_fabricated_for_provider(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    _start(transcript)
    transcript.append_assistant(
        AssistantTurn(
            "assistant_1",
            "Request uav.get_pose for observation obs_0001.",
            [AgentToolCall("call_pose", "uav.get_pose", {})],
            public_intent_source="tool_arguments_fallback",
        )
    )
    transcript.append_tool_result(ToolResult("call_pose", "uav.get_pose", "ok", "pose read"))

    messages = transcript.provider_messages(default_registry())
    assistant = next(item for item in messages if item["role"] == "assistant")
    records = transcript.to_dict()["items"]

    assert assistant["content"] is None
    assert next(item for item in records if item["kind"] == "assistant_turn")["payload"][
        "public_intent_source"
    ] == "tool_arguments_fallback"


def test_provider_rejection_is_persisted_but_omitted_from_projection(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    _start(transcript)
    transcript.append_provider_rejection({"error": "unknown provider tool"})
    transcript.append_protocol_feedback("Return a corrected ToolCall.")

    messages = transcript.provider_messages(default_registry())
    records = transcript.to_dict()["items"]

    assert [item["role"] for item in messages] == ["user", "user", "user"]
    assert messages[-1]["content"] == "Return a corrected ToolCall."
    assert any(item["kind"] == "provider_turn_rejected" for item in records)
