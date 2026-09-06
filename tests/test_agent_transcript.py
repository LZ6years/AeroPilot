"""Tests for the canonical chronological Agent transcript."""

from __future__ import annotations

import json

from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn, ToolResult
from src.memory.stores import MemoryEventWriter
from src.tools.registry import default_registry


def test_transcript_projects_state_calls_and_results_in_chronological_order(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("find the person", "inst_0001")
    transcript.append_state(
        {
            "current_phase": {"name": "find_person", "done_when": ["person found"]},
            "allowed_action_constraints": {"planner_tools": ["uav.hover"]},
            "runtime_control": {
                "available": ["semantic_plan_update"],
                "contracts": {"semantic_plan_update": "duplicate schema"},
                "status": {"semantic_plan_update": "pending"},
            },
            "fragments": [],
        },
        "obs_0001",
    )
    transcript.append_assistant(
        AssistantTurn(
            "assistant_1",
            "Confirm the current view.",
            [AgentToolCall("call_detect", "uav.hover", {})],
        )
    )
    transcript.append_tool_result(
        ToolResult("call_detect", "uav.hover", "ok", "stable")
    )

    messages = transcript.provider_messages(default_registry())

    assert [message["role"] for message in messages] == [
        "user",
        "user",
        "assistant",
        "tool",
    ]
    state_json = json.loads(messages[1]["content"][-1]["text"])
    assert "allowed_action_constraints" not in state_json
    assert state_json["runtime_control"] == {
        "status": {"semantic_plan_update": "pending"}
    }
    assert messages[2]["tool_calls"][0]["id"] == "call_detect"
    assert json.loads(messages[3]["content"])["status"] == "ok"
    assert transcript.latest_observation_id() == "obs_0001"


def test_transcript_keeps_images_in_memory_but_strips_binary_from_jsonl(tmp_path):
    image_data = "data:image/png;base64,AAAA"
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect", "inst_0001")
    transcript.append_state(
        {
            "current_phase": {"name": "inspect"},
            "fragments": [
                {
                    "name": "image",
                    "model_inputs": [
                        {
                            "type": "image",
                            "role": "current_observation",
                            "encoding": "data_url",
                            "data": image_data,
                            "observation_id": "obs_0001",
                            "frame_id": "frame_0001",
                            "artifact_ref": "memory/frames.jsonl#1",
                        }
                    ],
                }
            ],
        },
        "obs_0001",
    )

    messages = transcript.provider_messages(default_registry())
    transcript_lines = (tmp_path / "memory" / "agent_transcript.jsonl").read_text(
        encoding="utf-8"
    )

    assert any(
        part.get("image_url", {}).get("url") == image_data
        for part in messages[1]["content"]
        if isinstance(part, dict)
    )
    assert image_data not in transcript_lines
    assert "[provider_media_rehydrated_from_artifact]" in transcript_lines
    assert "observation_id=obs_0001" in json.dumps(messages[1], ensure_ascii=False)
    assert "frame_id=frame_0001" in json.dumps(messages[1], ensure_ascii=False)


def test_transcript_sends_each_identified_image_once_at_its_first_checkpoint(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect over time", "inst_0001")

    def context(current_id: str, current_data: str):
        return {
            "current_phase": {"name": "inspect"},
            "fragments": [
                {
                    "name": "image",
                    "model_inputs": [
                        {
                            "type": "image",
                            "role": "phase_start_reference",
                            "encoding": "data_url",
                            "data": "data:image/png;base64,PAST",
                            "observation_id": "obs_0001",
                            "frame_id": "frame_0001",
                            "artifact_ref": "memory/frames.jsonl#1",
                        },
                        {
                            "type": "image",
                            "role": "current_observation",
                            "encoding": "data_url",
                            "data": current_data,
                            "observation_id": current_id,
                            "frame_id": f"frame_{current_id}",
                            "artifact_ref": f"memory/frames.jsonl#{current_id}",
                        },
                    ],
                }
            ],
        }

    transcript.append_state(context("obs_0001", "data:image/png;base64,PAST"), "obs_0001")
    transcript.append_state(context("obs_0002", "data:image/png;base64,NOW"), "obs_0002")

    rendered = json.dumps(
        transcript.provider_messages(default_registry()), ensure_ascii=False
    )

    assert rendered.count("data:image/png;base64,PAST") == 1
    assert rendered.count("data:image/png;base64,NOW") == 1


def test_fallback_intent_is_audited_but_not_fabricated_for_provider(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect", "inst_0001")
    transcript.append_assistant(
        AssistantTurn(
            "assistant_1",
            "Request uav.get_pose for observation obs_0001.",
            [AgentToolCall("call_pose", "uav.get_pose", {})],
            public_intent_source="tool_arguments_fallback",
        )
    )
    transcript.append_tool_result(
        ToolResult("call_pose", "uav.get_pose", "ok", "pose read")
    )

    messages = transcript.provider_messages(default_registry())
    assistant = next(item for item in messages if item["role"] == "assistant")

    assert assistant["content"] is None
    assert transcript.to_dict()["items"][1]["payload"] == {
        "turn_id": "assistant_1",
        "public_intent": "Request uav.get_pose for observation obs_0001.",
        "public_intent_source": "tool_arguments_fallback",
        "tool_calls": [
            {"tool_call_id": "call_pose", "tool_name": "uav.get_pose", "args": {}}
        ],
    }


def test_provider_rejection_is_persisted_but_omitted_from_provider_projection(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect", "inst_0001")
    transcript.append_provider_rejection(
        {
            "error": "unknown provider tool",
            "finish_reason": "tool_calls",
            "content_present": False,
            "content": None,
            "tool_calls": [
                {
                    "tool_call_id": "bad_call",
                    "wire_name": "unknown_tool",
                    "arguments": {},
                    "arguments_status": "parsed",
                }
            ],
        }
    )
    transcript.append_protocol_feedback("Return a corrected ToolCall.")

    messages = transcript.provider_messages(default_registry())
    records = transcript.to_dict()["items"]

    assert [item["role"] for item in messages] == ["user", "user"]
    assert messages[-1]["content"] == "Return a corrected ToolCall."
    assert records[1]["kind"] == "provider_turn_rejected"
